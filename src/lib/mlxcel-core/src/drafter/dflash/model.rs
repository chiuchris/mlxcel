// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! `DFlashDraftModel` — the assembled 5-layer Qwen 3.5 DFlash drafter.
//!
//! Builds on the [`super::attention::DFlashAttention`],
//! [`super::layer::DFlashDecoderLayer`], and [`super::mlp::DFlashMlp`]
//! blocks ported in earlier commits.
//!
//! Apple Silicon precision rules (see `docs/apple-silicon-precision.md`):
//! - Hidden, K, V, and projected proposal/context tensors stay in bf16/f16.
//! - No `f32` promotion in the forward path.
//! - Weights remain bf16 or quantized as-loaded; the non-quantized bf16 →
//!   f16 conversion happens in the binary's `load_and_sanitize_weights`
//!   path that ultimately hands the [`WeightMap`] to
//!   [`DFlashDraftModel::from_weights`].

use crate::cache::KVCache;
use crate::ffi::{self, MlxArray};
use crate::layers::{Linear, RMSNorm, UnifiedEmbedding};
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::config::DFlashConfig;
use super::layer::DFlashDecoderLayer;

/// LM head selector: own checkpoint weights or target-shared (tied).
///
/// - [`LmHead::Own`] — drafter checkpoint shipped a dedicated
///   `lm_head.weight`. Used as a regular Linear.
/// - [`LmHead::Tied`] — drafter uses `embed_tokens.as_linear` for the
///   final projection. This is the published `z-lab/Qwen3.5-4B-DFlash`
///   default (`tie_word_embeddings = true`).
enum LmHead {
    Own(Linear),
    Tied,
}

/// Qwen 3.5 DFlash drafter (`DFlashDraftModel`).
///
/// The end-to-end forward consumes a `[B, L]` int32 token sequence (the
/// proposal block) and a `[B, T, num_target_layers * hidden_size]`
/// target-hidden buffer, producing `[B, L, vocab_size]` logits.
///
/// One [`KVCache`] per drafter layer is owned by the caller; the
/// drafter writes only the context-side K/V into each cache via
/// [`super::attention::DFlashAttention::forward`].
pub struct DFlashDraftModel {
    pub config: DFlashConfig,

    /// Token embedding table. Always loaded from the drafter's own
    /// checkpoint at construction time. The published checkpoint ships
    /// `embed_tokens.weight` and uses it both as the input embedder and
    /// (via [`LmHead::Tied`]) as the LM head when
    /// `tie_word_embeddings = true`.
    pub embed_tokens: UnifiedEmbedding,

    /// Per-target-layer projection: maps the
    /// `5 * hidden_size`-dim concatenation to `hidden_size`.
    /// Bias-free per upstream.
    pub fc: Linear,

    /// Norm over the projected target hidden (before consumption by
    /// every decoder layer).
    pub hidden_norm: RMSNorm,

    /// Five drafter transformer layers.
    pub layers: Vec<DFlashDecoderLayer>,

    /// Final RMSNorm before the LM head.
    pub norm: RMSNorm,

    /// LM head dispatch: own vs tied.
    lm_head: LmHead,
}

impl DFlashDraftModel {
    /// Construct the drafter from a sanitized weight map.
    ///
    /// `weights` must already have the `model.` prefix stripped (see
    /// [`Self::sanitize`]). The drafter is dtype-agnostic — caller
    /// converts bf16 → f16 in the load pipeline per Apple Silicon
    /// precision rules.
    pub fn from_weights(weights: &WeightMap, config: DFlashConfig) -> Result<Self, String> {
        // DFlash uses non-quantized weights in the published checkpoint;
        // but `UnifiedLinear::from_weights` auto-detects quantization so
        // a future quantized DFlash will Just Work without further changes.
        // The group_size/bits fallback only matters if the checkpoint
        // *is* quantized; for the published bf16 checkpoint these
        // values are dead code.
        let group_size = 64;
        let bits = 4;

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "embed_tokens", group_size, bits)?;
        let fc = Linear::from_weights(weights, "fc")?;

        let hidden_norm_w = weights
            .get("hidden_norm.weight")
            .map(|w| ffi::copy(w))
            .ok_or_else(|| "Weight not found: hidden_norm.weight".to_string())?;
        let hidden_norm = RMSNorm::new(hidden_norm_w, config.rms_norm_eps);

        let norm_w = weights
            .get("norm.weight")
            .map(|w| ffi::copy(w))
            .ok_or_else(|| "Weight not found: norm.weight".to_string())?;
        let norm = RMSNorm::new(norm_w, config.rms_norm_eps);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("layers.{i}");
            layers.push(DFlashDecoderLayer::from_weights(
                weights, &prefix, &config, group_size, bits,
            )?);
        }

        // LM head: present iff the checkpoint ships `lm_head.weight`.
        // `tie_word_embeddings = true` (the default) means no `lm_head.weight`
        // and we route through `embed_tokens.as_linear` instead.
        let lm_head = if weights.contains_key("lm_head.weight") {
            LmHead::Own(Linear::from_weights(weights, "lm_head")?)
        } else if config.tie_word_embeddings {
            LmHead::Tied
        } else {
            return Err(
                "tie_word_embeddings=false but lm_head.weight is missing from drafter \
                 checkpoint; either set tie_word_embeddings=true or ship lm_head.weight"
                    .to_string(),
            );
        };

        Ok(Self {
            config,
            embed_tokens,
            fc,
            hidden_norm,
            layers,
            norm,
            lm_head,
        })
    }

    /// Construct fresh per-layer K/V caches, one per drafter layer.
    ///
    /// Each cache is an FP16 [`KVCache`] (the default mode). Mirrors
    /// upstream `make_cache(self) -> List[KVCache]`.
    pub fn make_cache(&self) -> Vec<KVCache> {
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    /// End-to-end forward.
    ///
    /// Shapes (matching upstream `__call__`):
    ///
    /// - `inputs`        — `[B, L]`, dtype int32 (token ids in the
    ///   proposal block, with `mask_token_id` at positions `1..L`).
    /// - `target_hidden` — `[B, T, num_target_layers * hidden_size]`,
    ///   the multi-layer concatenation produced by the Qwen 3.5 target
    ///   hooks (`return_hidden=true`, `capture_layer_ids=[1,8,15,22,29]`).
    /// - `cache` — per-layer drafter K/V caches; length must equal
    ///   `config.num_hidden_layers`.
    ///
    /// Returns logits of shape `[B, L, vocab_size]`.
    pub fn forward(
        &self,
        inputs: &MlxArray,
        target_hidden: &MlxArray,
        cache: &mut [KVCache],
    ) -> UniquePtr<MlxArray> {
        debug_assert_eq!(
            cache.len(),
            self.layers.len(),
            "DFlashDraftModel::forward: cache length {} does not match num_hidden_layers {}",
            cache.len(),
            self.layers.len()
        );

        // h = embed_tokens(inputs)  →  [B, L, hidden_size]
        let mut h = self.embed_tokens.forward(inputs);

        // h_ctx = hidden_norm(fc(target_hidden))  →  [B, T, hidden_size]
        let fc_out = self.fc.forward(target_hidden);
        let h_ctx = self.hidden_norm.forward(&fc_out);

        // Five decoder layers, threaded through their own cache slot.
        for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
            h = layer.forward(&h, &h_ctx, c);
        }

        // Final norm + LM head.
        let h = self.norm.forward(&h);
        match &self.lm_head {
            LmHead::Own(linear) => linear.forward(&h),
            LmHead::Tied => self.embed_tokens.as_linear(&h),
        }
    }

    /// Run a single masked-forward draft block and argmax-sample one
    /// token per proposal position.
    ///
    /// Builds `block = [last_bonus, mask_id, mask_id, ..., mask_id]` of
    /// shape `[1, block_size]`, runs the masked forward, slices
    /// `logits[:, 1 - block_size:]` (the last `block_size - 1` rows
    /// corresponding to the masked positions), and argmax-samples each
    /// row.
    ///
    /// Returns `Vec<i32>` of length `block_size - 1`.
    ///
    /// This is the B = 1 variant; the trait surface uses `B = 1` for the
    /// classic single-stream draft loop. A batched variant lives behind a
    /// future-facing helper if/when batched DFlash drafting lands.
    pub fn draft_block(
        &self,
        last_bonus: i32,
        target_hidden: &MlxArray,
        cache: &mut [KVCache],
        block_size: usize,
    ) -> Vec<i32> {
        assert!(
            block_size >= 2,
            "DFlash draft_block requires block_size >= 2 (got {block_size})",
        );

        let mask_id = self.config.mask_token_id;
        let mut block: Vec<i32> = Vec::with_capacity(block_size);
        block.push(last_bonus);
        for _ in 1..block_size {
            block.push(mask_id);
        }
        let inputs = ffi::from_slice_i32(&block, &[1, block_size as i32]);

        let logits = self.forward(&inputs, target_hidden, cache);

        // Slice [B=1, L=block_size, vocab] → [1, block_size-1, vocab]
        // along axis 1 (rows `[1..block_size]`).
        let shape = ffi::array_shape(&logits);
        debug_assert_eq!(shape.len(), 3, "logits must be [B, L, V], got {shape:?}");
        let vocab = shape[2];
        let starts = [0_i32, 1_i32, 0_i32];
        let stops = [1_i32, block_size as i32, vocab];
        let slice = ffi::slice(&logits, &starts, &stops);

        // Argmax over the vocab axis (axis=2 → keepdims=false → [1, L-1]).
        let argmax = ffi::argmax(&slice, 2, false);
        ffi::eval(&argmax);

        // Materialize the [1, L-1] argmax into Rust ints. Each slot is a
        // [1, 1] sub-slice → i32 scalar via ffi::item_i32.
        let n = block_size - 1;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let element = ffi::slice(&argmax, &[0_i32, i as i32], &[1_i32, (i + 1) as i32]);
            ffi::eval(&element);
            out.push(ffi::item_i32(&element));
        }
        out
    }

    /// Strip the `model.` prefix from any weight key carrying it, matching
    /// upstream `sanitize` exactly:
    ///
    /// ```python
    /// def sanitize(self, weights: dict) -> dict:
    ///     out = {}
    ///     for k, v in weights.items():
    ///         if k.startswith("model."):
    ///             k = k[len("model."):]
    ///         out[k] = v
    ///     return out
    /// ```
    ///
    /// This is a borrow-friendly variant: it mutates the existing weight
    /// map in place rather than allocating a new HashMap, because the
    /// values are `UniquePtr<MlxArray>` and cloning them defeats the
    /// MLX lazy-array sharing.
    pub fn sanitize(weights: &mut WeightMap) {
        const PREFIX: &str = "model.";

        let renames: Vec<(String, String)> = weights
            .keys()
            .filter(|k| k.starts_with(PREFIX))
            .map(|k| (k.clone(), k[PREFIX.len()..].to_string()))
            .collect();

        for (old, new) in renames {
            if let Some(v) = weights.remove(&old) {
                weights.insert(new, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;
    use crate::ffi;

    /// Pure compile-time check: enum variants used in `forward`.
    /// If a future patch adds a third variant without updating the match
    /// in `forward`, the corresponding match arm will fail to compile.
    #[test]
    fn lm_head_match_arms_are_exhaustive_compile_time() {
        fn _exhaustive(h: &LmHead) -> &'static str {
            match h {
                LmHead::Own(_) => "own",
                LmHead::Tied => "tied",
            }
        }
    }

    /// Sanitize strips the `model.` prefix from keys without losing data.
    #[test]
    fn sanitize_strips_model_prefix() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        // Use a stand-in MLX array (an empty f32 array). We don't care
        // about contents; only the rename semantics matter.
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            ffi::zeros(&[2, 2], dtype::FLOAT32),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            ffi::zeros(&[2, 2], dtype::FLOAT32),
        );
        weights.insert(
            "lm_head.weight".to_string(),
            ffi::zeros(&[2, 2], dtype::FLOAT32),
        );

        DFlashDraftModel::sanitize(&mut weights);

        assert!(weights.contains_key("embed_tokens.weight"));
        assert!(weights.contains_key("layers.0.self_attn.q_proj.weight"));
        assert!(weights.contains_key("lm_head.weight"));
        // The prefixed keys must be gone.
        assert!(!weights.contains_key("model.embed_tokens.weight"));
        assert!(!weights.contains_key("model.layers.0.self_attn.q_proj.weight"));
    }

    /// Sanitize is idempotent: a second call leaves the map unchanged.
    #[test]
    fn sanitize_is_idempotent() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            ffi::zeros(&[2, 2], dtype::FLOAT32),
        );
        weights.insert("norm.weight".to_string(), ffi::zeros(&[2], dtype::FLOAT32));

        DFlashDraftModel::sanitize(&mut weights);
        let after_first: std::collections::HashSet<String> = weights.keys().cloned().collect();

        DFlashDraftModel::sanitize(&mut weights);
        let after_second: std::collections::HashSet<String> = weights.keys().cloned().collect();

        assert_eq!(
            after_first, after_second,
            "sanitize must be idempotent: second pass changed keys"
        );
    }

    /// Sanitize preserves non-prefixed keys untouched.
    #[test]
    fn sanitize_preserves_non_prefixed_keys() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        weights.insert("fc.weight".to_string(), ffi::zeros(&[2, 2], dtype::FLOAT32));
        weights.insert(
            "hidden_norm.weight".to_string(),
            ffi::zeros(&[2], dtype::FLOAT32),
        );

        let initial_len = weights.len();
        DFlashDraftModel::sanitize(&mut weights);
        assert_eq!(weights.len(), initial_len, "no keys should be lost");
        assert!(weights.contains_key("fc.weight"));
        assert!(weights.contains_key("hidden_norm.weight"));
    }

    /// Sanitize handles the `model.lm_head.weight` rename — important
    /// because the LM head may live at either `lm_head.weight` (un-tied)
    /// or `model.lm_head.weight` (some checkpoint conventions).
    #[test]
    fn sanitize_renames_model_lm_head_weight() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        weights.insert(
            "model.lm_head.weight".to_string(),
            ffi::zeros(&[2, 2], dtype::FLOAT32),
        );

        DFlashDraftModel::sanitize(&mut weights);

        assert!(
            weights.contains_key("lm_head.weight"),
            "lm_head.weight must be reachable after sanitize"
        );
        assert!(
            !weights.contains_key("model.lm_head.weight"),
            "the prefixed key must be gone"
        );
    }
}
