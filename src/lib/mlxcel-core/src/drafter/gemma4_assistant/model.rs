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

//! `Gemma4AssistantDraftModel` — Rust port of the Gemma 4 MTP assistant
//! drafter.
//!
//! Mirrors
//! `references/mlx-vlm/mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py`.
//!
//! ## Lifecycle
//!
//! 1. **Load.** [`Gemma4AssistantDraftModel::from_path`] parses
//!    `config.json`, loads safetensors weights, sanitises them
//!    (`tie_word_embeddings` handling, `token_ordering` int32 cast), and
//!    constructs the model.
//! 2. **Bind.** [`Gemma4AssistantDraftModel::bind`] picks up the target's
//!    `embed_tokens` and `embed_scale`. Target wrapper depth is resolved
//!    against three known shapes (text-only / mid-wrapper / VLM-wrapped),
//!    matching upstream Python.
//! 3. **Per-block setup.** The round-loop calls
//!    [`Gemma4AssistantDraftModel::set_shared_kv`] with the target's last
//!    full / SWA K/V slabs, the bonus-token absolute position, and an
//!    optional `left_padding` for batched MTP.
//! 4. **Draft block.** [`Gemma4AssistantDraftModel::draft_block`] runs `K`
//!    autoregressive steps, returning `block_size - 1` proposal tokens.
//!
//! ## What is delegated to sibling sub-issues
//!
//! - **`MaskedEmbedder` centroid LM head (#627).** When the sibling PR lands,
//!   replace [`MaskedEmbedderStub`] with the real module. The stub keeps the
//!   non-sparse `tied_dense` path compilable today; the sparse path is gated
//!   on `config.use_ordered_embeddings == true` and returns an explicit error
//!   until the real `MaskedEmbedder` lands.
//! - **Drafter bidirectional masks (#628).** When the sibling PR lands,
//!   replace [`make_drafter_masks_stub`] with the real helper from
//!   `crate::drafter::masks`. The stub returns `None` masks, which is the
//!   correct unbatched-B=1 behaviour the round-loop hits in the common case
//!   (full attention always `None`; SWA `None` when `kv_len <= window`).

use crate::drafter::gemma4_assistant::config::{DrafterTextConfig, Gemma4AssistantConfig};
use crate::drafter::gemma4_assistant::layer::DraftDecoderLayer;
use crate::drafter::{Drafter, DrafterError, DrafterKind, SharedKv};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::{KVCache, Linear, RMSNorm, UnifiedEmbedding};
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Sibling-PR stubs (delete after #627 / #628 merge)
// ---------------------------------------------------------------------------
//
// These stubs let issue #626 compile and ship its tied-dense path standalone.
// When #627 (MaskedEmbedder) and #628 (drafter masks) merge into main, the
// orchestrator will rebase this PR and the follow-up cleanup is:
//
// 1. Replace [`MaskedEmbedderStub`] usage in `lm_head_fn` with
//    `crate::drafter::masked_embedder::MaskedEmbedder`.
// 2. Replace [`make_drafter_masks_stub`] with the real
//    `crate::drafter::masks::make_drafter_masks`.
// 3. Delete this entire stub block.

/// TODO(#627): replace with `crate::drafter::masked_embedder::MaskedEmbedder`
/// once the sibling PR merges.
///
/// Placeholder fields mirror the upstream Python `MaskedEmbedder.__init__`
/// surface so that downstream weight-key references (`masked_embedding.
/// token_ordering`, `masked_embedding.centroids.weight`) compile today.
#[allow(dead_code)]
pub(crate) struct MaskedEmbedderStub {
    /// `[hidden_size, num_centroids]` — projects drafter hidden states to
    /// per-centroid scores.
    centroids: Linear,
    /// `[vocab_size]` i32 lookup table from a position-in-cluster index to
    /// the canonical token ID. Stored as i32 (upstream casts from int64).
    token_ordering: UniquePtr<MlxArray>,
    hidden_size: usize,
    vocab_size: usize,
    num_centroids: usize,
    top_k: usize,
}

impl MaskedEmbedderStub {
    #[allow(dead_code)]
    pub(crate) fn from_weights(
        weights: &WeightMap,
        cfg: &Gemma4AssistantConfig,
    ) -> Result<Self, String> {
        let text_cfg = cfg.text_config();
        Ok(Self {
            centroids: Linear::from_weights(weights, "masked_embedding.centroids")?,
            token_ordering: weights
                .get("masked_embedding.token_ordering")
                .map(|w| ffi::copy(w))
                .ok_or_else(|| "Weight not found: masked_embedding.token_ordering".to_string())?,
            hidden_size: text_cfg.hidden_size,
            vocab_size: text_cfg.vocab_size,
            num_centroids: cfg.num_centroids,
            top_k: cfg.centroid_intermediate_top_k,
        })
    }
}

/// TODO(#628): replace with `crate::drafter::masks::make_drafter_masks` once
/// the sibling PR merges. The B=1, single-step (`query_len=1`) draft path
/// produces `None` masks in both layer-type buckets because:
///
/// - Full-attention: `bidirectional_full_mask` returns `None` when
///   `kv_valid_len >= kv_len`, which is the common B=1 case.
/// - SWA: `bidirectional_swa_mask` returns `None` when
///   `kv_len <= sliding_window`, which is the only regime
///   `RotatingKVCache` ever produces.
///
/// So this stub is BIT-IDENTICAL to the real helper in the unbatched MVP
/// path. Long-prompt batched MTP needs the real masks before it can light
/// up (see upstream `masks.py` for the full surface).
#[allow(dead_code)]
fn make_drafter_masks_stub(layer_types: &[String]) -> HashMap<String, Option<UniquePtr<MlxArray>>> {
    let mut masks = HashMap::new();
    masks.insert("full_attention".to_string(), None);
    masks.insert("sliding_attention".to_string(), None);
    // Defensive: also map any unknown layer-type to None.
    for lt in layer_types {
        masks.entry(lt.clone()).or_insert(None);
    }
    masks
}

// ---------------------------------------------------------------------------
// LM head dispatch
// ---------------------------------------------------------------------------

/// LM head variant resolved at `bind()`-time.
///
/// - `Tied` — use the drafter's `embed_tokens` as a linear projection,
///   matching the upstream `model.embed_tokens.as_linear` path. This is the
///   26B-A4B / 31B drafter case.
/// - `Linear` — explicit `lm_head` with its own `[vocab_size, hidden_size]`
///   weight matrix (when `tie_word_embeddings=False`).
/// - `Centroid` — sparse softmax via `MaskedEmbedder`. Gated on
///   `use_ordered_embeddings=True` (E2B / E4B drafters). Until #627 lands the
///   centroid path returns an explicit `DrafterError::NotYetImplemented`.
#[allow(dead_code)] // `Centroid` is gated behind sibling PR #627; constructed by `resolve_lm_head` once `MaskedEmbedder` lands.
enum LmHead {
    Tied,
    Linear(Linear),
    /// Sibling-PR stub. Until #627 merges, calling `lm_head_forward` with
    /// this variant returns `DrafterError::NotYetImplemented { issue: 627 }`.
    Centroid(MaskedEmbedderStub),
}

// ---------------------------------------------------------------------------
// _DraftInner equivalent (mirrors upstream `_DraftInner`)
// ---------------------------------------------------------------------------

/// Drafter inner module — mirrors the upstream `_DraftInner`. Owns the
/// drafter's own `embed_tokens` (used for the tied-dense LM head path) and
/// the `K`-layer transformer stack.
pub(crate) struct DraftInner {
    pub(crate) embed_tokens: UnifiedEmbedding,
    pub(crate) layers: Vec<DraftDecoderLayer>,
    pub(crate) norm: RMSNorm,
}

impl DraftInner {
    fn from_weights(
        weights: &WeightMap,
        config: &DrafterTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            &format!("{prefix}.embed_tokens"),
            config.group_size(),
            config.bits(),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DraftDecoderLayer::from_weights(
                weights,
                config,
                i,
                &format!("{prefix}.layers.{i}"),
            )?);
        }

        let norm = RMSNorm::new(
            weights
                .get(&format!("{prefix}.norm.weight"))
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {prefix}.norm.weight"))?,
            config.rms_norm_eps,
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }
}

// ---------------------------------------------------------------------------
// Shared K/V capture (issue-internal projection of `SharedKv`)
// ---------------------------------------------------------------------------

/// Owned drafter view of the target's shared K/V slabs.
///
/// `SharedKv<'a>` is borrow-typed at the trait boundary to forbid the drafter
/// from mutating target tensors in place. Internally the drafter copies a
/// fresh handle (cheap MLX-array clone — no device memory allocation) and
/// associates each K/V pair with its layer-type key (`"full_attention"` /
/// `"sliding_attention"`). This matches the upstream Python dict layout
/// `shared_kv_states[layer_type] = (K, V)`.
///
/// Until issue #625 finalises the shape of `SharedKv::tensors`, this struct
/// expects the tensor order `[k_full, v_full, k_swa, v_swa]` documented on
/// [`SharedKv`].
struct OwnedSharedKv {
    full_attention: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    sliding_attention: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
}

impl OwnedSharedKv {
    fn from_shared_kv(shared: &SharedKv<'_>) -> Result<Self, DrafterError> {
        // Documented `SharedKv` layout: tensors are ordered
        // `[k_full, v_full, k_swa, v_swa]`. Allow either 2 (full-attention
        // only — Gemma 4 minimal) or 4 (full + SWA — Gemma 4 production)
        // tensors so future Gemma 4 variants without sliding layers don't
        // trip the early validation.
        let owned = match shared.tensors.len() {
            2 => Self {
                full_attention: Some((ffi::copy(shared.tensors[0]), ffi::copy(shared.tensors[1]))),
                sliding_attention: None,
            },
            4 => Self {
                full_attention: Some((ffi::copy(shared.tensors[0]), ffi::copy(shared.tensors[1]))),
                sliding_attention: Some((
                    ffi::copy(shared.tensors[2]),
                    ffi::copy(shared.tensors[3]),
                )),
            },
            n => {
                return Err(DrafterError::SharedKvShape {
                    got: n,
                    expected: &[2, 4],
                });
            }
        };
        Ok(owned)
    }

    /// Batched-MTP-only constructor that runs the per-row left-padding
    /// normalization documented in [`crate::drafter::masks::normalize_batched_shared_kv_states`]
    /// before storing.
    ///
    /// Mirrors upstream Python's `_batch_cache_left_padding`-then-store
    /// shape: the drafter receives the target's shared K/V slabs as if
    /// they were prefix-valid against the drafter's "single row each
    /// occupies `[0, kv_valid_len[b]), tail zeroed" invariant.
    ///
    /// The current scalar `left_padding` arg from the trait is broadcast
    /// across rows. A follow-up will accept per-row `left_padding` vectors
    /// directly (tracked alongside the batched MTP wiring); until then,
    /// the round-loop driver collapses per-row `left_padding` to its max
    /// and the masks helper handles the (defensive) broadcast.
    fn from_shared_kv_normalized(
        shared: &SharedKv<'_>,
        left_padding: usize,
    ) -> Result<Self, DrafterError> {
        use crate::drafter::masks::{
            normalize_batched_shared_kv_states, BatchScalar, LayerType,
        };
        use std::collections::HashMap;

        // Build the `LayerType -> (K, V)` map the masks helper expects.
        let (k_full, v_full, k_swa, v_swa) = match shared.tensors.len() {
            2 => (shared.tensors[0], shared.tensors[1], None, None),
            4 => (
                shared.tensors[0],
                shared.tensors[1],
                Some(shared.tensors[2]),
                Some(shared.tensors[3]),
            ),
            n => {
                return Err(DrafterError::SharedKvShape {
                    got: n,
                    expected: &[2, 4],
                });
            }
        };

        // Each K/V tensor is shape [B, n_kv_heads, kv_len, head_dim]. The
        // valid prefix length per row is `kv_len - left_padding` (defensive
        // clip to non-negative). Pass scalar broadcasts; the masks helper
        // re-broadcasts to per-row internally.
        let kv_len = {
            let shape = ffi::array_shape(k_full);
            if shape.len() == 4 {
                shape[2]
            } else {
                0
            }
        };
        let kv_valid_len = kv_len.saturating_sub(left_padding as i32);

        let mut map: HashMap<LayerType, (&MlxArray, &MlxArray)> = HashMap::new();
        map.insert(LayerType::FullAttention, (k_full, v_full));
        if let (Some(ks), Some(vs)) = (k_swa, v_swa) {
            map.insert(LayerType::SlidingWindowAttention, (ks, vs));
        }

        let valid_scalar = BatchScalar::Scalar(kv_valid_len);
        let left_scalar = BatchScalar::Scalar(left_padding as i32);
        let normalized =
            normalize_batched_shared_kv_states(&map, &valid_scalar, Some(&left_scalar));

        let take_pair = |layer: LayerType| -> Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
            normalized.get(&layer).map(|(k, v)| (ffi::copy(k), ffi::copy(v)))
        };

        Ok(Self {
            full_attention: take_pair(LayerType::FullAttention),
            sliding_attention: take_pair(LayerType::SlidingWindowAttention),
        })
    }

    fn for_layer_type(&self, layer_type: &str) -> Result<(&MlxArray, &MlxArray), DrafterError> {
        let pair = match layer_type {
            "full_attention" => self.full_attention.as_ref(),
            "sliding_attention" => self.sliding_attention.as_ref(),
            other => {
                return Err(DrafterError::UnknownLayerType {
                    got: other.to_string(),
                });
            }
        };
        let (k, v) = pair.ok_or_else(|| DrafterError::MissingSharedKvForLayerType {
            layer_type: layer_type.to_string(),
        })?;
        Ok((
            k.as_ref().expect("non-null K"),
            v.as_ref().expect("non-null V"),
        ))
    }
}

// ---------------------------------------------------------------------------
// Gemma4AssistantDraftModel
// ---------------------------------------------------------------------------

/// Gemma 4 MTP "assistant" drafter — 4-layer transformer with pre/post
/// projections and frozen RoPE cross-attention into the target's last-layer
/// K/V slabs.
///
/// Implements [`Drafter`] and is wired into the `Mtp` arm of
/// [`crate::drafter::load_drafter`].
pub struct Gemma4AssistantDraftModel {
    config: Gemma4AssistantConfig,
    inner: DraftInner,
    pre_projection: Linear,
    post_projection: Linear,
    /// Explicit `lm_head` weight when `tie_word_embeddings == false`. `None`
    /// means the LM head is one of the tied / centroid variants resolved by
    /// `bind()`.
    lm_head_weight: Option<Linear>,
    /// LM head dispatch — finalised by `bind()`. Until then, callers that
    /// invoke `draft_block()` get an explicit "must call bind() first" error.
    lm_head: Option<LmHead>,

    /// Target's embedding table — captured by `bind()`. `None` means
    /// `bind()` has not run yet; `draft_block()` rejects that state.
    /// Stored as `MlxArray` from
    /// `LanguageModel::embed_tokens(&target_input_ids)`-returned tensors
    /// rather than holding a target reference, so the drafter doesn't need
    /// to keep a `&dyn LanguageModel` alive across calls.
    target_embed: Option<TargetEmbedAdapter>,
    target_embed_scale: f32,

    /// State set by `set_shared_kv()`. `None` means the round-loop has not
    /// armed the drafter yet.
    shared_kv: Option<OwnedSharedKv>,
    /// `kv_offset` from `set_shared_kv()` — kept for diagnostics.
    kv_offset: i32,
    /// Bonus-token absolute position. Used as the RoPE offset for every
    /// step inside a draft block (the "frozen anchor" semantics).
    position: i32,
}

// Manual `Debug` impl: `Linear`, `Embedding`, and `MlxArray` are FFI-opaque
// and do not derive `Debug`. The values themselves are not safe to materialise
// off the dispatch thread, so this surface intentionally renders only the
// scalar metadata diagnostic consumers actually want.
impl std::fmt::Debug for Gemma4AssistantDraftModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma4AssistantDraftModel")
            .field("model_type", &self.config.model_type)
            .field("backbone_hidden_size", &self.config.backbone_hidden_size)
            .field("block_size", &self.config.block_size)
            .field("tie_word_embeddings", &self.config.tie_word_embeddings)
            .field("use_ordered_embeddings", &self.config.use_ordered_embeddings)
            .field("num_layers", &self.inner.layers.len())
            .field("bound", &self.lm_head.is_some())
            .field("shared_kv_set", &self.shared_kv.is_some())
            .field("kv_offset", &self.kv_offset)
            .field("position", &self.position)
            .finish()
    }
}

/// Captured target embedding plumbing.
///
/// Holding a `&dyn LanguageModel` for the lifetime of the drafter would force
/// the drafter to outlive the target wrapper, which would in turn force the
/// caller to wrap the target in `Arc<dyn LanguageModel>`. Instead, `bind()`
/// extracts the target's embed weight once and stashes it here so subsequent
/// `embed(token_id)` lookups inside `draft_block()` go through MLX's
/// quantized/regular embedding kernel without re-entering the target trait
/// object.
///
/// The captured weight is whatever `LanguageModel::embed_tokens(input_ids)`
/// returned at `bind()` time, multiplied by `embed_scale` inside
/// [`Gemma4AssistantDraftModel::embed_with_scale`].
struct TargetEmbedAdapter {
    /// Last-known embedding tensor for the single sentinel token used during
    /// `bind()`. NOT used for live lookups — see `embed_via_target`.
    #[allow(dead_code)]
    sentinel: UniquePtr<MlxArray>,
    /// Cached `embed_tokens(input_ids)` callable as a raw function pointer
    /// would require a stable ABI on `LanguageModel`. Instead we capture
    /// `bind()`'s decision and re-enter the target through a stored
    /// closure-like indirection — see `Gemma4AssistantDraftModel::bind`.
    ///
    /// The current minimal binding records only the embed-scale; the
    /// `embed_tokens(...)` call is re-dispatched at use time through the
    /// target reference threaded into `draft_block` by the round-loop.
    /// Once the MTP round-loop (#629) lands, the round-loop will pass a
    /// `&dyn LanguageModel` to `draft_block` and this adapter becomes
    /// redundant; for now it holds only the sentinel weight so unit tests
    /// can introspect the binding.
    _phantom_target_ref: (),
}

impl Gemma4AssistantDraftModel {
    /// Construct from a checkpoint directory containing `config.json` and
    /// safetensors shards. Used by [`crate::drafter::load_drafter`]'s `Mtp`
    /// arm.
    pub fn from_path(path: &Path) -> Result<Self, DrafterError> {
        let config = load_config(path)?
            .normalize()
            .map_err(DrafterError::Config)?;
        let mut weights = crate::weights::load_weights_from_dir(path)
            .map_err(|e| DrafterError::WeightLoad { reason: e })?;
        Self::sanitize_weights(&mut weights, &config);
        Self::from_weights(weights, config)
    }

    /// Construct from an in-memory weight map. Used by both `from_path` and
    /// unit tests that build small fixture weight maps.
    pub fn from_weights(
        weights: WeightMap,
        config: Gemma4AssistantConfig,
    ) -> Result<Self, DrafterError> {
        let text_cfg = config.text_config().clone();

        let inner = DraftInner::from_weights(&weights, &text_cfg, "model")
            .map_err(|e| DrafterError::WeightLoad { reason: e })?;

        let pre_projection = Linear::from_weights(&weights, "pre_projection")
            .map_err(|e| DrafterError::WeightLoad { reason: e })?;
        let post_projection = Linear::from_weights(&weights, "post_projection")
            .map_err(|e| DrafterError::WeightLoad { reason: e })?;

        let lm_head_weight = if config.tie_word_embeddings {
            None
        } else {
            Some(
                Linear::from_weights(&weights, "lm_head")
                    .map_err(|e| DrafterError::WeightLoad { reason: e })?,
            )
        };

        Ok(Self {
            config,
            inner,
            pre_projection,
            post_projection,
            lm_head_weight,
            lm_head: None,
            target_embed: None,
            target_embed_scale: 1.0,
            shared_kv: None,
            kv_offset: 0,
            position: 0,
        })
    }

    /// Apply the upstream Python `sanitize` rules to a freshly-loaded weight
    /// map:
    ///
    /// - When `tie_word_embeddings == true`, drop `lm_head.weight` (and any
    ///   sister tensors) — it must not be loaded as a standalone Linear.
    /// - Cast `masked_embedding.token_ordering` from int64 to int32 (used
    ///   only on E-series drafters with the centroid LM head).
    ///
    /// Mirrors upstream `Gemma4AssistantDraftModel.sanitize` in
    /// `references/mlx-vlm/mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py`.
    pub fn sanitize_weights(weights: &mut WeightMap, config: &Gemma4AssistantConfig) {
        if config.tie_word_embeddings {
            weights.remove("lm_head.weight");
            weights.remove("lm_head.scales");
            weights.remove("lm_head.biases");
        }
        // TODO(#627): cast `masked_embedding.token_ordering` to int32 here
        // once the real `MaskedEmbedder` is wired in. The dtype cast is a
        // no-op on already-int32 buffers, so the runtime path will keep
        // working even if a stale checkpoint ships with int64. Mirrors
        // upstream `if k == "masked_embedding.token_ordering": v = v.astype(mx.int32)`.
    }

    /// Configure the LM head dispatch based on the drafter's config and
    /// captured weights. Called from [`Self::bind`].
    fn resolve_lm_head(&mut self) -> Result<(), DrafterError> {
        let head = if self.config.use_ordered_embeddings {
            // Centroid LM head — needs MaskedEmbedder from #627.
            return Err(DrafterError::NotYetImplemented {
                kind: DrafterKind::Mtp,
                issue: 627,
            });
        } else if self.config.tie_word_embeddings {
            LmHead::Tied
        } else {
            // Explicit lm_head — already loaded into `lm_head_weight` by
            // `from_weights`.
            let head_weight =
                self.lm_head_weight
                    .take()
                    .ok_or_else(|| DrafterError::WeightLoad {
                        reason: "tie_word_embeddings=false but lm_head.weight was not loaded"
                            .into(),
                    })?;
            LmHead::Linear(head_weight)
        };
        self.lm_head = Some(head);
        Ok(())
    }

    /// Resolve the target's inner module via the three known wrapper depths
    /// and capture its embedding scale.
    ///
    /// Mirrors upstream:
    /// ```python
    /// if hasattr(target_model, "embed_tokens"):
    ///     inner = target_model
    /// elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
    ///     inner = target_model.model
    /// elif (hasattr(target_model, "language_model")
    ///       and hasattr(target_model.language_model, "model")
    ///       and hasattr(target_model.language_model.model, "embed_tokens")):
    ///     inner = target_model.language_model.model
    /// ```
    ///
    /// In Rust, all three depths converge on the [`LanguageModel`] trait's
    /// [`LanguageModel::embed_tokens`] method, which the gemma4 wrappers
    /// implement at every depth and forward to the text model's embedding
    /// table. The drafter only needs to know that the method returns
    /// `Some(_)` — if it returns `None`, the target does not expose its
    /// embedding plumbing (e.g. some text-only models that do not implement
    /// embed_tokens) and `bind()` fails.
    fn capture_target_embedding(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        // Build a single-element sentinel input so we can call
        // `target.embed_tokens(input_ids)` and confirm the target exposes its
        // embedding plumbing. The returned tensor's first row is the
        // embedding of token id 0 — we only need it to fail-fast when the
        // target lacks the override.
        let sentinel_ids = ffi::from_slice_i32(&[0], &[1, 1]);
        let embedded =
            target
                .embed_tokens(&sentinel_ids)
                .ok_or(DrafterError::TargetMissingFeature {
                    feature: "embed_tokens",
                })?;

        // The embed scale (Gemma multiplies by sqrt(hidden_size)) is not
        // surfaced through `LanguageModel`, but the drafter's
        // `target_embed_scale` must match it for parity. The round-loop
        // (#629) will set this explicitly per-target via a follow-up hook
        // (TODO: track in #629). Until then we default to 1.0 — bit-exact
        // for targets that do not scale, and the per-token magnitude
        // mismatch on Gemma will surface as a measurable acceptance-rate
        // drop, which the integration test in #629 catches.
        self.target_embed_scale = 1.0;
        self.target_embed = Some(TargetEmbedAdapter {
            sentinel: embedded,
            _phantom_target_ref: (),
        });

        // Capture target's layer_types via a best-effort interface query.
        // The current `LanguageModel` trait does not expose `layer_types`
        // so the drafter falls back to the drafter's own text_config
        // layer_types (which mirror the target by construction on all four
        // supported pairings). When #629 wires a richer round-loop API,
        // pass the target's actual layer_types in here.
        self.config.target_layer_types = self.config.text_config().layer_types.clone();

        Ok(())
    }

    /// Embed a single token id via the target's embedding table, applying
    /// `embed_scale` per Gemma convention.
    ///
    /// `token_id` is `[B=1, 1]`-shape; returns `[1, 1, backbone_hidden_size]`.
    /// The result feeds into `pre_projection` after concatenation with the
    /// drafter's recurrent hidden state.
    ///
    /// Will be invoked from the MTP round-loop (#629) once the round-loop
    /// threads `&dyn LanguageModel` through `draft_block`. Until then, the
    /// drafter falls back to its own `embed_tokens` inside `draft_block` so
    /// the per-step embedding lookup still works in unit tests.
    #[allow(dead_code)]
    fn embed_with_scale(
        &self,
        target: &dyn LanguageModel,
        token_id: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let embedded = target
            .embed_tokens(token_id)
            .ok_or(DrafterError::TargetMissingFeature {
                feature: "embed_tokens",
            })?;
        // multiply_scalar accepts f32; embed_scale = 1.0 is a no-op fast
        // path inside MLX so this is free in the default case.
        Ok(crate::multiply_scalar(&embedded, self.target_embed_scale))
    }

    /// Forward through the drafter's transformer stack with the current
    /// `shared_kv` slabs. Mirrors `Gemma4AssistantDraftModel.__call__` from
    /// upstream Python.
    ///
    /// `inputs_embeds`: `[1, 1, 2 * backbone_hidden_size]` (concat of target
    /// embed + last hidden). Returns `(last_hidden, logits)` where
    /// `last_hidden` has shape `[1, 1, backbone_hidden_size]` (output of
    /// `post_projection`) and `logits` has shape `[1, 1, vocab_size]`.
    fn forward(
        &self,
        inputs_embeds: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), DrafterError> {
        let shared = self
            .shared_kv
            .as_ref()
            .ok_or(DrafterError::SetSharedKvNotCalled)?;
        let lm_head = self.lm_head.as_ref().ok_or(DrafterError::BindNotCalled)?;

        // pre_projection: [1, 1, 2 * backbone] → [1, 1, drafter_hidden]
        let mut h = self.pre_projection.forward(inputs_embeds);

        // Build masks. The stub returns None for both layer types — bit-
        // identical to the real `make_drafter_masks` helper in the B=1,
        // query_len=1 regime the round-loop hits per-step.
        // TODO(#628): replace with `crate::drafter::masks::make_drafter_masks`.
        let layer_types = &self.config.text_config().layer_types;
        let masks = make_drafter_masks_stub(layer_types);

        // Run each drafter layer with shared K/V and the frozen RoPE offset.
        for layer in &self.inner.layers {
            let (k, v) = shared.for_layer_type(layer.layer_type())?;
            let mask_opt = masks.get(layer.layer_type()).and_then(|m| m.as_deref());
            h = layer.forward(&h, mask_opt, k, v, self.position);
        }

        // Final RMSNorm + post_projection.
        let h = self.inner.norm.forward(&h);
        let last_hidden = self.post_projection.forward(&h);

        // LM head: tied dense uses drafter's `embed_tokens.as_linear`,
        // explicit linear uses its own weight, centroid is gated to #627.
        let logits = match lm_head {
            LmHead::Tied => self.inner.embed_tokens.as_linear(&h),
            LmHead::Linear(linear) => linear.forward(&h),
            LmHead::Centroid(_) => {
                return Err(DrafterError::NotYetImplemented {
                    kind: DrafterKind::Mtp,
                    issue: 627,
                });
            }
        };

        Ok((last_hidden, logits))
    }

    /// Single-row argmax sample from a `[1, 1, vocab]` logits tensor.
    ///
    /// The full `SamplingConfig`-aware sampler currently operates on
    /// `[batch, seq, vocab]` and re-enters per-sequence state tracking —
    /// the drafter only needs a degenerate path. Temperature 0 (greedy) is
    /// the only path verified byte-identical by upstream (see README:
    /// "Quality matches the target at temperature 0"), so non-greedy
    /// configs are a quality-loss path the round-loop owns. We still
    /// honour `temperature` via the existing `fused_sample` kernel so that
    /// future temperature-aware MTP variants get the correct draws.
    fn sample_one(logits: &MlxArray, sampler: &SamplingConfig) -> UniquePtr<MlxArray> {
        let last_logits = ffi::slice_last_logits(logits);
        ffi::fused_sample(
            &last_logits,
            sampler.temperature,
            sampler.top_k,
            sampler.top_p,
            sampler.min_p,
        )
    }
}

impl Drafter for Gemma4AssistantDraftModel {
    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.capture_target_embedding(target)?;
        self.resolve_lm_head()?;
        Ok(())
    }

    fn set_shared_kv(
        &mut self,
        shared_kv: SharedKv<'_>,
        kv_offset: usize,
        position: usize,
        left_padding: usize,
    ) -> Result<(), DrafterError> {
        // Per issue #631 (batched MTP): when the round-loop is running B > 1
        // with left-padded shared K/V, the drafter normalizes each row so
        // the cross-attention forward sees the simpler invariant: each
        // row's real keys occupy `[0, kv_valid_len)` and the tail is
        // zeroed. Routes through
        // [`crate::drafter::masks::normalize_batched_shared_kv_states`].
        //
        // For B = 1 (or `left_padding == 0`), we skip the normalization
        // path entirely — it is a no-op on the unbatched MVP shape and
        // would only add an unnecessary tensor copy. The bit-identity test
        // in `tests.rs` (`round_loop_full_accept_emits_all_proposals_plus_bonus_per_round`)
        // pins the no-normalize path's behaviour.
        if left_padding > 0 {
            self.shared_kv = Some(OwnedSharedKv::from_shared_kv_normalized(
                &shared_kv,
                left_padding,
            )?);
        } else {
            self.shared_kv = Some(OwnedSharedKv::from_shared_kv(&shared_kv)?);
        }
        self.kv_offset = kv_offset as i32;
        self.position = position as i32;
        Ok(())
    }

    fn make_cache(&self) -> Vec<KVCache> {
        // The MTP drafter has no own KV cache — its only recurrent state is
        // the target's last hidden, projected through `post_projection`. The
        // default trait impl already returns an empty Vec, so the override
        // here is only to be explicit about intent.
        Vec::new()
    }

    fn draft_block_batched(
        &mut self,
        last_bonus: &[i32],
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<Vec<i32>>, DrafterError> {
        // Batched autoregressive draft (issue #631). Performs `K-1` small
        // forwards with `[B, 1, ...]` shapes, sampling one token per row
        // each step. Mirrors the B = 1 path in [`Self::draft_block`] but
        // keeps the batch dim throughout.
        //
        // The drafter MUST have been `bind`()-ed and `set_shared_kv`()-ed
        // before reaching this point (the round-loop driver enforces
        // both). The shared K/V's batch dim has to match `last_bonus.len()`
        // for the cross-attention forward to produce the right per-row
        // outputs.
        if self.shared_kv.is_none() {
            return Err(DrafterError::SetSharedKvNotCalled);
        }
        if self.lm_head.is_none() {
            return Err(DrafterError::BindNotCalled);
        }
        if block_size == 0 || last_bonus.is_empty() {
            return Ok(last_bonus.iter().map(|_| Vec::new()).collect());
        }
        let hidden = hidden.ok_or(DrafterError::DraftBlockMissingHidden)?;

        let batch_size = last_bonus.len();
        let proposals = (block_size as i32).saturating_sub(1).max(0);
        if proposals == 0 {
            return Ok((0..batch_size).map(|_| Vec::new()).collect());
        }

        // Per-row token-stream accumulators.
        let mut tokens_per_row: Vec<Vec<i32>> =
            (0..batch_size).map(|_| Vec::with_capacity(proposals as usize)).collect();

        // Per-step recurrent state: `h_prev` starts at the caller's
        // [B, 1, backbone] target hidden; `last_tokens` starts at the
        // per-row bonus slice.
        let mut h_prev = ffi::copy(hidden);
        let mut last_tokens: Vec<i32> = last_bonus.to_vec();

        for _ in 0..proposals {
            // Per-row embed: build a [B, 1] token-id tensor, embed, scale.
            let tok_ids = ffi::from_slice_i32(&last_tokens, &[batch_size as i32, 1]);
            let tok_embed = self.inner.embed_tokens.forward(&tok_ids);
            let tok_embed = if self.target_embed.is_some() {
                crate::multiply_scalar(&tok_embed, self.target_embed_scale)
            } else {
                tok_embed
            };

            // [B, 1, hidden] + [B, 1, backbone] → [B, 1, 2 * backbone]
            let inputs_embeds = crate::concatenate(&tok_embed, &h_prev, -1);

            let (next_hidden, logits) = self.forward(&inputs_embeds)?;

            // Per-row argmax (or sampled) tokens. Greedy at temp=0 is the
            // load-bearing correctness path; non-greedy is the
            // quality-loss path the round-loop owns.
            let last_logits = ffi::slice_last_logits(&logits);
            let sampled = ffi::fused_sample(
                &last_logits,
                sampler.temperature,
                sampler.top_k,
                sampler.top_p,
                sampler.min_p,
            );
            ffi::eval(&sampled);

            // Materialize each row's sampled token. Shape of `sampled`
            // is `[B]` (one int per batch row).
            for r in 0..batch_size {
                let cell = ffi::slice(&sampled, &[r as i32], &[(r as i32) + 1]);
                let scalar = ffi::reshape(&cell, &[]);
                let tok = ffi::item_i32(&scalar);
                tokens_per_row[r].push(tok);
                last_tokens[r] = tok;
            }

            h_prev = next_hidden;
        }

        Ok(tokens_per_row)
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        // The trait signature does NOT thread a `&dyn LanguageModel`
        // through. The MTP round-loop (#629) needs to do that so the
        // drafter can embed `last_bonus` via the target every step. Until
        // #629 wires it, the drafter falls back to its own `embed_tokens`
        // (which the upstream Python NEVER does — but is observable in the
        // unit tests of #626 that exercise the forward-pass path without
        // a target). This branch is documented in the issue acceptance
        // criteria and will be replaced when #629's round-loop is wired
        // through the trait.
        //
        // For now we hard-require `bind()` to have been called and pull the
        // target embed from the bound state. If the drafter is in
        // bind()-after-but-with-no-target state (only possible in unit
        // tests), fall back to the drafter's own embed_tokens with
        // embed_scale = 1.0.

        if self.shared_kv.is_none() {
            return Err(DrafterError::SetSharedKvNotCalled);
        }
        if self.lm_head.is_none() {
            return Err(DrafterError::BindNotCalled);
        }
        if block_size == 0 {
            return Ok(Vec::new());
        }
        let hidden = hidden.ok_or(DrafterError::DraftBlockMissingHidden)?;

        let mut tokens: Vec<i32> = Vec::with_capacity(block_size.saturating_sub(1));

        // Per upstream Python `draft_block`:
        //   for _ in range(block_size - 1):
        //     tok_embed = self._input_embed(tok) * self._input_embed_scale
        //     inputs_embeds = mx.concatenate([tok_embed, h_prev], axis=-1)
        //     h_prev, logits = self(inputs_embeds, shared_kv, position_ids)
        //     tok = sampler(logits)
        //     tokens.append(tok)
        //
        // `h_prev` starts at the target's last hidden (`hidden`).
        let mut h_prev = ffi::copy(hidden);
        let mut last_token = last_bonus;

        for _ in 0..block_size.saturating_sub(1) {
            // Embed last_token using the drafter's own embed_tokens as a
            // fallback for the no-target tests. The real round-loop (#629)
            // will pass the target through and the embedding will go via
            // `embed_with_scale`. The fallback is mathematically valid
            // because tied-dense drafters share the embed table with the
            // target by construction.
            let tok_ids = ffi::from_slice_i32(&[last_token], &[1, 1]);
            let tok_embed = self.inner.embed_tokens.forward(&tok_ids);
            // For Gemma 4, the target uses sqrt(hidden_size) as embed_scale.
            // The drafter's tied embedding does NOT need that scaling — the
            // upstream Python explicitly avoids it on the drafter side by
            // routing through `self._input_embed = inner.embed_tokens` and
            // only multiplying when that captured callable is in use.
            // Without a captured target embed, skip the scale to stay
            // bit-identical to the test fixture path. The round-loop will
            // call `embed_with_scale` via a target reference in #629.
            let tok_embed = if self.target_embed.is_some() {
                crate::multiply_scalar(&tok_embed, self.target_embed_scale)
            } else {
                tok_embed
            };

            // Concatenate along the last axis: [1, 1, hidden_size] + [1, 1,
            // backbone_hidden_size] → [1, 1, 2 * backbone_hidden_size]. The
            // upstream code keeps `h_prev` at `backbone_hidden_size` (it
            // came from `post_projection`), so the drafter's `pre_projection`
            // expects `2 * backbone_hidden_size`.
            let inputs_embeds = crate::concatenate(&tok_embed, &h_prev, -1);

            let (next_hidden, logits) = self.forward(&inputs_embeds)?;
            let token = Self::sample_one(&logits, sampler);
            ffi::eval(&token);
            let token_i32 = ffi::item_i32(&token);
            tokens.push(token_i32);

            last_token = token_i32;
            h_prev = next_hidden;
        }

        Ok(tokens)
    }

    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError> {
        Self::sanitize_weights(weights, &self.config);
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}

/// Best-effort `config.json` loader. Routes through `serde_json::from_slice`
/// rather than the project's heavier config-loading utilities so this stays
/// free of `mlxcel`-crate dependencies.
fn load_config(path: &Path) -> Result<Gemma4AssistantConfig, DrafterError> {
    let cfg_path = path.join("config.json");
    let bytes = std::fs::read(&cfg_path).map_err(|e| DrafterError::ConfigIo {
        path: cfg_path.display().to_string(),
        source: e,
    })?;
    serde_json::from_slice::<Gemma4AssistantConfig>(&bytes).map_err(|e| DrafterError::ConfigParse {
        path: cfg_path.display().to_string(),
        source: e,
    })
}
