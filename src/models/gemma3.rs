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

//! Gemma 3 model implementation using mlxcel-core
//!
//! Key features:
//! - sliding_window_pattern (int): layer i is global if (i+1) % pattern == 0
//! - Local: RotatingKVCache + sliding window mask + local RoPE base
//! - Global: KVCache + full mask + global RoPE base
//! - Q/K norm with (1+weight) GemmaRMSNorm
//! - 4 RMSNorm per layer
//! - GELU activation
//! - clip_residual_f16 for float16 safety
//! - Embedding scaling: h *= sqrt(hidden_size)

use mlxcel_core::layers::{
    GemmaRMSNorm, KVCache, RotatingKVCache, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::utils::{create_causal_mask, create_causal_mask_with_window};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    /// Global RoPE base frequency (used for non-sliding-window layers)
    #[serde(alias = "rope_global_base_freq")]
    pub rope_theta: f32,
    pub rope_local_base_freq: f32,
    pub query_pre_attn_scalar: f32,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,

    pub rope_scaling: Option<std::collections::HashMap<String, serde_json::Value>>,

    pub quantization: Option<Quantization>,
}

impl Default for ModelArgs {
    fn default() -> Self {
        // Defaults match Python mlx-vlm TextConfig for Gemma3
        Self {
            model_type: "gemma3_text".to_string(),
            hidden_size: 2048,
            num_hidden_layers: 26,
            intermediate_size: 16384,
            num_attention_heads: 8,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            vocab_size: 262208,
            num_key_value_heads: 4,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            query_pre_attn_scalar: 256.0,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            max_position_embeddings: 4096,
            rope_scaling: None,
            quantization: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

impl ModelArgs {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }
}

// Attention.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub is_sliding: bool,
    pub rope_base: f32,
    pub q_norm: GemmaRMSNorm,
    pub k_norm: GemmaRMSNorm,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape and transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Apply Q/K normalization AFTER transpose (matches Python mlx-lm)
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        let offset = cache.offset();

        // Apply RoPE
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Use fused scaled dot-product attention (handles GQA internally)
        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let attn_out = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(
                &q, &cache_k, &cache_v, self.scale, mask_ptr,
            )
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let head_dim = args.head_dim as i32;
        let scale = 1.0 / args.query_pre_attn_scalar.sqrt();

        // Determine if this is a sliding window layer
        let is_sliding = !(layer_idx + 1).is_multiple_of(args.sliding_window_pattern);

        // Choose RoPE base based on layer type
        let rope_base = if is_sliding {
            args.rope_local_base_freq
        } else {
            args.rope_theta
        };

        // Load Q/K normalization
        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let q_norm = GemmaRMSNorm::new(q_norm_weight, args.rms_norm_eps);
        let k_norm = GemmaRMSNorm::new(k_norm_weight, args.rms_norm_eps);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale,
            is_sliding,
            rope_base,
            q_norm,
            k_norm,
        })
    }
}

// MLP (GELU activation).
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // GeGLU: gelu(gate_proj(x)) * up_proj(x), then down_proj
        // Use compiled GELU MLP when possible for kernel fusion
        if let (Some(gate_qw), Some(up_qw), Some(down_qw)) = (
            self.gate_proj.quantized_weight(),
            self.up_proj.quantized_weight(),
            self.down_proj.quantized_weight(),
        ) {
            return unsafe {
                mlxcel_core::compiled_gelu_mlp_forward(
                    x,
                    &gate_qw.weight,
                    &gate_qw.scales,
                    gate_qw.biases_ptr(),
                    &up_qw.weight,
                    &up_qw.scales,
                    up_qw.biases_ptr(),
                    &down_qw.weight,
                    &down_qw.scales,
                    down_qw.biases_ptr(),
                    gate_qw.group_size,
                    gate_qw.bits,
                    &gate_qw.mode,
                )
            };
        }

        // Fallback: separate operations with compiled activation
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_geglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let gate_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate_proj", prefix),
            group_size,
            bits,
        )?;
        let up_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

// Transformer Block (4 RMSNorm per layer).
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: GemmaRMSNorm,
    pub post_attention_layernorm: GemmaRMSNorm,
    pub pre_feedforward_layernorm: GemmaRMSNorm,
    pub post_feedforward_layernorm: GemmaRMSNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let post_attn_normed = self.post_attention_layernorm.forward(&attn_out);
        let h = mlxcel_core::compiled_clip_residual(x, &post_attn_normed);

        // Pre-norm FFN
        let normed = self.pre_feedforward_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        let post_ff_normed = self.post_feedforward_layernorm.forward(&ff_out);
        mlxcel_core::compiled_clip_residual(&h, &post_ff_normed)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn =
            Attention::from_weights(weights, args, &format!("{}.self_attn", prefix), layer_idx)?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let pre_ff_norm_weight = get_weight_copy(
            weights,
            &format!("{}.pre_feedforward_layernorm.weight", prefix),
        )?;
        let post_ff_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_feedforward_layernorm.weight", prefix),
        )?;

        let input_layernorm = GemmaRMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = GemmaRMSNorm::new(post_attn_norm_weight, args.rms_norm_eps);
        let pre_feedforward_layernorm = GemmaRMSNorm::new(pre_ff_norm_weight, args.rms_norm_eps);
        let post_feedforward_layernorm = GemmaRMSNorm::new(post_ff_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }
}

// Cache Interface.
pub trait CacheInterface {
    fn offset(&self) -> i32;
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>);
}

impl CacheInterface for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(k, v)
    }
}

impl CacheInterface for RotatingKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(k, v)
    }
}

pub enum Cache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl Cache {
    fn as_interface(&mut self) -> &mut dyn CacheInterface {
        match self {
            Cache::Standard(c) => c,
            Cache::Rotating(c) => c,
        }
    }
}

// Gemma3 Model.
pub struct Gemma3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: GemmaRMSNorm,
    pub lm_head: UnifiedLinear,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub hidden_size: usize,
}

impl Gemma3Model {
    /// Get token embeddings scaled by sqrt(hidden_size)
    /// Used by: VisionModule for merging vision and text embeddings
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        let scale = (self.hidden_size as f32).sqrt();
        mlxcel_core::multiply_scalar(&h, scale)
    }

    /// Forward pass with pre-computed embeddings (for VLM)
    /// If input_embeddings is Some, skip embed_tokens and use provided embeddings
    pub fn forward_with_caches_and_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        external_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape[1];

        // Use pre-computed embeddings if provided, otherwise embed tokens
        let mut h = if let Some(embeddings) = input_embeddings {
            mlxcel_core::copy(embeddings)
        } else {
            self.get_embed_tokens(input_ids)
        };

        // If external 4D mask is provided (from VLM), use it directly
        if let Some(ext_mask) = external_mask {
            // VLM provides a 4D attention mask — apply it to all layers
            for (i, layer) in self.layers.iter().enumerate() {
                h = layer.forward(&h, caches[i].as_interface(), Some(ext_mask));
            }
        } else if seq_len == 1 {
            // Decode path (seq_len=1): no mask needed — matches Python mlx-lm
            // which returns None from create_attention_mask when N=1.
            // The fused SDPA handles single-token attention without explicit masks.
            for (i, layer) in self.layers.iter().enumerate() {
                h = layer.forward(&h, caches[i].as_interface(), None);
            }
        } else {
            // Prefill path (seq_len > 1): create causal masks
            let global_idx = self.sliding_window_pattern - 1;
            let global_offset = caches[global_idx].as_interface().offset();
            let global_mask = Some(create_causal_mask(seq_len, global_offset));

            let sliding_mask = if self.sliding_window_pattern > 1 {
                let sliding_offset = caches[0].as_interface().offset();
                // Clamp offset so mask shape matches RotatingKVCache output.
                // The cache returns at most max_size tokens, so the mask's
                // total_len (= seq_len + offset) must not exceed max_size.
                let max_cache = self.sliding_window as i32;
                let effective_offset = sliding_offset.min((max_cache - seq_len).max(0));
                Some(create_causal_mask_with_window(
                    seq_len,
                    effective_offset,
                    Some(max_cache),
                ))
            } else {
                None
            };

            for (i, layer) in self.layers.iter().enumerate() {
                let is_global =
                    (i % self.sliding_window_pattern) == (self.sliding_window_pattern - 1);
                let mask = if is_global {
                    global_mask.as_ref().map(|m| m.as_ref().unwrap())
                } else {
                    sliding_mask.as_ref().map(|m| m.as_ref().unwrap())
                };
                h = layer.forward(&h, caches[i].as_interface(), mask);
            }
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        self.lm_head.forward(&h)
    }

    /// Forward pass through the entire model
    pub fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> UniquePtr<MlxArray> {
        self.forward_with_caches_and_embeddings(input_ids, None, caches, None)
    }

    /// Create KV caches for all layers
    pub fn make_caches(&self) -> Vec<Cache> {
        (0..self.layers.len())
            .map(|i| {
                let is_global =
                    (i % self.sliding_window_pattern) == (self.sliding_window_pattern - 1);
                if is_global {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.sliding_window as i32))
                }
            })
            .collect()
    }

    /// Load model from directory
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights (with tied-embedding sanitization)
        let weights = crate::models::load_and_sanitize_weights(model_dir)?;

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load quantized embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = GemmaRMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: args.sliding_window,
            sliding_window_pattern: args.sliding_window_pattern,
            hidden_size: args.hidden_size,
        })
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
use std::cell::RefCell;

/// Wrapper for Gemma3Model that implements LanguageModel trait
/// Uses internal cache management for sliding window attention
pub struct Gemma3Wrapper {
    model: Gemma3Model,
    caches: RefCell<Vec<Cache>>,
}

impl Gemma3Wrapper {
    pub fn new(model: Gemma3Model) -> Self {
        let caches = model.make_caches();
        Self {
            model,
            caches: RefCell::new(caches),
        }
    }

    pub fn reset_caches(&self) {
        let caches = self.model.make_caches();
        *self.caches.borrow_mut() = caches;
    }
}

impl mlxcel_core::generate::LanguageModel for Gemma3Wrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut caches = self.caches.borrow_mut();
        self.model.forward_with_caches(input_ids, &mut caches)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [mlxcel_core::layers::KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut caches = self.caches.borrow_mut();
        self.model.forward_with_caches_and_embeddings(
            input_ids,
            input_embeddings,
            &mut caches,
            mask,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.model.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
        // Reset internal caches
        self.reset_caches();
        // Return dummy caches (won't be used - internal caches are used instead)
        (0..self.model.layers.len())
            .map(|_| mlxcel_core::layers::KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    fn supports_batching(&self) -> bool {
        false // Gemma3 uses internal RefCell mixed caches (KV + Rotating), not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0, 1, 106] // Gemma3: <pad> (0), <eos> (1), <end_of_turn> (106)
    }
}
