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

//! Gemma 2 model implementation using mlxcel-core
//!
//! Key features:
//! - Softcapping attention: scores = tanh(scores / attn_softcap) * attn_softcap
//! - Final logit softcapping: logits = tanh(logits / final_softcap) * final_softcap
//! - 4 RMSNorm per layer (input, post_attention, pre_feedforward, post_feedforward)
//! - (1+weight) RMSNorm pattern
//! - GELU activation
//! - query_pre_attn_scalar for attention scale

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{GemmaRMSNorm, KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::softcap;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub head_dim: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<std::collections::HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub quantization: Option<Quantization>,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_tie_word_embeddings() -> bool {
    true
}

impl ModelArgs {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    pub fn query_pre_attn_scalar(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|m| m.get("query_pre_attn_scalar"))
            .and_then(|v| v.as_f64())
            .unwrap_or(self.head_dim as f64) as f32
    }

    pub fn attn_logit_softcapping(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|m| m.get("attn_logit_softcapping"))
            .and_then(|v| v.as_f64())
            .unwrap_or(50.0) as f32
    }

    pub fn final_logit_softcapping(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|m| m.get("final_logit_softcapping"))
            .and_then(|v| v.as_f64())
            .unwrap_or(30.0) as f32
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
    pub rope_dims: i32,
    pub rope_base: f32,
    pub softcapping: f32,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE
        let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Repeat KV heads for GQA (if num_kv_heads < num_heads)
        let n_rep = self.num_heads / self.num_kv_heads;
        let cache_k = mlxcel_core::utils::repeat_kv(&cache_k, n_rep);
        let cache_v = mlxcel_core::utils::repeat_kv(&cache_v, n_rep);

        // Compute attention scores
        let k_t = mlxcel_core::transpose_axes(&cache_k, &[0, 1, 3, 2]);
        let scores = mlxcel_core::matmul(&q, &k_t);

        // Scale scores
        let scores = mlxcel_core::multiply_scalar(&scores, self.scale);

        // Apply softcapping: tanh(scores / cap) * cap
        let scores = softcap(&scores, self.softcapping);

        // Apply mask if provided
        let scores = if let Some(m) = mask {
            mlxcel_core::add(&scores, m)
        } else {
            scores
        };

        // Softmax
        let probs = mlxcel_core::softmax(&scores, -1);

        // Multiply by values
        let attn_out = mlxcel_core::matmul(&probs, &cache_v);

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
        let q_pre_attn_scalar = args.query_pre_attn_scalar();
        let softcapping = args.attn_logit_softcapping();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / q_pre_attn_scalar.sqrt(),
            rope_dims: head_dim,
            rope_base: args.rope_theta,
            softcapping,
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
        // GELU(gate_proj(x)) * up_proj(x)
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // GELU activation
        let gate_gelu = mlxcel_core::gelu(&gate);

        // Element-wise product
        let activated = mlxcel_core::multiply(&gate_gelu, &up);

        // Down projection
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
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let post_attn_normed = self.post_attention_layernorm.forward(&attn_out);
        let h = mlxcel_core::add(x, &post_attn_normed);

        // Pre-norm FFN
        let normed = self.pre_feedforward_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        let post_ff_normed = self.post_feedforward_layernorm.forward(&ff_out);
        mlxcel_core::add(&h, &post_ff_normed)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
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

// Gemma 2 Model.
pub struct Gemma2Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: GemmaRMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub final_logit_softcapping: f32,
}

impl Gemma2Model {
    /// Forward pass through the entire model
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens and multiply by sqrt(hidden_size)
        let mut h = self.embed_tokens.forward(input_ids);
        let hidden_size = mlxcel_core::array_shape(&h)[2]; // [..., hidden_size]
        let scale = (hidden_size as f32).sqrt();
        h = mlxcel_core::multiply_scalar(&h, scale);

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        let mut logits = if let Some(head) = &self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        };

        // Apply final logit softcapping
        logits = softcap(&logits, self.final_logit_softcapping);

        logits
    }

    /// Create KV caches for all layers
    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Get raw token embeddings without scaling (for VLM merge)
    ///
    /// Returns raw embeddings without sqrt(hidden_size) normalization.
    /// The normalization is applied in forward_with_embeddings_impl to ALL
    /// embeddings (text + image), matching Python GemmaModel behavior.
    /// Used by: PaliGemma VLM
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Forward with pre-computed embeddings (for VLM prefill)
    ///
    /// Always applies sqrt(hidden_size) normalization to match Python
    /// GemmaModel.__call__ which scales ALL embeddings (lines 227-228).
    /// Used by: PaliGemma VLM
    pub fn forward_with_embeddings_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        // Apply sqrt(hidden_size) normalization to ALL embeddings.
        // Python: normalizer = sqrt(config.hidden_size); h = h * normalizer
        // Used by: PaliGemma VLM (Gemma2 backbone)
        let hidden_size = mlxcel_core::array_shape(&h)[2];
        let scale = (hidden_size as f32).sqrt();
        h = mlxcel_core::multiply_scalar(&h, scale);

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.norm.forward(&h);

        let mut logits = if let Some(head) = &self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        };

        logits = softcap(&logits, self.final_logit_softcapping);

        logits
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

        // Load weights
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

        // Load LM head (or use tied embeddings)
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        let final_logit_softcapping = args.final_logit_softcapping();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping,
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
impl LanguageModel for Gemma2Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Gemma2Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Gemma2Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.get_embed_tokens(input_ids))
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![1, 107] // Gemma2 EOS tokens: <eos> (1) and <end_of_turn> (107)
    }
}
