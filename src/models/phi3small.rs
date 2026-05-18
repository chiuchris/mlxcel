//! Phi3Small model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - Fused query_key_value projection
//! - GeGELU activation (gelu * (linear + 1))
//! - mup scaling (attention, embedding, width multipliers)
//! - Blocksparse attention pattern (with dense fallback)
//! - LayerNorm (not RMSNorm)
//! - bias=True for all projections
//! - Tied word embeddings

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedLinear, UnifiedEmbedding};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub dense_attention_every_n_layers: usize,
    pub ff_intermediate_size: usize,
    pub gegelu_limit: f32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub layer_norm_epsilon: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,

    #[serde(default = "default_mup_attn")]
    pub mup_attn_multiplier: f32,

    #[serde(default = "default_mup_scaling")]
    pub mup_use_scaling: bool,

    #[serde(default = "default_mup_emb")]
    pub mup_embedding_multiplier: f32,

    #[serde(default = "default_mup_width")]
    pub mup_width_multiplier: f32,

    #[serde(default = "default_rope_base")]
    pub rope_embedding_base: f32,

    #[serde(default = "default_rope_scale")]
    pub rope_position_scale: f32,

    #[serde(default = "default_block_size")]
    pub blocksparse_block_size: usize,

    #[serde(default = "default_local_blocks")]
    pub blocksparse_num_local_blocks: usize,

    #[serde(default = "default_vert_stride")]
    pub blocksparse_vert_stride: usize,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_mup_attn() -> f32 {
    1.0
}
fn default_mup_scaling() -> bool {
    true
}
fn default_mup_emb() -> f32 {
    10.0
}
fn default_mup_width() -> f32 {
    8.0
}
fn default_rope_base() -> f32 {
    1000000.0
}
fn default_rope_scale() -> f32 {
    1.0
}
fn default_block_size() -> usize {
    64
}
fn default_local_blocks() -> usize {
    16
}
fn default_vert_stride() -> usize {
    8
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
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
}

// ============================================================================
// Attention with Fused QKV and mup scaling
// ============================================================================

pub struct Attention {
    pub query_key_value: UnifiedLinear, // Fused Q, K, V projection
    pub dense: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub n_q_per_kv: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub block_sparse: bool, // Whether this layer uses block sparse attention
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

        // Fused QKV projection
        let qkv = self.query_key_value.forward(x);

        // Reshape to [B, L, n_kv_heads, n_q_per_kv + 2, head_dim]
        let qkv = mlxcel_core::reshape(
            &qkv,
            &[b, l, self.num_kv_heads, self.n_q_per_kv + 2, self.head_dim],
        );

        // Split into Q, K, V using slicing
        // queries = qkv[..., :-2, :] -> qkv[..., :n_q_per_kv, :]
        // keys = qkv[..., -2, :]
        // values = qkv[..., -1, :]

        // Q: slice [0:n_q_per_kv] on axis 3, then reshape
        let q = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, 0, 0],
            &[b, l, self.num_kv_heads, self.n_q_per_kv, self.head_dim],
        );
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);

        // K: slice [n_q_per_kv:n_q_per_kv+1] on axis 3, then squeeze
        let k = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, self.n_q_per_kv, 0],
            &[b, l, self.num_kv_heads, self.n_q_per_kv + 1, self.head_dim],
        );
        let k = mlxcel_core::squeeze_axis(&k, 3);

        // V: slice [n_q_per_kv+1:n_q_per_kv+2] on axis 3, then squeeze
        let v = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, self.n_q_per_kv + 1, 0],
            &[b, l, self.num_kv_heads, self.n_q_per_kv + 2, self.head_dim],
        );
        let v = mlxcel_core::squeeze_axis(&v, 3);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let mut k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE
        q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        // Note: block sparse attention would be more efficient for long sequences,
        // but we use dense attention for simplicity (fallback)
        let attn_out = if l > 1 && mask.is_none() {
            mlxcel_core::fast_scaled_dot_product_attention_causal(
                &q, &cache_k, &cache_v, self.scale,
            )
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::fast_scaled_dot_product_attention(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr,
                )
            }
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.dense.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let query_key_value = UnifiedLinear::from_weights(
            weights,
            &format!("{}.query_key_value", prefix),
            group_size,
            bits,
        )?;
        let dense =
            UnifiedLinear::from_weights(weights, &format!("{}.dense", prefix), group_size, bits)?;

        let head_dim = args.head_dim() as i32;
        let n_q_per_kv = (args.num_attention_heads / args.num_key_value_heads) as i32;

        // mup scaling for attention
        let scale = if args.mup_use_scaling {
            1.0 / (head_dim as f32 / args.mup_attn_multiplier)
        } else {
            1.0 / (head_dim as f32).sqrt()
        };

        // Block sparse for non-dense layers
        let block_sparse = !(layer_idx + 1).is_multiple_of(args.dense_attention_every_n_layers);

        Ok(Self {
            query_key_value,
            dense,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            n_q_per_kv,
            head_dim,
            scale,
            rope_dims: head_dim,
            rope_base: args.rope_embedding_base,
            block_sparse,
        })
    }
}

// ============================================================================
// MLP with GeGELU activation
// ============================================================================

pub struct MLP {
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
    pub gegelu_limit: f32,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.up_proj.forward(x);
        let h = mlxcel_core::utils::gegelu(&h, self.gegelu_limit);
        self.down_proj.forward(&h)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let up_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.up_proj", prefix),
            group_size,
            bits,
        )?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            up_proj,
            down_proj,
            gegelu_limit: args.gegelu_limit,
        })
    }
}

// ============================================================================
// Transformer Block
// ============================================================================

pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: LayerNorm,
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
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm FFN
        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
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

        // LayerNorm with bias
        let input_ln_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let input_ln_bias = weights
            .get(&format!("{}.input_layernorm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let input_layernorm =
            LayerNorm::new(input_ln_weight, input_ln_bias, args.layer_norm_epsilon);

        let post_ln_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let post_ln_bias = weights
            .get(&format!("{}.post_attention_layernorm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let post_attention_layernorm =
            LayerNorm::new(post_ln_weight, post_ln_bias, args.layer_norm_epsilon);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// ============================================================================
// Phi3Small Model
// ============================================================================

pub struct Phi3SmallModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub final_layernorm: LayerNorm,
    pub mup_embedding_multiplier: f32,
    pub mup_width_multiplier: f32,
}

impl Phi3SmallModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens with mup scaling
        let mut h = self.embed_tokens.forward(input_ids);
        if self.mup_embedding_multiplier != 1.0 {
            h = mlxcel_core::multiply_scalar(&h, self.mup_embedding_multiplier);
        }

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.final_layernorm.forward(&h);

        // LM head (tied embeddings) with mup width scaling
        let mut logits = self.embed_tokens.as_linear(&h);
        if self.mup_width_multiplier != 1.0 {
            logits = mlxcel_core::divide_scalar(&logits, self.mup_width_multiplier);
        }
        logits
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

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

        // Load final layernorm with bias
        let norm_weight = get_weight_copy(weights, "model.final_layernorm.weight")?;
        let norm_bias = weights
            .get("model.final_layernorm.bias")
            .map(|w| mlxcel_core::copy(w));
        let final_layernorm = LayerNorm::new(norm_weight, norm_bias, args.layer_norm_epsilon);

        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            mup_embedding_multiplier: args.mup_embedding_multiplier,
            mup_width_multiplier: args.mup_width_multiplier,
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// ============================================================================
// LanguageModel trait implementation
// ============================================================================

impl LanguageModel for Phi3SmallModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Phi3SmallModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Phi3SmallModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![32000] // Phi3Small EOS token
    }
}
