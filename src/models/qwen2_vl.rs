//! Qwen2-VL Language Model with MRoPE (Multi-dimensional Rotary Position Embeddings)
//!
//! Based on Qwen2 architecture (GQA, SwiGLU) but with MRoPE instead of standard RoPE.
//! MRoPE uses 3D position IDs [T, H, W] for encoding spatial structure of vision tokens.
//!
//! Used by: Qwen2-VL
//! Reference: references/mlx-vlm/mlx_vlm/models/qwen2_vl/language.py

use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::cell::RefCell;

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2VLConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<i32>,
    #[serde(rename = "type", default)]
    pub scaling_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_num_kv_heads() -> Option<usize> {
    None
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1000000.0
}
fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}

impl Qwen2VLConfig {
    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(0)
    }
    fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(0)
    }
    fn mrope_section(&self) -> Vec<i32> {
        self.rope_scaling
            .as_ref()
            .map(|rs| rs.mrope_section.clone())
            .unwrap_or_else(|| vec![16, 24, 24])
    }
}

// ============================================================================
// MRoPE (Multimodal Rotary Position Embedding)
// ============================================================================

struct MRoPE {
    inv_freq: Vec<f32>,
    mrope_section: Vec<i32>,
}

impl MRoPE {
    fn new(dim: usize, base: f32, mrope_section: Vec<i32>) -> Self {
        let mut inv_freq = Vec::with_capacity(dim / 2);
        for i in (0..dim).step_by(2) {
            inv_freq.push(1.0 / base.powf(i as f32 / dim as f32));
        }
        Self {
            inv_freq,
            mrope_section,
        }
    }

    /// Compute cos/sin for MRoPE
    /// position_ids: [3, batch, seq_len] for multimodal, or [batch, seq_len] for text-only
    /// Returns (cos, sin) each [batch, seq_len, head_dim]
    fn forward(&self, position_ids: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let pos_shape = mlxcel_core::array_shape(position_ids);

        // If 2D, broadcast to [3, batch, seq_len]
        let position_ids_3d = if pos_shape.len() == 2 {
            let expanded = mlxcel_core::expand_dims(position_ids, 0);
            mlxcel_core::broadcast_to(&expanded, &[3, pos_shape[0], pos_shape[1]])
        } else {
            mlxcel_core::copy(position_ids)
        };

        let pos_shape = mlxcel_core::array_shape(&position_ids_3d);
        let batch = pos_shape[1];
        let seq_len = pos_shape[2];
        let half_dim = self.inv_freq.len() as i32;

        // inv_freq: [half_dim] -> [1, 1, half_dim, 1]
        let inv_freq_arr = mlxcel_core::from_slice_f32(&self.inv_freq, &[half_dim]);
        let inv_freq_arr = mlxcel_core::astype(&inv_freq_arr, mlxcel_core::dtype::FLOAT32);
        let inv_freq_4d = mlxcel_core::reshape(&inv_freq_arr, &[1, 1, half_dim, 1]);
        // Broadcast: [3, batch, half_dim, 1]
        let inv_freq_4d = mlxcel_core::broadcast_to(&inv_freq_4d, &[3, batch, half_dim, 1]);

        // position_ids: [3, batch, seq_len] -> [3, batch, 1, seq_len]
        let pos_expanded = mlxcel_core::reshape(&position_ids_3d, &[3, batch, 1, seq_len]);
        let pos_expanded = mlxcel_core::astype(&pos_expanded, mlxcel_core::dtype::FLOAT32);

        // freqs = inv_freq @ position_ids: [3, batch, half_dim, seq_len]
        let freqs = mlxcel_core::matmul(&inv_freq_4d, &pos_expanded);
        // Transpose: [3, batch, seq_len, half_dim]
        let freqs = mlxcel_core::transpose_axes(&freqs, &[0, 1, 3, 2]);

        // Apply MRoPE section mixing: combine T, H, W sections
        let freqs = self.apply_mrope(&freqs);
        // freqs: [batch, seq_len, half_dim]

        // Double the frequencies: [batch, seq_len, head_dim]
        let emb = mlxcel_core::concatenate(&freqs, &freqs, -1);

        let cos = mlxcel_core::cos(&emb);
        let sin = mlxcel_core::sin(&emb);

        (cos, sin)
    }

    /// Apply MRoPE: combine T/H/W sections
    /// freqs: [3, batch, seq_len, half_dim]
    /// Returns: [batch, seq_len, half_dim]
    fn apply_mrope(&self, freqs: &MlxArray) -> UniquePtr<MlxArray> {
        let freqs_shape = mlxcel_core::array_shape(freqs);
        let batch = freqs_shape[1];
        let seq_len = freqs_shape[2];
        let half_dim = freqs_shape[3];

        // Start with T (temporal) as base
        let mut result = mlxcel_core::slice(freqs, &[0, 0, 0, 0], &[1, batch, seq_len, half_dim]);
        result = mlxcel_core::squeeze_axis(&result, 0);

        // Replace sections with H and W
        let mut offset = self.mrope_section[0];
        for (dim_idx, &length) in self.mrope_section[1..].iter().enumerate() {
            let src_dim = dim_idx as i32 + 1; // H=1, W=2
            let end = offset + length;

            // Extract section from this dimension
            let section = mlxcel_core::slice(
                freqs,
                &[src_dim, 0, 0, offset],
                &[src_dim + 1, batch, seq_len, end],
            );
            let section = mlxcel_core::squeeze_axis(&section, 0);

            // Replace in result via slice_update
            mlxcel_core::slice_update(&result, &section, &[0, 0, offset], &[batch, seq_len, end]);

            offset = end;
        }

        result
    }
}

/// Apply MRoPE to Q and K tensors
/// q, k: [batch, heads, seq, head_dim]
/// cos, sin: [batch, seq, head_dim]
fn apply_multimodal_rotary_pos_emb(
    q: &MlxArray,
    k: &MlxArray,
    cos: &MlxArray,
    sin: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Expand: [batch, seq, head_dim] -> [batch, 1, seq, head_dim]
    let cos = mlxcel_core::expand_dims(cos, 1);
    let sin = mlxcel_core::expand_dims(sin, 1);

    let q_embed = {
        let t1 = mlxcel_core::multiply(q, &cos);
        let r = rotate_half(q);
        let t2 = mlxcel_core::multiply(&r, &sin);
        mlxcel_core::add(&t1, &t2)
    };
    let k_embed = {
        let t1 = mlxcel_core::multiply(k, &cos);
        let r = rotate_half(k);
        let t2 = mlxcel_core::multiply(&r, &sin);
        mlxcel_core::add(&t1, &t2)
    };

    (q_embed, k_embed)
}

fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let half = shape[shape.len() - 1] / 2;
    let ndim = shape.len();

    let mut starts = vec![0i32; ndim];
    let mut stops = shape.clone();
    stops[ndim - 1] = half;
    let x1 = mlxcel_core::slice(x, &starts, &stops);

    starts[ndim - 1] = half;
    stops[ndim - 1] = shape[ndim - 1];
    let x2 = mlxcel_core::slice(x, &starts, &stops);

    let neg_x2 = mlxcel_core::negative(&x2);
    mlxcel_core::concatenate(&neg_x2, &x1, ndim as i32 - 1)
}

// ============================================================================
// Attention with MRoPE
// ============================================================================

struct Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    mrope: MRoPE,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen2VLConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        let q_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.q_proj", prefix),
            gs,
            bits,
        )?;
        let k_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.k_proj", prefix),
            gs,
            bits,
        )?;
        let v_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.v_proj", prefix),
            gs,
            bits,
        )?;
        let o_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.o_proj", prefix),
            gs,
            bits,
        )?;

        let head_dim = config.head_dim();
        let mrope = MRoPE::new(head_dim, config.rope_theta, config.mrope_section());

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            mrope,
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_kv_heads() as i32,
            head_dim: head_dim as i32,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape: [B, L, dim] -> [B, heads, L, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Apply MRoPE
        let (cos, sin) = self.mrope.forward(position_ids);
        let (q, k) = apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin);

        // KV cache
        let (k, v) = cache.update_and_fetch(k, v);

        // Repeat KV heads if GQA
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 {
            mlxcel_core::utils::repeat_kv(&k, n_rep)
        } else {
            mlxcel_core::copy(&k)
        };
        let v = if n_rep > 1 {
            mlxcel_core::utils::repeat_kv(&v, n_rep)
        } else {
            mlxcel_core::copy(&v)
        };

        // Attention
        let output = if let Some(m) = mask {
            unsafe {
                mlxcel_core::fast_scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    m as *const MlxArray,
                )
            }
        } else {
            unsafe {
                mlxcel_core::fast_scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    std::ptr::null(),
                )
            }
        };

        // [B, heads, L, head_dim] -> [B, L, dim]
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        self.o_proj.forward(&output)
    }
}

// ============================================================================
// MLP (SwiGLU, same as standard Qwen2/Llama)
// ============================================================================

struct MLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl MLP {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen2VLConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.gate_proj", prefix),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.up_proj", prefix),
                gs,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.down_proj", prefix),
                gs,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

struct DecoderLayer {
    attn: Attention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen2VLConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            attn: Attention::from_weights(weights, config, prefix)?,
            mlp: MLP::from_weights(weights, config, prefix)?,
            input_layernorm: load_rms_norm(
                weights,
                &format!("{}.input_layernorm", prefix),
                config.rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{}.post_attention_layernorm", prefix),
                config.rms_norm_eps,
            )?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let r = self
            .attn
            .forward(&self.input_layernorm.forward(x), cache, mask, position_ids);
        let h = mlxcel_core::add(x, &r);
        let r = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &r)
    }
}

fn load_rms_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    let key = format!("{}.weight", prefix);
    let weight = weights
        .get(&key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", key))?;
    Ok(RMSNorm::new(weight, eps))
}

// ============================================================================
// Qwen2VLModel - Full language model
// ============================================================================

pub struct Qwen2VLModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    _config: Qwen2VLConfig,
    /// Cached MRoPE state across prefill/decode
    rope_deltas: RefCell<Option<i32>>,
    position_ids: RefCell<Option<UniquePtr<MlxArray>>>,
}

impl Qwen2VLModel {
    pub fn from_weights(weights: &WeightMap, config: &Qwen2VLConfig) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();

        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(
                weights,
                config,
                &format!("model.layers.{}", i),
            )?);
        }

        let norm = load_rms_norm(weights, "model.norm", config.rms_norm_eps)?;

        let lm_head = if config.tie_word_embeddings {
            UnifiedLinear::from_weights(weights, "model.embed_tokens", gs, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            _config: config.clone(),
            rope_deltas: RefCell::new(None),
            position_ids: RefCell::new(None),
        })
    }

    /// Set MRoPE state after vision processing
    pub fn set_mrope_state(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        *self.position_ids.borrow_mut() = Some(position_ids);
        *self.rope_deltas.borrow_mut() = Some(rope_deltas);
    }

    /// Clear MRoPE state (for new image/video)
    pub fn clear_mrope_state(&self) {
        *self.position_ids.borrow_mut() = None;
        *self.rope_deltas.borrow_mut() = None;
    }

    /// Forward pass
    pub fn forward_impl(
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

        let ids_shape = mlxcel_core::array_shape(input_ids);
        let batch = ids_shape[0];
        let seq_len = ids_shape[1];
        let cache_offset = caches[0].offset;

        // Compute position_ids
        let position_ids = {
            let stored = self.position_ids.borrow();
            if let Some(ref stored_pos) = *stored {
                // Use stored position_ids (from prefill with images)
                if cache_offset == 0 {
                    // Prefill: slice the full position_ids
                    mlxcel_core::copy(stored_pos)
                } else {
                    // Decode: slice from cache_offset
                    let pos_shape = mlxcel_core::array_shape(stored_pos);
                    if pos_shape.len() == 3 && cache_offset < pos_shape[2] {
                        mlxcel_core::slice(
                            stored_pos,
                            &[0, 0, cache_offset],
                            &[pos_shape[0], pos_shape[1], cache_offset + seq_len],
                        )
                    } else {
                        // Fall back to delta-based position computation
                        self.compute_decode_position_ids(batch, seq_len, cache_offset)
                    }
                }
            } else if cache_offset > 0 {
                // Decode without stored positions (text-only after first forward)
                self.compute_decode_position_ids(batch, seq_len, cache_offset)
            } else {
                // Text-only prefill: standard sequential positions
                let pos = mlxcel_core::arange_i32(0, seq_len, 1);
                let pos = mlxcel_core::reshape(&pos, &[1, seq_len]);
                let pos = mlxcel_core::broadcast_to(&pos, &[batch, seq_len]);
                let pos = mlxcel_core::expand_dims(&pos, 0);
                mlxcel_core::broadcast_to(&pos, &[3, batch, seq_len])
            }
        };

        // Create causal mask if needed
        let auto_mask;
        let mask = if mask.is_some() {
            mask
        } else {
            auto_mask = mlxcel_core::utils::create_causal_mask(seq_len, cache_offset);
            Some(auto_mask.as_ref().unwrap() as &MlxArray)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask, &position_ids);
        }

        h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    /// Compute position_ids for decode steps using rope_deltas
    fn compute_decode_position_ids(
        &self,
        batch: i32,
        seq_len: i32,
        cache_offset: i32,
    ) -> UniquePtr<MlxArray> {
        let delta = self.rope_deltas.borrow().unwrap_or(0);
        let offset = cache_offset + delta;

        let pos = mlxcel_core::arange_i32(offset, offset + seq_len, 1);
        let pos = mlxcel_core::reshape(&pos, &[1, seq_len]);
        let pos = mlxcel_core::broadcast_to(&pos, &[batch, seq_len]);
        let pos = mlxcel_core::expand_dims(&pos, 0);
        mlxcel_core::broadcast_to(&pos, &[3, batch, seq_len])
    }

    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }
}

// ============================================================================
// LanguageModel trait implementation
// ============================================================================

impl mlxcel_core::generate::LanguageModel for Qwen2VLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, None, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Qwen2VLModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![151645, 151643] // Qwen2 EOS tokens
    }
}
