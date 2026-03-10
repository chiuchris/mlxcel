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

//! Qwen3.5: Hybrid Transformer + GatedDeltaNet (Linear Attention) + MoE
//!
//! Key differences from Qwen3Next:
//! - GatedDeltaNet uses 4 separate projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a)
//!   instead of 2 combined (in_proj_qkvz, in_proj_ba)
//! - Config uses rope_parameters dict instead of flat rope_theta/partial_rotary_factor
//! - Weight sanitization handles MTP weights and norm weight shifting
//! - MoE variant (qwen3_5_moe) uses text_config indirection and gate_up_proj split
//!
//! Reuses from qwen3_next: Qwen3NextAttention, MLP, SparseMoeBlock, SwitchGLU, SwitchLinear
//!
//! Reference: mlx-lm/mlx_lm/models/qwen3_5.py

use crate::models::gated_delta::{GatedDeltaCache, RMSNormGated, gated_delta_update};
use crate::models::qwen3_next::{
    MLP, Quantization, Qwen3NextAttention, Qwen3NextCache, Qwen3NextConfig, SparseMoeBlock,
};
use mlxcel_core::dtype;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, silu, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::cell::RefCell;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,

    // Linear attention parameters
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,

    // MoE parameters (0 = dense)
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,

    // Rope parameters (dict format)
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,

    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    pub vocab_size: usize,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_full_attention_interval() -> usize {
    4
}
fn default_linear_num_value_heads() -> usize {
    64
}
fn default_linear_num_key_heads() -> usize {
    16
}
fn default_linear_key_head_dim() -> usize {
    192
}
fn default_linear_value_head_dim() -> usize {
    128
}
fn default_linear_conv_kernel_dim() -> usize {
    4
}
fn default_decoder_sparse_step() -> usize {
    1
}
fn default_true() -> bool {
    true
}

impl Qwen35Config {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(100000.0)
    }

    fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.25)
    }

    fn head_dim_resolved(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_dims(&self) -> i32 {
        (self.head_dim_resolved() as f32 * self.partial_rotary_factor()) as i32
    }

    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        !(layer_idx + 1).is_multiple_of(self.full_attention_interval)
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    /// Convert to Qwen3NextConfig for reusing shared components
    pub fn to_qwen3next_config(&self) -> Qwen3NextConfig {
        Qwen3NextConfig {
            model_type: self.model_type.clone(),
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim_resolved(),
            linear_num_value_heads: self.linear_num_value_heads,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            decoder_sparse_step: self.decoder_sparse_step,
            moe_intermediate_size: self.moe_intermediate_size,
            shared_expert_intermediate_size: self.shared_expert_intermediate_size,
            mlp_only_layers: self.mlp_only_layers.clone(),
            full_attention_interval: self.full_attention_interval,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            rope_theta: self.rope_theta(),
            partial_rotary_factor: self.partial_rotary_factor(),
            max_position_embeddings: None,
            norm_topk_prob: self.norm_topk_prob,
            tie_word_embeddings: self.tie_word_embeddings,
            attention_bias: self.attention_bias,
            quantization: self.quantization.clone(),
        }
    }
}

// GatedDeltaNet - Qwen3.5 variant with separate projections.
/// GatedDeltaNet for Qwen3.5 with separate in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
#[allow(dead_code)]
struct Qwen35GatedDeltaNet {
    hidden_size: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv_dim: usize,

    conv1d_weight: UniquePtr<MlxArray>,
    in_proj_qkv: UnifiedLinear,
    in_proj_z: UnifiedLinear,
    in_proj_b: UnifiedLinear,
    in_proj_a: UnifiedLinear,
    dt_bias: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    norm: RMSNormGated,
    out_proj: UnifiedLinear,
}

impl Qwen35GatedDeltaNet {
    fn forward(
        &self,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        mut cache: Option<&mut GatedDeltaCache>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs);
        let b = shape[0];
        let s = shape[1];

        // Separate projections (different from Qwen3Next's combined projections)
        let qkv = self.in_proj_qkv.forward(inputs);
        let z = self.in_proj_z.forward(inputs);
        let z = mlxcel_core::reshape(&z, &[b, s, self.num_v_heads as i32, self.head_v_dim as i32]);
        let b_proj = self.in_proj_b.forward(inputs);
        let a = self.in_proj_a.forward(inputs);

        // Get conv state from cache
        let conv_state = if let Some(ref c) = cache {
            c.conv_state
                .as_ref()
                .map(|s| mlxcel_core::copy(s.as_ref().unwrap()))
                .unwrap_or_else(|| {
                    mlxcel_core::zeros(
                        &[b, (self.conv_kernel_size - 1) as i32, self.conv_dim as i32],
                        dtype::FLOAT32,
                    )
                })
        } else {
            mlxcel_core::zeros(
                &[b, (self.conv_kernel_size - 1) as i32, self.conv_dim as i32],
                dtype::FLOAT32,
            )
        };

        // Apply mask if present (mask qkv before conv)
        let qkv = if let Some(m) = mask {
            let m_exp = mlxcel_core::expand_dims(m, -1);
            let zero = mlxcel_core::full_f32(&[1], 0.0, dtype::FLOAT32);
            mlxcel_core::where_cond(&m_exp, &qkv, &zero)
        } else {
            qkv
        };

        // Concatenate with conv state
        let conv_input = concatenate(&conv_state, &qkv, 1);

        // Update cache with new conv state
        if let Some(c) = cache.as_deref_mut() {
            let n_keep = (self.conv_kernel_size - 1) as i32;
            let conv_shape = mlxcel_core::array_shape(&conv_input);
            let conv_len = conv_shape[1];
            c.conv_state = Some(mlxcel_core::slice(
                &conv_input,
                &[0, conv_len - n_keep, 0],
                &[b, conv_len, self.conv_dim as i32],
            ));
        }

        // Apply conv1d with SiLU activation
        let conv_out = mlxcel_core::conv1d(
            &conv_input,
            &self.conv1d_weight,
            1,
            0,
            1,
            self.conv_dim as i32,
        );
        let conv_out = silu(&conv_out);

        // Split conv output into q, k, v
        // Note: MLX slice with stop=-1 means dim_size-1 (excludes last), not "to end"
        // Use actual conv_out seq length for correct slicing
        let conv_out_shape = mlxcel_core::array_shape(&conv_out);
        let conv_seq = conv_out_shape[1];
        let q_out = mlxcel_core::slice(&conv_out, &[0, 0, 0], &[b, conv_seq, self.key_dim as i32]);
        let k_out = mlxcel_core::slice(
            &conv_out,
            &[0, 0, self.key_dim as i32],
            &[b, conv_seq, (2 * self.key_dim) as i32],
        );
        let v_out = mlxcel_core::slice(
            &conv_out,
            &[0, 0, (2 * self.key_dim) as i32],
            &[b, conv_seq, self.conv_dim as i32],
        );

        // Reshape to heads
        let q = mlxcel_core::reshape(
            &q_out,
            &[b, s, self.num_k_heads as i32, self.head_k_dim as i32],
        );
        let k = mlxcel_core::reshape(
            &k_out,
            &[b, s, self.num_k_heads as i32, self.head_k_dim as i32],
        );
        let v = mlxcel_core::reshape(
            &v_out,
            &[b, s, self.num_v_heads as i32, self.head_v_dim as i32],
        );

        // Get recurrent state from cache
        let state = cache.as_ref().and_then(|c| {
            c.state_cache
                .as_ref()
                .map(|s| mlxcel_core::copy(s.as_ref().unwrap()))
        });

        // Apply RMS norm with scaling (same as Qwen3Next)
        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let eps_arr = mlxcel_core::full_f32(&[1], 1e-6, dtype::FLOAT32);

        let q_sq = mlxcel_core::square(&q);
        let q_sq_mean = mlxcel_core::mean_axis(&q_sq, -1, true);
        let q_rms = mlxcel_core::sqrt(&mlxcel_core::add(&q_sq_mean, &eps_arr));
        let scale_q = mlxcel_core::full_f32(&[1], inv_scale * inv_scale, dtype::FLOAT32);
        let q = mlxcel_core::multiply(&mlxcel_core::divide(&q, &q_rms), &scale_q);

        let k_sq = mlxcel_core::square(&k);
        let k_sq_mean = mlxcel_core::mean_axis(&k_sq, -1, true);
        let k_rms = mlxcel_core::sqrt(&mlxcel_core::add(&k_sq_mean, &eps_arr));
        let scale_k = mlxcel_core::full_f32(&[1], inv_scale, dtype::FLOAT32);
        let k = mlxcel_core::multiply(&mlxcel_core::divide(&k, &k_rms), &scale_k);

        // Run gated delta update
        let (out, new_state) = gated_delta_update(
            &q,
            &k,
            &v,
            &a,
            &b_proj,
            &self.a_log,
            &self.dt_bias,
            state.as_deref(),
            mask,
        );

        // Update cache state
        if let Some(c) = cache {
            c.state_cache = Some(new_state);
            c.advance(s);
        }

        // Apply norm with gating
        let out = self.norm.forward(&out, Some(&z));
        let out = mlxcel_core::reshape(&out, &[b, s, -1]);

        self.out_proj.forward(&out)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Qwen35Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = config.linear_conv_kernel_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let group_size = config.group_size();
        let bits = config.bits();

        let conv1d_weight = weights
            .get(&format!("{}.conv1d.weight", prefix))
            .map(|w| {
                let shape = mlxcel_core::array_shape(w);
                if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                    mlxcel_core::swap_axes(w, -1, -2)
                } else {
                    mlxcel_core::copy(w)
                }
            })
            .ok_or_else(|| format!("Missing conv1d weight: {}", prefix))?;

        // Qwen3.5 uses separate projections instead of combined
        let in_proj_qkv = UnifiedLinear::from_weights(
            weights,
            &format!("{}.in_proj_qkv", prefix),
            group_size,
            bits,
        )?;
        let in_proj_z = UnifiedLinear::from_weights(
            weights,
            &format!("{}.in_proj_z", prefix),
            group_size,
            bits,
        )?;
        let in_proj_b = UnifiedLinear::from_weights(
            weights,
            &format!("{}.in_proj_b", prefix),
            group_size,
            bits,
        )?;
        let in_proj_a = UnifiedLinear::from_weights(
            weights,
            &format!("{}.in_proj_a", prefix),
            group_size,
            bits,
        )?;

        let dt_bias = weights
            .get(&format!("{}.dt_bias", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing dt_bias: {}", prefix))?;

        let a_log = weights
            .get(&format!("{}.A_log", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing A_log: {}", prefix))?;

        let norm_weight = weights
            .get(&format!("{}.norm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing norm weight: {}", prefix))?;

        let out_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.out_proj", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            hidden_size,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv_dim,
            conv1d_weight,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            dt_bias,
            a_log,
            norm: RMSNormGated::new(norm_weight, config.rms_norm_eps),
            out_proj,
        })
    }
}

// Decoder Layer.
/// Attention variant for Qwen3.5
enum Qwen35AttentionVariant {
    FullAttention(Qwen3NextAttention),
    Linear(Qwen35GatedDeltaNet),
}

/// MLP variant for Qwen3.5
enum Qwen35MLPVariant {
    Dense(MLP),
    MoE(SparseMoeBlock),
}

struct Qwen35DecoderLayer {
    is_linear: bool,
    attention: Qwen35AttentionVariant,
    mlp: Qwen35MLPVariant,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl Qwen35DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Qwen3NextCache,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);

        let r = match (&self.attention, cache) {
            (Qwen35AttentionVariant::Linear(attn), Qwen3NextCache::Linear(c)) => {
                attn.forward(&normed, mask, Some(c))
            }
            (Qwen35AttentionVariant::Linear(attn), _) => attn.forward(&normed, mask, None),
            (Qwen35AttentionVariant::FullAttention(attn), Qwen3NextCache::Attention(c)) => {
                attn.forward_with_position_ids(&normed, c, mask, position_ids)
            }
            (Qwen35AttentionVariant::FullAttention(attn), _) => {
                let mut temp_cache = KVCache::new();
                attn.forward_with_position_ids(&normed, &mut temp_cache, mask, position_ids)
            }
        };

        let h = mlxcel_core::add(x, &r);

        let mlp_out = match &self.mlp {
            Qwen35MLPVariant::Dense(mlp) => mlp.forward(&self.post_attention_layernorm.forward(&h)),
            Qwen35MLPVariant::MoE(moe) => moe.forward(&self.post_attention_layernorm.forward(&h)),
        };
        mlxcel_core::add(&h, &mlp_out)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Qwen35Config,
        qn_config: &Qwen3NextConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);
        let is_linear = config.is_linear_layer(layer_idx);

        let attention = if is_linear {
            Qwen35AttentionVariant::Linear(Qwen35GatedDeltaNet::from_weights(
                weights,
                config,
                &format!("{}.linear_attn", prefix),
            )?)
        } else {
            Qwen35AttentionVariant::FullAttention(Qwen3NextAttention::from_weights(
                weights,
                qn_config,
                &format!("{}.self_attn", prefix),
            )?)
        };

        let mlp = if config.is_moe_layer(layer_idx) {
            Qwen35MLPVariant::MoE(SparseMoeBlock::from_weights(
                weights,
                qn_config,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            Qwen35MLPVariant::Dense(MLP::from_weights(
                weights,
                qn_config,
                &format!("{}.mlp", prefix),
            )?)
        };

        let input_norm_weight = weights
            .get(&format!("{}.input_layernorm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing input_layernorm: {}", prefix))?;

        let post_norm_weight = weights
            .get(&format!("{}.post_attention_layernorm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing post_attention_layernorm: {}", prefix))?;

        Ok(Self {
            is_linear,
            attention,
            mlp,
            input_layernorm: RMSNorm::new(input_norm_weight, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_norm_weight, config.rms_norm_eps),
        })
    }
}

// Qwen3.5 Model.
pub struct Qwen35Model {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<Qwen35DecoderLayer>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    config: Qwen35Config,
    /// Internal caches for LanguageModel trait compatibility
    /// Using RefCell to allow mutation through shared reference (required by trait)
    internal_caches: RefCell<Vec<Qwen3NextCache>>,
    /// MRoPE position_ids for VLM [3, batch, seq_len]
    position_ids: RefCell<Option<UniquePtr<MlxArray>>>,
    /// Rope deltas for token generation after VLM prefill
    rope_deltas: RefCell<Option<i32>>,
}

impl Qwen35Model {
    fn forward_internal(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Qwen3NextCache],
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];

        // Create masks
        let fa_idx = self.config.full_attention_interval - 1;
        let fa_mask = if seq_len > 1 {
            let offset = if fa_idx < caches.len() {
                caches[fa_idx].offset()
            } else {
                0
            };
            Some(create_causal_mask(seq_len, offset))
        } else {
            None
        };

        // SSM mask: for linear attention layers
        // None means all tokens are valid, which covers:
        // - Generation (L=1): single token always valid
        // - Full prefill (no prior cache): all tokens valid
        // The only case needing a non-None SSM mask is resuming prefill after
        // partial generation, which is rare and can be added later.

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let mask = if layer.is_linear {
                None
            } else {
                fa_mask.as_deref()
            };
            h = layer.forward(&h, mask, cache, position_ids);
        }

        let h = self.norm.forward(&h);

        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    fn make_internal_caches(&self) -> Vec<Qwen3NextCache> {
        self.layers
            .iter()
            .map(|l| {
                if l.is_linear {
                    Qwen3NextCache::Linear(GatedDeltaCache::new())
                } else {
                    Qwen3NextCache::Attention(KVCache::new())
                }
            })
            .collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, Qwen35Config), String> {
        let model_dir = model_dir.as_ref();

        println!("[Qwen3.5] Loading config...");
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let v: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Handle text_config indirection (VLM wrapper format)
        let mut text_config_val = if let Some(tc) = v.get("text_config") {
            tc.clone()
        } else {
            v.clone()
        };

        // Merge quantization from top level if text_config doesn't have it
        if text_config_val.get("quantization").is_none() && v.get("quantization").is_some() {
            text_config_val
                .as_object_mut()
                .unwrap()
                .insert("quantization".to_string(), v["quantization"].clone());
        }

        let config: Qwen35Config = serde_json::from_value(text_config_val)
            .map_err(|e| format!("Failed to parse config: {}", e))?;

        println!(
            "[Qwen3.5] Config loaded: {} layers ({} full attention, {} linear attention)",
            config.num_hidden_layers,
            (0..config.num_hidden_layers)
                .filter(|&i| !config.is_linear_layer(i))
                .count(),
            (0..config.num_hidden_layers)
                .filter(|&i| config.is_linear_layer(i))
                .count(),
        );

        println!("[Qwen3.5] Loading weights...");
        let weights = crate::models::load_and_sanitize_weights(model_dir)?;

        // Strip language_model. prefix and sanitize
        let weights = sanitize_moe_weights(weights, &config);

        println!("[Qwen3.5] Building model...");
        let model = Self::from_weights(&weights, &config)?;

        println!("[Qwen3.5] Model loaded successfully");
        Ok((model, config))
    }

    pub fn from_weights(weights: &WeightMap, config: &Qwen35Config) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();
        let qn_config = config.to_qwen3next_config();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = Qwen35DecoderLayer::from_weights(weights, config, &qn_config, i)?;
            layers.push(layer);
        }

        let norm_weight = weights
            .get("model.norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Missing model.norm.weight".to_string())?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        let config_clone = config.clone();
        let internal_caches: Vec<Qwen3NextCache> = (0..config.num_hidden_layers)
            .map(|i| {
                if config.is_linear_layer(i) {
                    Qwen3NextCache::Linear(GatedDeltaCache::new())
                } else {
                    Qwen3NextCache::Attention(KVCache::new())
                }
            })
            .collect();

        Ok(Self {
            embed_tokens,
            layers,
            norm: RMSNorm::new(norm_weight, config.rms_norm_eps),
            lm_head,
            config: config_clone,
            internal_caches: RefCell::new(internal_caches),
            position_ids: RefCell::new(None),
            rope_deltas: RefCell::new(None),
        })
    }

    /// Set MRoPE state after vision processing (called by VLM wrapper)
    pub fn set_mrope_state(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        *self.position_ids.borrow_mut() = Some(position_ids);
        *self.rope_deltas.borrow_mut() = Some(rope_deltas);
    }

    /// Get token embeddings (used by VLM wrapper)
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Forward pass with VLM support
    ///
    /// Position IDs handling (for MRoPE VLM):
    /// - Prefill (cache_offset == 0, stored position_ids): use stored position_ids, then clear
    /// - Decode (cache_offset > 0, has rope_deltas): compute sequential position_ids with offset
    /// - Text-only (no rope_deltas): position_ids = None, uses fast_rope
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut internal = self.internal_caches.borrow_mut();

        // Get cache offset from first full attention layer's cache
        let cache_offset = internal
            .iter()
            .find_map(|c| match c {
                super::qwen3_next::Qwen3NextCache::Attention(kv) => Some(kv.offset),
                _ => None,
            })
            .unwrap_or(0);

        let ids_shape = mlxcel_core::array_shape(input_ids);
        let batch = ids_shape[0];
        let seq_len = ids_shape[1];

        // Compute position_ids for MRoPE
        let has_stored = self.position_ids.borrow().is_some();
        let has_deltas = self.rope_deltas.borrow().is_some();

        let position_ids = if has_stored && cache_offset == 0 {
            // Prefill: consume stored position_ids from VLM preprocessing
            self.position_ids.borrow_mut().take()
        } else if has_deltas {
            // Decode: compute sequential position_ids with rope_deltas offset
            let delta = self.rope_deltas.borrow().unwrap_or(0);
            let offset = cache_offset + delta;
            let pos = mlxcel_core::arange_i32(offset, offset + seq_len, 1);
            let pos = mlxcel_core::reshape(&pos, &[1, seq_len]);
            let pos = mlxcel_core::broadcast_to(&pos, &[batch, seq_len]);
            let pos = mlxcel_core::expand_dims(&pos, 0);
            Some(mlxcel_core::broadcast_to(&pos, &[3, batch, seq_len]))
        } else {
            // Text-only: no MRoPE, position_ids = None
            None
        };

        self.forward_internal(
            input_ids,
            input_embeddings,
            &mut internal,
            position_ids.as_deref(),
        )
    }

    /// Number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Set MRoPE on all full-attention layers
    pub fn set_mrope(&mut self, mrope_section: Vec<i32>, rope_theta: f32, rope_dims: usize) {
        for layer in &mut self.layers {
            if let Qwen35AttentionVariant::FullAttention(ref mut attn) = layer.attention {
                attn.mrope = Some(super::qwen3_vl::InterleavedMRoPE::new(
                    rope_dims, // dim = rope_dims (MRoPE sections sum to dim/2)
                    rope_theta,
                    mrope_section.clone(),
                ));
            }
        }
    }
}

// Weight Sanitization.
pub fn sanitize_weights(mut weights: WeightMap, config: &Qwen35Config) -> WeightMap {
    // 1. Detect sanitization needs
    let has_mtp = weights.keys().any(|k| k.contains("mtp."));
    let has_unsanitized_conv1d = weights.iter().any(|(k, v)| {
        k.contains("conv1d.weight") && {
            let shape = mlxcel_core::array_shape(v);
            shape.last() != Some(&1)
        }
    });
    let should_shift_norms = has_mtp || has_unsanitized_conv1d;

    // 2. Filter MTP weights
    weights.retain(|k, _| !k.contains("mtp."));

    // 3. Remove lm_head if tied
    if config.tie_word_embeddings {
        weights.remove("lm_head.weight");
    }

    // 4. Conv1d weight transpose and 5. Norm weight shift
    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];

    let keys: Vec<String> = weights.keys().cloned().collect();
    for k in &keys {
        // Conv1d weight: moveaxis(2, 1) when shape[-1] != 1
        if k.contains("conv1d.weight") {
            let v = weights.get(k.as_str()).unwrap();
            let shape = mlxcel_core::array_shape(v);
            if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                let transposed = mlxcel_core::swap_axes(v, -1, -2);
                weights.insert(k.clone(), transposed);
            }
        }

        // Norm weight shift (+1.0) when should_shift_norms
        if should_shift_norms && norm_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            let v = weights.get(k.as_str()).unwrap();
            let ndim = mlxcel_core::array_shape(v).len();
            if ndim == 1 {
                let one = mlxcel_core::full_f32(&[1], 1.0, dtype::FLOAT32);
                let shifted = mlxcel_core::add(v, &one);
                weights.insert(k.clone(), shifted);
            }
        }
    }

    // 6. MoE expert stacking (same as qwen3_next)
    for l in 0..config.num_hidden_layers {
        if !config.is_moe_layer(l) {
            continue;
        }

        let base = format!("model.layers.{}.mlp.switch_mlp", l);
        for proj in ["w1", "w2", "w3"] {
            let mut expert_weights: Vec<UniquePtr<MlxArray>> = Vec::new();
            let mut expert_scales: Vec<UniquePtr<MlxArray>> = Vec::new();
            let mut expert_biases: Vec<UniquePtr<MlxArray>> = Vec::new();

            let mut e = 0;
            while let Some(w) = weights.remove(&format!(
                "model.layers.{}.mlp.experts.{}.{}.weight",
                l, e, proj
            )) {
                expert_weights.push(w);
                if let Some(s) = weights.remove(&format!(
                    "model.layers.{}.mlp.experts.{}.{}.scales",
                    l, e, proj
                )) {
                    expert_scales.push(s);
                }
                if let Some(b) = weights.remove(&format!(
                    "model.layers.{}.mlp.experts.{}.{}.biases",
                    l, e, proj
                )) {
                    expert_biases.push(b);
                }
                e += 1;
            }

            if !expert_weights.is_empty() {
                let stacked = stack_arrays(&expert_weights, 0);
                weights.insert(format!("{}.{}.weight", base, proj), stacked);

                if !expert_scales.is_empty() {
                    let stacked = stack_arrays(&expert_scales, 0);
                    weights.insert(format!("{}.{}.scales", base, proj), stacked);
                }

                if !expert_biases.is_empty() {
                    let stacked = stack_arrays(&expert_biases, 0);
                    weights.insert(format!("{}.{}.biases", base, proj), stacked);
                }
            }
        }
    }

    // 7. MoE gate_up_proj split (for qwen3_5_moe format)
    for l in 0..config.num_hidden_layers {
        let gate_up_key = format!("model.layers.{}.mlp.experts.gate_up_proj", l);
        if let Some(gate_up) = weights.remove(&gate_up_key) {
            let shape = mlxcel_core::array_shape(&gate_up);
            // shape: [num_experts, gate_up_size, hidden] or similar
            // mid = shape[-2] // 2
            let mid = shape[shape.len() - 2] / 2;
            let ndims = shape.len();

            // gate_proj = gate_up[..., :mid, :]
            let mut starts = vec![0i32; ndims];
            let mut stops: Vec<i32> = shape.clone();
            stops[ndims - 2] = mid;
            let gate_proj = mlxcel_core::slice(&gate_up, &starts, &stops);

            // up_proj = gate_up[..., mid:, :]
            starts[ndims - 2] = mid;
            stops[ndims - 2] = shape[ndims - 2];
            let up_proj = mlxcel_core::slice(&gate_up, &starts, &stops);

            let base = format!("model.layers.{}.mlp.switch_mlp", l);
            weights.insert(format!("{}.gate_proj.weight", base), gate_proj);
            weights.insert(format!("{}.up_proj.weight", base), up_proj);

            // Move down_proj if present
            let down_key = format!("model.layers.{}.mlp.experts.down_proj", l);
            if let Some(down) = weights.remove(&down_key) {
                weights.insert(format!("{}.down_proj.weight", base), down);
            }
        }
    }

    // 8. Rename switch_mlp.{gate_proj,up_proj,down_proj} → switch_mlp.{w1,w3,w2}
    // Pre-quantized MoE models use gate_proj/up_proj/down_proj naming,
    // but SparseMoeBlock expects w1/w2/w3 naming.
    let rename_map = [
        ("switch_mlp.gate_proj.", "switch_mlp.w1."),
        ("switch_mlp.up_proj.", "switch_mlp.w3."),
        ("switch_mlp.down_proj.", "switch_mlp.w2."),
    ];
    let keys_to_rename: Vec<String> = weights
        .keys()
        .filter(|k| rename_map.iter().any(|(from, _)| k.contains(from)))
        .cloned()
        .collect();
    for key in keys_to_rename {
        for (from, to) in &rename_map {
            if key.contains(from) {
                let new_key = key.replace(from, to);
                if let Some(v) = weights.remove(&key) {
                    weights.insert(new_key, v);
                }
                break;
            }
        }
    }

    weights
}

/// Sanitize weights for MoE wrapper variant (qwen3_5_moe)
/// Handles language_model prefix stripping and gate_up_proj splitting
pub fn sanitize_moe_weights(weights: WeightMap, config: &Qwen35Config) -> WeightMap {
    let mut sanitized = WeightMap::new();

    for (key, value) in weights {
        // Skip vision tower weights
        if key.starts_with("vision_tower") || key.starts_with("model.visual") {
            continue;
        }

        let new_key = if key.starts_with("model.language_model") {
            key.replace("model.language_model", "language_model.model")
        } else if key.starts_with("language_model.") {
            key.clone()
        } else {
            format!("language_model.{}", key)
        };

        sanitized.insert(new_key, value);
    }

    // Handle gate_up_proj split for MoE
    let keys: Vec<String> = sanitized.keys().cloned().collect();
    for key in &keys {
        if key.contains("experts.gate_up_proj") && sanitized.contains_key(key.as_str()) {
            let gate_up = sanitized.remove(key).unwrap();
            let shape = mlxcel_core::array_shape(&gate_up);
            let ndims = shape.len();
            let mid = shape[ndims - 2] / 2;

            let mut starts = vec![0i32; ndims];
            let mut stops: Vec<i32> = shape.clone();
            stops[ndims - 2] = mid;
            let gate_proj = mlxcel_core::slice(&gate_up, &starts, &stops);

            starts[ndims - 2] = mid;
            stops[ndims - 2] = shape[ndims - 2];
            let up_proj = mlxcel_core::slice(&gate_up, &starts, &stops);

            let base = key.replace("experts.gate_up_proj", "switch_mlp");
            sanitized.insert(format!("{}.gate_proj.weight", base), gate_proj);
            sanitized.insert(format!("{}.up_proj.weight", base), up_proj);

            // Move down_proj
            let down_key = key.replace("experts.gate_up_proj", "experts.down_proj");
            if let Some(down) = sanitized.remove(&down_key) {
                sanitized.insert(format!("{}.down_proj.weight", base), down);
            }
        }
    }

    // Strip language_model. prefix for internal model loading
    let mut final_weights = WeightMap::new();
    for (key, value) in sanitized {
        let stripped = if let Some(rest) = key.strip_prefix("language_model.") {
            rest.to_string()
        } else {
            key
        };
        final_weights.insert(stripped, value);
    }

    // Apply standard sanitization
    sanitize_weights(final_weights, config)
}

// LanguageModel trait implementation.
impl LanguageModel for Qwen35Model {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Qwen3.5 uses mixed cache types (KVCache + GatedDeltaCache)
        // We use internal RefCell caches to maintain state through shared reference
        let mut internal = self.internal_caches.borrow_mut();
        // Use stored position_ids if available (set by VLM during prefill)
        let pos_ids = self.position_ids.borrow();
        let pos_ref = pos_ids.as_deref();
        self.forward_internal(input, None, &mut internal, pos_ref)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut internal = self.internal_caches.borrow_mut();
        let pos_ids = self.position_ids.borrow();
        let pos_ref = pos_ids.as_deref();
        self.forward_internal(input_ids, input_embeddings, &mut internal, pos_ref)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Reset internal caches
        *self.internal_caches.borrow_mut() = self.make_internal_caches();
        // Return dummy KV caches for trait compatibility
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![248046, 248044] // Qwen 3.5 EOS tokens
    }
}
