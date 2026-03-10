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

//! Gemma 3n (Nano) model implementation using mlxcel-core
//!
//! Key features:
//! - AltUp (Alternating Updates): Multiple parallel inputs with predict/correct mechanism
//! - LAUREL (Learned Augmented Residual Layer): Low-rank augmentation
//! - Per-layer inputs: Layer-specific input embeddings
//! - KV sharing: Some layers share KV cache
//! - Sliding window + Full attention mix
//! - gelu_topk activation with sparsity pattern
//! - Logit softcapping

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, Linear, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_causal_mask_with_window};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: serde_json::Value, // Can be int or list
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    pub num_kv_shared_layers: usize,
    pub vocab_size_per_layer_input: usize,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub rope_local_base_freq: f32,
    pub rope_theta: f32,
    pub final_logit_softcapping: Option<f32>,
    pub layer_types: Vec<String>,
    pub activation_sparsity_pattern: Option<Vec<f32>>,
    pub hidden_size_per_layer_input: usize,
    pub altup_num_inputs: usize,
    pub altup_coef_clip: Option<f32>,
    pub altup_correct_scale: bool,
    pub altup_active_idx: usize,
    pub laurel_rank: usize,

    #[serde(default)]
    pub quantization: Option<QuantizationArgs>,
}

impl TextConfig {
    pub fn get_intermediate_size(&self, layer_idx: usize) -> usize {
        match &self.intermediate_size {
            serde_json::Value::Number(n) => n.as_u64().unwrap() as usize,
            serde_json::Value::Array(arr) => arr[layer_idx].as_u64().unwrap() as usize,
            _ => panic!("Invalid intermediate_size"),
        }
    }

    pub fn get_activation_sparsity(&self, layer_idx: usize) -> f32 {
        self.activation_sparsity_pattern
            .as_ref()
            .map(|p| p[layer_idx])
            .unwrap_or(0.0)
    }

    pub fn first_kv_shared_layer_idx(&self) -> usize {
        self.num_hidden_layers - self.num_kv_shared_layers
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationArgs {
    pub group_size: usize,
    pub bits: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub text_config: serde_json::Value,
    #[serde(default)]
    pub quantization: Option<RootQuantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RootQuantization {
    pub group_size: usize,
    pub bits: usize,
}

impl ModelArgs {
    pub fn text_args(&self) -> TextConfig {
        let mut config: TextConfig =
            serde_json::from_value(self.text_config.clone()).expect("Failed to parse text_config");
        // Apply root quantization if text_config doesn't have it
        if config.quantization.is_none()
            && let Some(ref q) = self.quantization
        {
            config.quantization = Some(QuantizationArgs {
                group_size: q.group_size,
                bits: q.bits,
            });
        }
        config
    }
}

// Helper functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

/// Inverse error function approximation (Winitzki approximation)
fn erfinv(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return f32::INFINITY;
    }
    if x <= -1.0 {
        return f32::NEG_INFINITY;
    }

    let a = 0.147;
    let ln_one_minus_x2 = (1.0 - x * x).ln();
    let term_a = 2.0 / (std::f32::consts::PI * a) + ln_one_minus_x2 / 2.0;
    let term_b = ln_one_minus_x2 / a;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * ((term_a * term_a - term_b).sqrt() - term_a).sqrt()
}

// RMSNoScale - RMSNorm without learnable scale (uses unit weight).
pub struct RMSNoScale {
    pub eps: f32,
    pub unit_weight: UniquePtr<MlxArray>,
}

impl RMSNoScale {
    pub fn new(dim: i32, eps: f32) -> Self {
        // Create unit weight (all ones) for fast_rms_norm
        Self {
            eps,
            unit_weight: mlxcel_core::ones(&[dim], mlxcel_core::dtype::FLOAT32),
        }
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::fast_rms_norm(x, &self.unit_weight, self.eps)
    }
}

// LAUREL (Learned Augmented Residual Layer).
pub struct LaurelBlock {
    pub linear_left: UnifiedLinear,
    pub linear_right: UnifiedLinear,
    pub post_laurel_norm: RMSNorm,
}

impl LaurelBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let laurel_x = self.linear_left.forward(x);
        let laurel_x = self.linear_right.forward(&laurel_x);
        let normed = self.post_laurel_norm.forward(&laurel_x);
        mlxcel_core::add(x, &normed)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        let linear_left = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_left", prefix),
            group_size,
            bits,
        )?;
        let linear_right = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_right", prefix),
            group_size,
            bits,
        )?;

        let norm_weight = get_weight_copy(weights, &format!("{}.post_laurel_norm.weight", prefix))?;
        let post_laurel_norm = RMSNorm::new(norm_weight, config.rms_norm_eps);

        Ok(Self {
            linear_left,
            linear_right,
            post_laurel_norm,
        })
    }
}

// AltUp (Alternating Updates).
pub struct AltUp {
    pub correct_output_scale: UniquePtr<MlxArray>,
    // correction_coefs and prediction_coefs are NOT quantized (small 4x4 and 16x4 layers)
    pub correction_coefs: Linear,
    pub prediction_coefs: Linear,
    // modality_router IS quantized
    pub modality_router: UnifiedLinear,
    pub router_norm: RMSNorm,
    pub altup_num_inputs: usize,
    pub altup_active_idx: usize,
    pub hidden_size: usize,
}

impl AltUp {
    fn compute_router_modalities(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.router_norm.forward(x);
        let scale_val = (self.hidden_size as f32).powf(-1.0);
        let scale = mlxcel_core::full_f32(&[1], scale_val, mlxcel_core::dtype::FLOAT32);
        let scaled = mlxcel_core::multiply(&normed, &scale);
        let routed = self.modality_router.forward(&scaled);
        mlxcel_core::tanh(&routed)
    }

    /// Predict: expand inputs through altup_num_inputs parallel paths
    pub fn predict(&self, x: &[UniquePtr<MlxArray>]) -> Vec<UniquePtr<MlxArray>> {
        // x is [altup_num_inputs] arrays, each [B, L, hidden_size]
        // Get active input for routing
        let active = &x[self.altup_active_idx];
        let modalities = self.compute_router_modalities(active);

        // Get prediction coefficients
        let coefs = self.prediction_coefs.forward(&modalities);
        // coefs shape: [B, L, altup_num_inputs * altup_num_inputs]

        let n = self.altup_num_inputs as i32;
        let shape = mlxcel_core::array_shape(&modalities);
        let b = shape[0];
        let l = shape[1];

        // Reshape to [B, L, n, n] then transpose to [B, L, n, n] with axes swapped
        let all_coefs = mlxcel_core::reshape(&coefs, &[b, l, n, n]);
        let all_coefs = mlxcel_core::transpose_axes(&all_coefs, &[0, 1, 3, 2]);

        // Convert x to float32 for computation
        let x_f32: Vec<_> = x
            .iter()
            .map(|arr| mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32))
            .collect();

        // Stack x to [B, L, hidden, altup]
        let x_stacked = stack_arrays(&x_f32, 0);
        // x_stacked shape: [altup, B, L, hidden]
        let x_permuted = mlxcel_core::transpose_axes(&x_stacked, &[1, 2, 3, 0]);
        // x_permuted shape: [B, L, hidden, altup]

        // Matrix multiply: [B, L, hidden, altup] @ [B, L, altup, altup] = [B, L, hidden, altup]
        let predictions = mlxcel_core::matmul(&x_permuted, &all_coefs);
        // Transpose back to [altup, B, L, hidden]
        let predictions = mlxcel_core::transpose_axes(&predictions, &[3, 0, 1, 2]);

        // Add residual
        let predictions = mlxcel_core::add(&predictions, &x_stacked);

        // Split back to individual arrays
        let mut result = Vec::with_capacity(self.altup_num_inputs);
        for i in 0..self.altup_num_inputs {
            let start = vec![i as i32, 0, 0, 0];
            let hidden = mlxcel_core::array_shape(&x_f32[0])[2];
            let stop = vec![(i + 1) as i32, b, l, hidden];
            let sliced = mlxcel_core::slice(&predictions, &start, &stop);
            let squeezed = mlxcel_core::squeeze_axis(&sliced, 0);
            // Cast back to original dtype if needed
            result.push(squeezed);
        }

        result
    }

    /// Correct: apply correction to predictions based on activated output
    pub fn correct(
        &self,
        predictions: &[UniquePtr<MlxArray>],
        activated: &MlxArray,
    ) -> Vec<UniquePtr<MlxArray>> {
        let modalities = self.compute_router_modalities(activated);

        // correction_coefs output shape: [B, L, altup_num_inputs]
        let all_coefs = self.correction_coefs.forward(&modalities);
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
        let all_coefs = mlxcel_core::add(&all_coefs, &one);

        // Get active prediction
        let active_x = &predictions[self.altup_active_idx];
        // innovation = activated - active_prediction
        let innovation = mlxcel_core::subtract(activated, active_x);

        let shape = mlxcel_core::array_shape(&all_coefs);
        let b = shape[0];
        let l = shape[1];

        // Move axis: [B, L, altup] -> [altup, B, L]
        let all_coefs = mlxcel_core::transpose_axes(&all_coefs, &[2, 0, 1]);

        // innovation[None] shape: [1, B, L, hidden]
        let hidden = mlxcel_core::array_shape(&innovation)[2];
        let innovation_expanded = mlxcel_core::reshape(&innovation, &[1, b, l, hidden]);

        // all_coefs[..., None] shape: [altup, B, L, 1]
        let altup = self.altup_num_inputs as i32;
        let coefs_expanded = mlxcel_core::reshape(&all_coefs, &[altup, b, l, 1]);

        // Element-wise multiply: [1, B, L, hidden] * [altup, B, L, 1] = [altup, B, L, hidden]
        let correction = mlxcel_core::multiply(&innovation_expanded, &coefs_expanded);

        // Stack predictions and add correction
        let preds_stacked = stack_arrays(predictions, 0);
        let corrected = mlxcel_core::add(&preds_stacked, &correction);

        // Cast back to original dtype
        let original_dtype = mlxcel_core::array_dtype(activated);
        let corrected = mlxcel_core::astype(&corrected, original_dtype);

        // Split back to individual arrays
        let mut result = Vec::with_capacity(self.altup_num_inputs);
        for i in 0..self.altup_num_inputs {
            let start = vec![i as i32, 0, 0, 0];
            let stop = vec![(i + 1) as i32, b, l, hidden];
            let sliced = mlxcel_core::slice(&corrected, &start, &stop);
            let squeezed = mlxcel_core::squeeze_axis(&sliced, 0);
            result.push(squeezed);
        }

        result
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        // correction_coefs and prediction_coefs are NOT quantized (small 4x4 and 16x4 layers)
        let correction_coefs =
            Linear::from_weights(weights, &format!("{}.correction_coefs", prefix))?;
        let prediction_coefs =
            Linear::from_weights(weights, &format!("{}.prediction_coefs", prefix))?;

        // modality_router IS quantized
        let modality_router = UnifiedLinear::from_weights(
            weights,
            &format!("{}.modality_router", prefix),
            group_size,
            bits,
        )?;

        let norm_weight = get_weight_copy(weights, &format!("{}.router_norm.weight", prefix))?;
        let router_norm = RMSNorm::new(norm_weight, config.rms_norm_eps);

        let correct_output_scale =
            get_weight_copy(weights, &format!("{}.correct_output_scale", prefix))?;

        Ok(Self {
            correct_output_scale,
            correction_coefs,
            prediction_coefs,
            modality_router,
            router_norm,
            altup_num_inputs: config.altup_num_inputs,
            altup_active_idx: config.altup_active_idx,
            hidden_size: config.hidden_size,
        })
    }
}

// Attention.
pub struct Gemma3nAttention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub v_norm: RMSNoScale,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub is_sliding: bool,
    pub is_kv_shared_layer: bool,
    pub rope_theta: f32,
    pub scale: f32,
}

impl Gemma3nAttention {
    pub fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Query projection and reshape
        let queries = self.q_proj.forward(x);
        let queries = mlxcel_core::reshape(&queries, &[b, l, self.num_heads, self.head_dim]);

        // Apply Q norm
        let queries = self.q_norm.forward(&queries);
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);

        let cache_offset = cache.offset;

        // Compute KV (or get from cache for KV-shared layers)
        let (keys, values) = if self.is_kv_shared_layer {
            // For KV-shared layers, return slice view of filled portion only.
            // cache.keys is a pre-allocated buffer; we must slice to cache.offset
            // to avoid shape mismatch with the attention mask.
            let k = cache.keys.as_ref().unwrap();
            let v = cache.values.as_ref().unwrap();
            let ks = mlxcel_core::array_shape(k);
            let vs = mlxcel_core::array_shape(v);
            (
                mlxcel_core::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], cache.offset, ks[3]]),
                mlxcel_core::slice(v, &[0, 0, 0, 0], &[vs[0], vs[1], cache.offset, vs[3]]),
            )
        } else {
            self.compute_kv(x, b, l, cache_offset, cache)
        };

        // Apply RoPE to queries
        let queries = mlxcel_core::fast_rope(
            &queries,
            self.head_dim,
            false,
            self.rope_theta,
            1.0,
            cache_offset,
        );

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            // Prefill: use causal masking
            mlxcel_core::fast_scaled_dot_product_attention_causal(
                &queries, &keys, &values, self.scale,
            )
        } else {
            // Single token or explicit mask
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::fast_scaled_dot_product_attention(
                    &queries, &keys, &values, self.scale, mask_ptr,
                )
            }
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.o_proj.forward(&attn_out)
    }

    fn compute_kv(
        &self,
        x: &MlxArray,
        b: i32,
        l: i32,
        offset: i32,
        cache: &mut KVCache,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let keys = self.k_proj.forward(x);
        let keys = mlxcel_core::reshape(&keys, &[b, l, self.num_kv_heads, self.head_dim]);
        let keys = self.k_norm.forward(&keys);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);

        let values = self.v_proj.forward(x);
        let values = mlxcel_core::reshape(&values, &[b, l, self.num_kv_heads, self.head_dim]);
        let values = self.v_norm.forward(&values);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        // Apply RoPE to keys
        let keys =
            mlxcel_core::fast_rope(&keys, self.head_dim, false, self.rope_theta, 1.0, offset);

        // Update KV cache
        cache.update_and_fetch(keys, values)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        is_kv_shared_layer: bool,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        let is_sliding = config.layer_types[layer_idx] == "sliding_attention";
        let rope_theta = if is_sliding {
            config.rope_local_base_freq
        } else {
            config.rope_theta
        };

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let q_norm = RMSNorm::new(q_norm_weight, config.rms_norm_eps);
        let k_norm = RMSNorm::new(k_norm_weight, config.rms_norm_eps);
        let v_norm = RMSNoScale::new(config.head_dim as i32, config.rms_norm_eps);

        let head_dim = config.head_dim as i32;
        let scale = 1.0; // Gemma3n uses scale=1.0, softcapping handles scaling

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim,
            is_sliding,
            is_kv_shared_layer,
            rope_theta,
            scale,
        })
    }
}

// MLP with gelu_topk activation.
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
    pub activation_sparsity: f32,
    pub std_multiplier: f32,
}

impl MLP {
    /// Apply gelu_topk activation with sparsity
    fn gelu_topk(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.activation_sparsity <= 0.0 {
            return mlxcel_core::gelu_approx(x);
        }

        // Compute mean and std along last axis
        let mean = mlxcel_core::mean_axis(x, -1, true);

        // Manual std calculation: sqrt(mean((x - mean)^2))
        let diff = mlxcel_core::subtract(x, &mean);
        let diff_sq = mlxcel_core::square(&diff);
        let var = mlxcel_core::mean_axis(&diff_sq, -1, true);
        let std = mlxcel_core::sqrt(&var);

        // cutoff = mean + std * std_multiplier
        let std_mult =
            mlxcel_core::full_f32(&[1], self.std_multiplier, mlxcel_core::dtype::FLOAT32);
        let std_scaled = mlxcel_core::multiply(&std, &std_mult);
        let cutoff = mlxcel_core::add(&mean, &std_scaled);

        // shifted = x - cutoff
        let shifted = mlxcel_core::subtract(x, &cutoff);

        // zeroed = max(shifted, 0)
        let zero = mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);
        let zeroed = mlxcel_core::maximum(&shifted, &zero);

        // Apply gelu
        mlxcel_core::gelu_approx(&zeroed)
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let activated = self.gelu_topk(&gate);
        let up = self.up_proj.forward(x);
        let prod = mlxcel_core::multiply(&activated, &up);
        self.down_proj.forward(&prod)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

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

        let activation_sparsity = config.get_activation_sparsity(layer_idx);
        let std_multiplier = if activation_sparsity > 0.0 {
            std::f32::consts::SQRT_2 * erfinv(2.0 * activation_sparsity - 1.0)
        } else {
            0.0
        };

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            activation_sparsity,
            std_multiplier,
        })
    }
}

// Decoder Layer.
pub struct DecoderLayer {
    pub self_attn: Gemma3nAttention,
    pub mlp: MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub pre_feedforward_layernorm: RMSNorm,
    pub post_feedforward_layernorm: RMSNorm,
    pub altup: AltUp,
    pub laurel: LaurelBlock,
    pub per_layer_input_gate: UnifiedLinear,
    pub per_layer_projection: UnifiedLinear,
    pub post_per_layer_input_norm: RMSNorm,
    pub altup_active_idx: usize,
    pub altup_correct_scale: bool,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        x: &[UniquePtr<MlxArray>],
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        per_layer_input: &MlxArray,
    ) -> Vec<UniquePtr<MlxArray>> {
        // AltUp predict
        let predictions = self.altup.predict(x);
        let active_prediction = &predictions[self.altup_active_idx];

        // Input layernorm
        let active_normed = self.input_layernorm.forward(active_prediction);

        // LAUREL
        let laurel_output = self.laurel.forward(&active_normed);

        // Self attention
        let attn = self.self_attn.forward(&active_normed, mask, cache);
        let attn = self.post_attention_layernorm.forward(&attn);

        // Residual + LAUREL
        let attn_gated = mlxcel_core::add(active_prediction, &attn);
        let sqrt_half = mlxcel_core::full_f32(
            &[1],
            std::f32::consts::FRAC_1_SQRT_2,
            mlxcel_core::dtype::FLOAT32,
        );

        let sum = mlxcel_core::add(&attn_gated, &laurel_output);
        let attn_laurel = mlxcel_core::multiply(&sum, &sqrt_half);

        // FFN
        let attn_norm = self.pre_feedforward_layernorm.forward(&attn_laurel);
        let ffw = self.mlp.forward(&attn_norm);
        let ffw_norm = self.post_feedforward_layernorm.forward(&ffw);
        let ffw_gated = mlxcel_core::add(&attn_laurel, &ffw_norm);

        // AltUp correct
        let corrected = self.altup.correct(&predictions, &ffw_gated);

        // Per-layer input processing
        let first = &corrected[self.altup_active_idx];
        let first = if self.altup_correct_scale {
            mlxcel_core::multiply(first, &self.altup.correct_output_scale)
        } else {
            mlxcel_core::copy(first)
        };

        let first = self.per_layer_input_gate.forward(&first);
        let first = mlxcel_core::gelu_approx(&first);
        let first = mlxcel_core::multiply(&first, per_layer_input);
        let first = self.per_layer_projection.forward(&first);
        let first_prediction = self.post_per_layer_input_norm.forward(&first);

        // Add first_prediction to corrected[1:]
        let mut result = Vec::with_capacity(corrected.len());
        result.push(mlxcel_core::copy(&corrected[0]));
        for item in corrected.iter().skip(1) {
            result.push(mlxcel_core::add(item, &first_prediction));
        }

        result
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        is_kv_shared_layer: bool,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        let self_attn = Gemma3nAttention::from_weights(
            weights,
            config,
            layer_idx,
            is_kv_shared_layer,
            &format!("{}.self_attn", prefix),
        )?;
        let mlp = MLP::from_weights(weights, config, layer_idx, &format!("{}.mlp", prefix))?;
        let altup = AltUp::from_weights(weights, config, &format!("{}.altup", prefix))?;
        let laurel = LaurelBlock::from_weights(weights, config, &format!("{}.laurel", prefix))?;

        // Norms
        let input_layernorm = RMSNorm::new(
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?,
            config.rms_norm_eps,
        );
        let post_attention_layernorm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.post_attention_layernorm.weight", prefix),
            )?,
            config.rms_norm_eps,
        );
        let pre_feedforward_layernorm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.pre_feedforward_layernorm.weight", prefix),
            )?,
            config.rms_norm_eps,
        );
        let post_feedforward_layernorm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.post_feedforward_layernorm.weight", prefix),
            )?,
            config.rms_norm_eps,
        );
        let post_per_layer_input_norm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.post_per_layer_input_norm.weight", prefix),
            )?,
            config.rms_norm_eps,
        );

        // Per-layer projections
        let per_layer_input_gate = UnifiedLinear::from_weights(
            weights,
            &format!("{}.per_layer_input_gate", prefix),
            group_size,
            bits,
        )?;
        let per_layer_projection = UnifiedLinear::from_weights(
            weights,
            &format!("{}.per_layer_projection", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            altup,
            laurel,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            altup_active_idx: config.altup_active_idx,
            altup_correct_scale: config.altup_correct_scale,
        })
    }
}

// Language Model.
pub struct Gemma3nLanguageModel {
    pub embed_tokens: UnifiedEmbedding,
    pub embed_tokens_per_layer: UnifiedEmbedding,
    pub per_layer_model_projection: UnifiedLinear,
    pub per_layer_projection_norm: RMSNorm,
    pub layers: Vec<DecoderLayer>,
    pub altup_projections: Vec<UnifiedLinear>,
    pub altup_unembed_projections: Vec<UnifiedLinear>,
    pub norm: RMSNorm,
    pub config: TextConfig,
    pub layer_idx_to_cache_idx: Vec<usize>,
    pub first_sliding_idx: usize,
    pub first_full_idx: usize,
}

impl Gemma3nLanguageModel {
    pub fn forward(&self, inputs: &MlxArray, caches: &mut [KVCache]) -> UniquePtr<MlxArray> {
        // Embed tokens
        let h = self.embed_tokens.forward(inputs);
        let scale = mlxcel_core::full_f32(
            &[1],
            (self.config.hidden_size as f32).sqrt(),
            mlxcel_core::dtype::FLOAT32,
        );
        let h = mlxcel_core::multiply(&h, &scale);

        let shape = mlxcel_core::array_shape(&h);
        let b = shape[0];
        let l = shape[1];

        // Get per-layer inputs
        let per_layer_inputs = self.get_per_layer_inputs(inputs);
        let per_layer_inputs = self.project_per_layer_inputs(&h, &per_layer_inputs);

        // Create masks
        let global_offset = caches[self.first_full_idx].offset;
        let sliding_offset = caches[self.first_sliding_idx].offset;

        let global_mask = if l > 1 {
            Some(create_causal_mask(l, global_offset))
        } else {
            None
        };
        let sliding_mask = if l > 1 {
            Some(create_causal_mask_with_window(
                l,
                sliding_offset,
                Some(self.config.sliding_window as i32),
            ))
        } else {
            None
        };

        // Expand for AltUp
        let h0 = mlxcel_core::copy(&h);
        let target_magnitude = compute_magnitude(&h);

        // Create h_list: [h0, proj1(h0), proj2(h0), ...]
        let mut h_list = vec![mlxcel_core::copy(&h0)];
        for proj in &self.altup_projections {
            h_list.push(proj.forward(&h0));
        }

        // Normalize magnitudes of projected inputs
        normalize_magnitudes(&mut h_list, &target_magnitude);

        // Process layers
        for (i, layer) in self.layers.iter().enumerate() {
            let cache_idx = self.layer_idx_to_cache_idx[i];
            let mask = if self.config.layer_types[i] == "full_attention" {
                global_mask.as_ref()
            } else {
                sliding_mask.as_ref()
            };

            // Get per-layer input for this layer
            let per_layer_input = slice_layer_input(
                &per_layer_inputs,
                i as i32,
                b,
                l,
                self.config.hidden_size_per_layer_input as i32,
            );

            h_list = layer.forward(
                &h_list,
                mask.map(|m| m.as_ref().unwrap()),
                &mut caches[cache_idx],
                &per_layer_input,
            );
        }

        // Collapse AltUp dimension
        let h0_out = &h_list[0];
        let target_magnitude = compute_magnitude(h0_out);

        // Apply unembed projections
        for (i, proj) in self.altup_unembed_projections.iter().enumerate() {
            h_list[i + 1] = proj.forward(&h_list[i + 1]);
        }

        // Normalize magnitudes
        normalize_magnitudes_from_idx(&mut h_list, 1, &target_magnitude);

        // Mean across altup dimension
        let h = mean_arrays(&h_list);

        // Final norm
        let out = self.norm.forward(&h);

        // LM head (tied embeddings)
        let mut logits = self.embed_tokens.as_linear(&out);

        // Apply logit softcapping if configured
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = apply_softcap(&logits, cap);
        }

        logits
    }

    pub fn get_per_layer_inputs(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let vocab_limit = mlxcel_core::full_f32(
            &[1],
            self.config.vocab_size_per_layer_input as f32,
            mlxcel_core::dtype::INT32,
        );
        let vocab_limit = mlxcel_core::astype(&vocab_limit, mlxcel_core::dtype::INT32);
        let mask = mlxcel_core::less(input_ids, &vocab_limit);
        let zeros = mlxcel_core::zeros(
            &mlxcel_core::array_shape(input_ids),
            mlxcel_core::dtype::INT32,
        );
        let tokens = mlxcel_core::where_cond(&mask, input_ids, &zeros);

        let embedded = self.embed_tokens_per_layer.forward(&tokens);
        let scale = mlxcel_core::full_f32(
            &[1],
            (self.config.hidden_size_per_layer_input as f32).sqrt(),
            mlxcel_core::dtype::FLOAT32,
        );
        let result = mlxcel_core::multiply(&embedded, &scale);

        // Reshape to [B, L, num_hidden_layers, hidden_size_per_layer_input]
        let shape = mlxcel_core::array_shape(input_ids);
        let b = shape[0];
        let l = shape[1];
        mlxcel_core::reshape(
            &result,
            &[
                b,
                l,
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        )
    }

    pub fn project_per_layer_inputs(
        &self,
        inputs_embeds: &MlxArray,
        per_layer_inputs: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let proj = self.per_layer_model_projection.forward(inputs_embeds);
        let scale = mlxcel_core::full_f32(
            &[1],
            (self.config.hidden_size as f32).powf(-0.5),
            mlxcel_core::dtype::FLOAT32,
        );
        let proj = mlxcel_core::multiply(&proj, &scale);

        let shape = mlxcel_core::array_shape(inputs_embeds);
        let b = shape[0];
        let l = shape[1];
        let proj = mlxcel_core::reshape(
            &proj,
            &[
                b,
                l,
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        );

        let proj_normed = self.per_layer_projection_norm.forward(&proj);

        let sqrt_half = mlxcel_core::full_f32(
            &[1],
            std::f32::consts::FRAC_1_SQRT_2,
            mlxcel_core::dtype::FLOAT32,
        );
        let sum = mlxcel_core::add(&proj_normed, per_layer_inputs);
        mlxcel_core::multiply(&sum, &sqrt_half)
    }

    /// Get embedded token representations (for VLM use)
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        let scale = mlxcel_core::full_f32(
            &[1],
            (self.config.hidden_size as f32).sqrt(),
            mlxcel_core::dtype::FLOAT32,
        );
        mlxcel_core::multiply(&h, &scale)
    }

    /// Forward pass starting from pre-computed embeddings + per_layer_inputs (for VLM)
    pub fn forward_with_inputs(
        &self,
        inputs_embeds: &MlxArray,
        per_layer_inputs: &MlxArray,
        caches: &mut [KVCache],
    ) -> UniquePtr<MlxArray> {
        let h = inputs_embeds;

        let shape = mlxcel_core::array_shape(h);
        let _b = shape[0];
        let l = shape[1];

        // Create masks
        let global_offset = caches[self.first_full_idx].offset;
        let sliding_offset = caches[self.first_sliding_idx].offset;

        let global_mask = if l > 1 {
            Some(create_causal_mask(l, global_offset))
        } else {
            None
        };
        let sliding_mask = if l > 1 {
            Some(create_causal_mask_with_window(
                l,
                sliding_offset,
                Some(self.config.sliding_window as i32),
            ))
        } else {
            None
        };

        // Expand for AltUp
        let h0 = mlxcel_core::copy(h);
        let target_magnitude = compute_magnitude(h);

        let mut h_list = vec![mlxcel_core::copy(&h0)];
        for proj in &self.altup_projections {
            h_list.push(proj.forward(&h0));
        }
        normalize_magnitudes(&mut h_list, &target_magnitude);

        // Process layers
        for (i, layer) in self.layers.iter().enumerate() {
            let cache_idx = self.layer_idx_to_cache_idx[i];
            let mask = if self.config.layer_types[i] == "full_attention" {
                global_mask.as_ref()
            } else {
                sliding_mask.as_ref()
            };

            let per_layer_input = slice_layer_input(
                per_layer_inputs,
                i as i32,
                shape[0],
                l,
                self.config.hidden_size_per_layer_input as i32,
            );

            h_list = layer.forward(
                &h_list,
                mask.map(|m| m.as_ref().unwrap()),
                &mut caches[cache_idx],
                &per_layer_input,
            );
        }

        // Collapse AltUp dimension
        let h0_out = &h_list[0];
        let target_magnitude = compute_magnitude(h0_out);

        for (i, proj) in self.altup_unembed_projections.iter().enumerate() {
            h_list[i + 1] = proj.forward(&h_list[i + 1]);
        }
        normalize_magnitudes_from_idx(&mut h_list, 1, &target_magnitude);

        let h = mean_arrays(&h_list);
        let out = self.norm.forward(&h);
        let mut logits = self.embed_tokens.as_linear(&out);

        if let Some(cap) = self.config.final_logit_softcapping {
            logits = apply_softcap(&logits, cap);
        }

        logits
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        let first_kv_shared = self.config.first_kv_shared_layer_idx();
        (0..first_kv_shared).map(|_| KVCache::new()).collect()
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        let first_kv_shared = config.first_kv_shared_layer_idx();

        // Embeddings
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            &format!("{}.embed_tokens", prefix),
            group_size,
            bits,
        )?;
        let embed_tokens_per_layer = UnifiedEmbedding::from_weights(
            weights,
            &format!("{}.embed_tokens_per_layer", prefix),
            group_size,
            bits,
        )?;

        // Per-layer projection
        let per_layer_model_projection = UnifiedLinear::from_weights(
            weights,
            &format!("{}.per_layer_model_projection", prefix),
            group_size,
            bits,
        )?;
        let per_layer_projection_norm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.per_layer_projection_norm.weight", prefix),
            )?,
            config.rms_norm_eps,
        );

        // Layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let is_kv_shared = i >= first_kv_shared;
            let layer = DecoderLayer::from_weights(
                weights,
                config,
                i,
                is_kv_shared,
                &format!("{}.layers.{}", prefix, i),
            )?;
            layers.push(layer);
        }

        // AltUp projections
        let mut altup_projections = Vec::new();
        let mut altup_unembed_projections = Vec::new();
        for i in 0..(config.altup_num_inputs - 1) {
            altup_projections.push(UnifiedLinear::from_weights(
                weights,
                &format!("{}.altup_projections.{}", prefix, i),
                group_size,
                bits,
            )?);
            altup_unembed_projections.push(UnifiedLinear::from_weights(
                weights,
                &format!("{}.altup_unembed_projections.{}", prefix, i),
                group_size,
                bits,
            )?);
        }

        // Final norm
        let norm = RMSNorm::new(
            get_weight_copy(weights, &format!("{}.norm.weight", prefix))?,
            config.rms_norm_eps,
        );

        // Build layer to cache mapping
        let mut layer_idx_to_cache_idx = Vec::with_capacity(config.num_hidden_layers);
        let concrete_layers: Vec<_> = config.layer_types[..first_kv_shared].to_vec();
        let shared_full_idx = concrete_layers
            .iter()
            .rposition(|t| t == "full_attention")
            .unwrap_or(0);
        let shared_sliding_idx = concrete_layers
            .iter()
            .rposition(|t| t == "sliding_attention")
            .unwrap_or(0);

        for (i, layer_type) in config.layer_types.iter().enumerate() {
            if i < first_kv_shared {
                layer_idx_to_cache_idx.push(i);
            } else if layer_type == "full_attention" {
                layer_idx_to_cache_idx.push(shared_full_idx);
            } else {
                layer_idx_to_cache_idx.push(shared_sliding_idx);
            }
        }

        // Find first indices
        let first_sliding_idx = config
            .layer_types
            .iter()
            .position(|t| t == "sliding_attention")
            .unwrap_or(0);
        let first_full_idx = config
            .layer_types
            .iter()
            .position(|t| t == "full_attention")
            .unwrap_or(0);

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            altup_projections,
            altup_unembed_projections,
            norm,
            config: config.clone(),
            layer_idx_to_cache_idx,
            first_sliding_idx,
            first_full_idx,
        })
    }
}

// Model.
pub struct Gemma3nModel {
    pub language_model: Gemma3nLanguageModel,
    pub config: TextConfig,
}

impl Gemma3nModel {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        let config = args.text_args();

        // Load weights
        let weights = crate::models::load_and_sanitize_weights(model_dir)?;

        // Create model
        let language_model =
            Gemma3nLanguageModel::from_weights(&weights, &config, "language_model.model")?;

        Ok(Self {
            language_model,
            config,
        })
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        self.language_model.make_caches()
    }

    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

impl LanguageModel for Gemma3nModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.language_model.forward(input_ids, caches)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.language_model.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Gemma3nModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.language_model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Gemma 3n EOS tokens: <eos> (1), <end_of_turn> (106)
        vec![1, 106]
    }
}

// Multimodal Embedder (for VLM).
/// Embeds soft tokens (vision features) into language model space.
///
/// soft_embedding_norm → embedding_projection → post_projection_norm
pub struct Gemma3nMultimodalEmbedder {
    pub soft_embedding_norm: RMSNorm,        // with scale
    pub embedding_projection: UnifiedLinear, // vision_hidden → text_hidden
    pub post_projection_norm: RMSNoScale,    // without scale
}

impl Gemma3nMultimodalEmbedder {
    /// Forward pass for soft tokens (vision features)
    pub fn forward_soft(&self, inputs_embeds: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.soft_embedding_norm.forward(inputs_embeds);
        let projected = self.embedding_projection.forward(&normed);
        self.post_projection_norm.forward(&projected)
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str, // "embed_vision"
        vision_hidden_size: usize,
        text_hidden_size: usize,
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let soft_norm_weight =
            get_weight_copy(weights, &format!("{}.soft_embedding_norm.weight", prefix))?;
        let soft_embedding_norm = RMSNorm::new(soft_norm_weight, eps);

        let embedding_projection = UnifiedLinear::from_weights(
            weights,
            &format!("{}.embedding_projection", prefix),
            group_size,
            bits,
        )?;

        let post_projection_norm = RMSNoScale::new(text_hidden_size as i32, eps);

        let _ = vision_hidden_size; // used for documentation

        Ok(Self {
            soft_embedding_norm,
            embedding_projection,
            post_projection_norm,
        })
    }
}

// Helper functions.
/// Stack arrays along a new axis
fn stack_arrays(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    let ptrs: Vec<*const MlxArray> = arrays
        .iter()
        .map(|a| a.as_ref().unwrap() as *const _)
        .collect();
    mlxcel_core::stack(&ptrs, axis)
}

/// Compute magnitude (RMS) of array along last axis
fn compute_magnitude(x: &MlxArray) -> UniquePtr<MlxArray> {
    let sq = mlxcel_core::square(x);
    let mean = mlxcel_core::mean_axis(&sq, -1, true);
    mlxcel_core::sqrt(&mean)
}

/// Normalize magnitudes of arrays starting from idx 1
fn normalize_magnitudes(arrays: &mut [UniquePtr<MlxArray>], target_magnitude: &MlxArray) {
    let eps = mlxcel_core::full_f32(&[1], 1e-6, mlxcel_core::dtype::FLOAT32);
    for item in arrays.iter_mut().skip(1) {
        let mag = compute_magnitude(item);
        let mag_safe = mlxcel_core::maximum(&mag, &eps);
        let scale = mlxcel_core::divide(target_magnitude, &mag_safe);
        *item = mlxcel_core::multiply(item, &scale);
    }
}

/// Normalize magnitudes of arrays starting from specified index
fn normalize_magnitudes_from_idx(
    arrays: &mut [UniquePtr<MlxArray>],
    start_idx: usize,
    target_magnitude: &MlxArray,
) {
    let eps = mlxcel_core::full_f32(&[1], 1e-6, mlxcel_core::dtype::FLOAT32);
    for item in arrays.iter_mut().skip(start_idx) {
        let mag = compute_magnitude(item);
        let mag_safe = mlxcel_core::maximum(&mag, &eps);
        let scale = mlxcel_core::divide(target_magnitude, &mag_safe);
        *item = mlxcel_core::multiply(item, &scale);
    }
}

/// Mean of arrays
fn mean_arrays(arrays: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
    let stacked = stack_arrays(arrays, 0);
    mlxcel_core::mean_axis(&stacked, 0, false)
}

/// Apply softcap to logits: cap * tanh(logits / cap)
fn apply_softcap(logits: &MlxArray, cap: f32) -> UniquePtr<MlxArray> {
    let cap_arr = mlxcel_core::full_f32(&[1], cap, mlxcel_core::dtype::FLOAT32);
    let scaled = mlxcel_core::divide(logits, &cap_arr);
    let tanh_out = mlxcel_core::tanh(&scaled);
    mlxcel_core::multiply(&tanh_out, &cap_arr)
}

/// Slice per-layer input for a specific layer
fn slice_layer_input(
    per_layer_inputs: &MlxArray,
    layer_idx: i32,
    b: i32,
    l: i32,
    hidden_size: i32,
) -> UniquePtr<MlxArray> {
    let start = vec![0, 0, layer_idx, 0];
    let stop = vec![b, l, layer_idx + 1, hidden_size];
    let sliced = mlxcel_core::slice(per_layer_inputs, &start, &stop);
    mlxcel_core::squeeze_axis(&sliced, 2)
}
