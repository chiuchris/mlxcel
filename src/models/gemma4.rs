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

//! Gemma 4 text model implementation using mlxcel-core.
//!
//! Key features:
//! - Layer-type driven sliding/full attention
//! - Dual head dimensions (`head_dim` + `global_head_dim`)
//! - Partial RoPE on full-attention layers
//! - K-eq-V full-attention path for 26B/31B variants
//! - KV sharing for E-series models
//! - Per-layer input gating for E-series models
//! - Dense + MoE feed-forward paths
//! - Final logit softcapping

use crate::distributed::pipeline::LayerFilter;
use crate::distributed::pipeline::StageExecutionOutput;
use crate::distributed::pipeline::partial_loading::filter_weight_map;
use crate::models::switch_layers::{SwitchLinear, gather_sort};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{
    FusedQKVLinear, KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear,
    compiled_gelu_mlp_fp16,
};
use mlxcel_core::utils::{
    create_causal_mask, create_causal_mask_with_window, pipeline_hint, slice_axis,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QuantizationArgs {
    pub group_size: usize,
    pub bits: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    /// Upstream Gemma 4 (post-commit bb91e23 on mlx-vlm main) marks
    /// full-attention layers with `rope_type: "proportional"`, whose exponents
    /// are normalized by the FULL head dimension (`head_dim`) rather than the
    /// rotated-only slice. `"default"` means the usual
    /// `nn.RoPE(dims = head_dim * partial_rotary_factor)` path.
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_rope_type() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    #[serde(default)]
    pub vocab_size_per_layer_input: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub rope_traditional: bool,
    pub rope_parameters: HashMap<String, RopeParameters>,
    pub sliding_window: usize,
    #[serde(default)]
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub quantization: Option<QuantizationArgs>,
}

impl TextConfig {
    fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64)
    }

    fn bits(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4)
    }

    fn first_kv_shared_layer_idx(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers)
    }

    fn is_kv_shared_layer(&self, layer_idx: usize) -> bool {
        self.num_kv_shared_layers > 0 && layer_idx >= self.first_kv_shared_layer_idx()
    }

    fn layer_type(&self, layer_idx: usize) -> &str {
        self.layer_types[layer_idx].as_str()
    }

    fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.layer_type(layer_idx) == "sliding_attention"
    }

    fn head_dim_for_layer(&self, layer_idx: usize) -> i32 {
        if self.layer_type(layer_idx) == "full_attention" {
            self.global_head_dim.unwrap_or(self.head_dim) as i32
        } else {
            self.head_dim as i32
        }
    }

    fn num_kv_heads_for_layer(&self, layer_idx: usize) -> i32 {
        if self.attention_k_eq_v && !self.is_sliding_layer(layer_idx) {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads) as i32
        } else {
            self.num_key_value_heads as i32
        }
    }

    fn rope_params_for_layer(&self, layer_idx: usize) -> RopeParameters {
        let key = if self.is_sliding_layer(layer_idx) {
            "sliding_attention"
        } else {
            "full_attention"
        };
        self.rope_parameters
            .get(key)
            .cloned()
            .unwrap_or(RopeParameters {
                rope_theta: default_rope_theta(),
                partial_rotary_factor: default_partial_rotary_factor(),
                rope_type: default_rope_type(),
            })
    }

    fn mlp_intermediate_size(&self, layer_idx: usize) -> usize {
        let is_shared = self.is_kv_shared_layer(layer_idx);
        if self.use_double_wide_mlp && is_shared {
            self.intermediate_size * 2
        } else {
            self.intermediate_size
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RootQuantization {
    pub group_size: usize,
    pub bits: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub text_config: serde_json::Value,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<RootQuantization>,
}

impl ModelArgs {
    pub fn text_args(&self) -> TextConfig {
        let mut config: TextConfig =
            serde_json::from_value(self.text_config.clone()).expect("Failed to parse text_config");
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

    pub fn eos_token_ids(&self) -> Vec<i32> {
        parse_eos_ids(self.eos_token_id.as_ref())
    }
}

fn parse_eos_ids(value: Option<&serde_json::Value>) -> Vec<i32> {
    match value {
        Some(serde_json::Value::Number(n)) => {
            n.as_i64().map(|v| vec![v as i32]).unwrap_or_default()
        }
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_i64().map(|n| n as i32))
            .collect(),
        _ => Vec::new(),
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

pub struct RMSNormNoScale {
    eps: f32,
}

impl RMSNormNoScale {
    pub fn new(dim: i32, eps: f32) -> Self {
        let _ = dim;
        Self { eps }
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::fast_rms_norm_no_weight(x, self.eps)
    }
}

struct SwitchGeGLU {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGeGLU {
    fn forward(&self, x: &MlxArray, indices: &MlxArray, hidden_size: i32) -> UniquePtr<MlxArray> {
        // `MLXCEL_PROFILE_MOE_INNER=1` emits `[MOE] name=... ms=...` per
        // sub-sub-op so the `experts` band in the sub-op profiler can be
        // drilled one level further. Like the outer profilers this forces a
        // sync at every step and distorts absolute throughput; use it only
        // for relative distribution analysis.
        let profile_inner = std::env::var("MLXCEL_PROFILE_MOE_INNER").is_ok();
        let inner_tick = |name: &str, arr: &MlxArray, last: &mut std::time::Instant| {
            if !profile_inner {
                return;
            }
            mlxcel_core::eval(arr);
            let dt = last.elapsed();
            eprintln!("[MOE] name={} ms={:.4}", name, dt.as_secs_f64() * 1000.0);
            *last = std::time::Instant::now();
        };
        let mut last = std::time::Instant::now();

        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];
        let do_sort = n_tokens * top_k >= 64;

        // `Experts::forward` always passes a flattened `[tokens, hidden]`
        // input here. Python writes this as `expand_dims((-2, -3))`, which
        // gives `[tokens, 1, 1, hidden]`; a reshape is equivalent for this
        // rank-2 internal path and avoids an extra shape primitive in the
        // decode-hot SwitchGeGLU graph.
        let x_exp = mlxcel_core::reshape(x, &[n_tokens, 1, 1, hidden_size]);
        inner_tick("expand_dims", &x_exp, &mut last);

        if do_sort {
            let (sorted_x, sorted_idx, inv_order) = gather_sort(&x_exp, indices);
            inner_tick("gather_sort", &sorted_x, &mut last);
            let up = self.up_proj.forward(&sorted_x, &sorted_idx, true);
            inner_tick("up_proj", &up, &mut last);
            let gate = self.gate_proj.forward(&sorted_x, &sorted_idx, true);
            inner_tick("gate_proj", &gate, &mut last);
            let activated = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
            inner_tick("geglu", &activated, &mut last);
            let output = self.down_proj.forward(&activated, &sorted_idx, true);
            inner_tick("down_proj", &output, &mut last);
            let unsorted = scatter_unsort(&output, &inv_order, &indices_shape);
            inner_tick("scatter_unsort", &unsorted, &mut last);
            unsorted
        } else {
            // Keep the decode-hot path in the same op shape as mlx-lm by
            // default: separate gather_qmm(gate/up/down) with only the GeGLU
            // elementwise chain compiled. A wider compile window around
            // gather_qmm has regressed Gemma 4 26B/31B decode in profiling, so
            // it is retained as an explicit experiment only.
            let enable_compiled_switch =
                std::env::var_os("MLXCEL_ENABLE_COMPILED_SWITCH_QGEGLU").is_some();
            let output = if enable_compiled_switch
                && let (Some(gate_q), Some(up_q), Some(down_q)) = (
                    self.gate_proj.quantized_parts(),
                    self.up_proj.quantized_parts(),
                    self.down_proj.quantized_parts(),
                ) {
                let output = unsafe {
                    mlxcel_core::compiled_switch_qgeglu_forward(
                        &x_exp,
                        gate_q.weight,
                        gate_q.scales,
                        gate_q.biases.as_ref().unwrap() as *const _,
                        up_q.weight,
                        up_q.scales,
                        up_q.biases.as_ref().unwrap() as *const _,
                        down_q.weight,
                        down_q.scales,
                        down_q.biases.as_ref().unwrap() as *const _,
                        indices,
                        gate_q.group_size,
                        gate_q.bits,
                        gate_q.mode,
                    )
                };
                inner_tick("switch_geglu_fused", &output, &mut last);
                output
            } else {
                let up = self.up_proj.forward(&x_exp, indices, false);
                inner_tick("up_proj", &up, &mut last);
                let gate = self.gate_proj.forward(&x_exp, indices, false);
                inner_tick("gate_proj", &gate, &mut last);
                let activated = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
                inner_tick("geglu", &activated, &mut last);
                let output = self.down_proj.forward(&activated, indices, false);
                inner_tick("down_proj", &output, &mut last);
                output
            };
            let squeezed = mlxcel_core::squeeze_axis(&output, -2);
            inner_tick("squeeze", &squeezed, &mut last);
            squeezed
        }
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                group_size,
                bits,
            )?,
        })
    }
}

fn scatter_unsort(x: &MlxArray, inv_order: &MlxArray, orig_shape: &[i32]) -> UniquePtr<MlxArray> {
    let unsorted = mlxcel_core::take(x, inv_order, 0);
    let x_shape = mlxcel_core::array_shape(&unsorted);
    let n_tokens = orig_shape[0];
    let top_k = orig_shape[1];
    let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
    mlxcel_core::squeeze_axis(&reshaped, 2)
}

pub struct MLP {
    pub(crate) gate_proj: UnifiedLinear,
    pub(crate) up_proj: UnifiedLinear,
    pub(crate) down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if let (Some(gate_qw), Some(up_qw), Some(down_qw)) = (
            self.gate_proj.quantized_weight(),
            self.up_proj.quantized_weight(),
            self.down_proj.quantized_weight(),
        ) {
            return unsafe {
                mlxcel_core::compiled_gelu_approx_mlp_forward(
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

        if let Some(out) =
            compiled_gelu_mlp_fp16(x, &self.gate_proj, &self.up_proj, &self.down_proj)
        {
            return out;
        }

        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let hidden = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
        self.down_proj.forward(&hidden)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let _ = config.mlp_intermediate_size(layer_idx);
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                config.group_size(),
                config.bits(),
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                config.group_size(),
                config.bits(),
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                config.group_size(),
                config.bits(),
            )?,
        })
    }
}

pub struct Router {
    /// Precomputed `scale * hidden_size^-0.5` fed as the weight to
    /// `fast_rms_norm`, collapsing the old
    /// `rms_norm_no_weight → multiply_scalar → multiply` trio into
    /// one fused Metal kernel to match Python's
    /// `mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)`.
    pub(crate) scale_with_root: UniquePtr<MlxArray>,
    pub(crate) rms_eps: f32,
    pub(crate) proj: UnifiedLinear,
    pub(crate) per_expert_scale: UniquePtr<MlxArray>,
    pub(crate) top_k_experts: i32,
}

impl Router {
    pub fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let x = mlxcel_core::fast_rms_norm(x, &self.scale_with_root, self.rms_eps);
        let expert_scores = self.proj.forward(&x);

        // Pick top-k before softmax so softmax runs over k=8 values,
        // not the full num_experts=128. Use the same negative-kth
        // argpartition shape as mlx-lm to avoid an extra negation graph.
        let top_k_indices = mlxcel_core::argpartition(&expert_scores, -self.top_k_experts, -1);
        let top_k_indices = slice_axis(&top_k_indices, -1, -self.top_k_experts, -1);

        let top_k_weights = mlxcel_core::take_along_axis(&expert_scores, &top_k_indices, -1);
        let top_k_weights = mlxcel_core::softmax(&top_k_weights, -1);
        let expert_scale = mlxcel_core::take(&self.per_expert_scale, &top_k_indices, 0);
        let top_k_weights = mlxcel_core::multiply(&top_k_weights, &expert_scale);

        (top_k_indices, top_k_weights)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let raw_scale = get_weight_copy(weights, &format!("{}.scale", prefix))?;
        let root = (config.hidden_size as f32).powf(-0.5);
        let scale_with_root = mlxcel_core::multiply_scalar(&raw_scale, root);
        mlxcel_core::eval(&scale_with_root);

        Ok(Self {
            scale_with_root,
            rms_eps: config.rms_norm_eps,
            proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.proj", prefix),
                config.group_size(),
                config.bits(),
            )?,
            per_expert_scale: get_weight_copy(weights, &format!("{}.per_expert_scale", prefix))?,
            top_k_experts: config
                .top_k_experts
                .ok_or_else(|| "Missing top_k_experts for Gemma4 MoE router".to_string())?
                as i32,
        })
    }
}

pub struct Experts {
    switch_geglu: SwitchGeGLU,
}

impl Experts {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        top_k_indices: &MlxArray,
        top_k_weights: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let s = shape[1];
        let h = shape[2];
        let k = mlxcel_core::array_shape(top_k_indices)[2];

        let x_flat = mlxcel_core::reshape(x, &[b * s, h]);
        let indices_flat = mlxcel_core::reshape(top_k_indices, &[b * s, k]);
        let expert_out = self.switch_geglu.forward(&x_flat, &indices_flat, h);

        let weights = mlxcel_core::reshape(top_k_weights, &[b * s, k, 1]);
        let weighted = mlxcel_core::multiply(&expert_out, &weights);
        let reduced = mlxcel_core::sum_axis(&weighted, -2, false);
        mlxcel_core::reshape(&reduced, &[b, s, h])
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            switch_geglu: SwitchGeGLU::from_weights(
                weights,
                &format!("{}.switch_glu", prefix),
                config.group_size(),
                config.bits(),
            )?,
        })
    }
}

enum AttentionProjection {
    Fused(FusedQKVLinear),
    Separate {
        q_proj: UnifiedLinear,
        k_proj: UnifiedLinear,
        v_proj: Option<UnifiedLinear>,
    },
}

pub(crate) trait CacheInterface {
    fn offset(&self) -> i32;
    fn set_offset(&mut self, offset: i32);
    fn update_and_fetch(
        &mut self,
        keys: UniquePtr<MlxArray>,
        values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>);
}

impl CacheInterface for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn set_offset(&mut self, offset: i32) {
        self.offset = offset;
    }

    fn update_and_fetch(
        &mut self,
        keys: UniquePtr<MlxArray>,
        values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(keys, values)
    }
}

impl CacheInterface for RotatingKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn set_offset(&mut self, offset: i32) {
        self.offset = offset;
    }

    fn update_and_fetch(
        &mut self,
        keys: UniquePtr<MlxArray>,
        values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(keys, values)
    }
}

pub enum Cache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl Cache {
    pub(crate) fn offset(&self) -> i32 {
        match self {
            Self::Standard(cache) => cache.offset,
            Self::Rotating(cache) => cache.offset,
        }
    }

    pub(crate) fn as_interface(&mut self) -> &mut dyn CacheInterface {
        match self {
            Self::Standard(cache) => cache,
            Self::Rotating(cache) => cache,
        }
    }
}

pub struct Attention {
    projection: AttentionProjection,
    pub(crate) o_proj: UnifiedLinear,
    pub(crate) q_norm: RMSNorm,
    pub(crate) k_norm: RMSNorm,
    pub(crate) v_norm: RMSNormNoScale,
    pub(crate) n_heads: i32,
    pub(crate) n_kv_heads: i32,
    pub(crate) head_dim: i32,
    /// For `rope_type == "default"`, this is `head_dim * partial_rotary_factor`
    /// (the historical mlxcel / pre-bb91e23 mlx-vlm behavior).
    /// For `rope_type == "proportional"`, this is unused; proportional RoPE
    /// is driven by the precomputed full-head `proportional_rope_freqs` table
    /// below.
    pub(crate) rope_dims: i32,
    pub(crate) rope_theta: f32,
    /// If `Some`, the layer uses proportional RoPE (Gemma 4 full-attention
    /// layers). Holds the length-`head_dim / 2` frequency table consumed by
    /// `mlxcel_core::rope_proportional::apply_proportional_rope`; entries past
    /// the rotated prefix are `inf` so MLX's RoPE applies zero phase there.
    pub(crate) proportional_rope_freqs: Option<UniquePtr<MlxArray>>,
    /// Only meaningful when `proportional_rope_freqs.is_some()`. Matches the
    /// `partial_rotary_factor` passed in to
    /// `compute_proportional_rope_freqs`, so `apply_proportional_rope` can
    /// recompute `rotated_dims` without serializing it as a separate field.
    pub(crate) proportional_partial_rotary_factor: f32,
    pub(crate) scale: f32,
    pub(crate) is_kv_shared_layer: bool,
    pub(crate) kv_shared_layer_index: Option<usize>,
    pub(crate) store_full_length_kv: bool,
    pub(crate) use_k_eq_v: bool,
    pub(crate) window_size: i32,
}

impl Attention {
    /// Apply the layer's configured RoPE variant.
    ///
    /// * Proportional RoPE (Gemma 4 full-attention layers, `rope_type ==
    ///   "proportional"`): call `mx.fast.rope` across the full `head_dim`
    ///   with exponents normalized by the full dimension and an `inf`
    ///   frequency tail. This matches `mlx_lm.models.rope_utils.ProportionalRoPE`.
    /// * Default RoPE (sliding-attention layers and legacy configs): rotate
    ///   only the first `rope_dims = head_dim * partial_rotary_factor` slots
    ///   with the standard `fast_rope` kernel.
    fn apply_rope(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        if let Some(ref freqs) = self.proportional_rope_freqs {
            mlxcel_core::rope_proportional::apply_proportional_rope(
                x,
                self.head_dim,
                self.proportional_partial_rotary_factor,
                offset,
                Some(freqs),
            )
        } else {
            mlxcel_core::fast_rope(x, self.rope_dims, false, self.rope_theta, 1.0, offset)
        }
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut dyn CacheInterface,
        shared_kv: Option<(&MlxArray, &MlxArray)>,
    ) -> (
        UniquePtr<MlxArray>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    ) {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let offset = cache.offset();

        let q_proj_out = match &self.projection {
            AttentionProjection::Fused(proj) => {
                let (q, _, _) = proj.forward(x);
                q
            }
            AttentionProjection::Separate { q_proj, .. } => q_proj.forward(x),
        };

        // Fast path: full-attention Gemma 4 layers run
        // `reshape -> q_norm -> transpose -> full-head ProportionalRoPE`
        // inside a single `mx::core::compile` window. Sliding layers and
        // layers with non-proportional RoPE stay on the op-at-a-time chain.
        let queries = if let Some(ref freqs) = self.proportional_rope_freqs {
            let rotated_dims = 2 * ((self.proportional_partial_rotary_factor as f64
                * self.head_dim as f64
                / 2.0)
                .floor() as i32)
                .max(0);
            mlxcel_core::compiled_q_path_proportional(
                &q_proj_out,
                &self.q_norm.weight,
                freqs,
                self.q_norm.eps,
                self.n_heads,
                self.head_dim,
                rotated_dims,
                offset,
            )
        } else {
            let queries = mlxcel_core::reshape(&q_proj_out, &[b, l, self.n_heads, self.head_dim]);
            let queries = self.q_norm.forward(&queries);
            let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
            self.apply_rope(&queries, offset)
        };

        if self.is_kv_shared_layer
            && let Some((keys, values)) = shared_kv
        {
            let attn_out = self.attend(&queries, keys, values, mask);
            return (self.project_output(&attn_out, b, l), None);
        }

        let (keys, values) = self.project_kv(x, b, l, offset, cache);
        let attn_out = self.attend(&queries, &keys, &values, mask);
        let stored = if self.store_full_length_kv {
            Some((keys, values))
        } else {
            None
        };

        (self.project_output(&attn_out, b, l), stored)
    }

    fn attend(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let local_mask = trim_mask_to_keys(mask, keys);

        // When mask was discarded (undersized) or originally None,
        // use causal attention if possible.
        if local_mask.is_none() {
            return mlxcel_core::causal_attention(
                queries,
                keys,
                values,
                self.scale,
                0.0,
                self.window_size,
            );
        }

        let mask_ptr = local_mask
            .as_ref()
            .map(|m| m.as_ref().unwrap() as *const MlxArray)
            .unwrap_or(std::ptr::null());

        unsafe {
            mlxcel_core::layers::attention_from_ptr(
                queries,
                keys,
                values,
                self.scale,
                mask_ptr,
                0.0,
                self.window_size,
            )
        }
    }

    fn project_output(&self, attn_out: &MlxArray, b: i32, l: i32) -> UniquePtr<MlxArray> {
        let attn_out = mlxcel_core::transpose_axes(attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.n_heads * self.head_dim]);
        self.o_proj.forward(&attn_out)
    }

    fn project_kv(
        &self,
        x: &MlxArray,
        b: i32,
        l: i32,
        offset: i32,
        cache: &mut dyn CacheInterface,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let (raw_keys, raw_values) = match &self.projection {
            AttentionProjection::Fused(proj) => {
                let (_, k, v) = proj.forward(x);
                (k, v)
            }
            AttentionProjection::Separate { k_proj, v_proj, .. } => {
                let raw_keys = k_proj.forward(x);
                let raw_values = if self.use_k_eq_v {
                    mlxcel_core::copy(&raw_keys)
                } else {
                    v_proj
                        .as_ref()
                        .expect("Gemma4 attention expected v_proj for non-k_eq_v layer")
                        .forward(x)
                };
                (raw_keys, raw_values)
            }
        };

        // Fast path: on full-attention layers the K branch is the same
        // `reshape -> norm -> transpose -> full-head ProportionalRoPE` shape
        // as the Q branch, so it reuses `compiled_q_path_proportional` with
        // `n_kv_heads` and the k_norm weight.
        let keys = if let Some(ref freqs) = self.proportional_rope_freqs {
            let rotated_dims = 2 * ((self.proportional_partial_rotary_factor as f64
                * self.head_dim as f64
                / 2.0)
                .floor() as i32)
                .max(0);
            mlxcel_core::compiled_q_path_proportional(
                &raw_keys,
                &self.k_norm.weight,
                freqs,
                self.k_norm.eps,
                self.n_kv_heads,
                self.head_dim,
                rotated_dims,
                offset,
            )
        } else {
            let keys = mlxcel_core::reshape(&raw_keys, &[b, l, self.n_kv_heads, self.head_dim]);
            let keys = self.k_norm.forward(&keys);
            let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
            self.apply_rope(&keys, offset)
        };

        let values = mlxcel_core::reshape(&raw_values, &[b, l, self.n_kv_heads, self.head_dim]);
        let values = self.v_norm.forward(&values);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        cache.update_and_fetch(keys, values)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim_for_layer(layer_idx);
        let n_heads = config.num_attention_heads as i32;
        let n_kv_heads = config.num_kv_heads_for_layer(layer_idx);
        let rope_params = config.rope_params_for_layer(layer_idx);
        let rope_dims = (head_dim as f32 * rope_params.partial_rotary_factor) as i32;
        // For `rope_type == "proportional"`, precompute the proportional
        // frequency table once per layer. The exponents are normalized by the
        // full head_dim, and the non-rotated tail is filled with `inf`,
        // matching upstream `ProportionalRoPE` semantics.
        // Other rope_types fall through to the historical partial-dim
        // `fast_rope` path (no precomputed freqs).
        let proportional_rope_freqs = if rope_params.rope_type == "proportional" {
            mlxcel_core::rope_proportional::compute_proportional_rope_freqs(
                head_dim,
                rope_params.partial_rotary_factor,
                rope_params.rope_theta,
                1.0,
            )
        } else {
            None
        };
        let proportional_partial_rotary_factor = rope_params.partial_rotary_factor;
        let use_k_eq_v = config.attention_k_eq_v && !config.is_sliding_layer(layer_idx);
        let first_kv_shared_idx = config.first_kv_shared_layer_idx();
        let is_kv_shared_layer = config.is_kv_shared_layer(layer_idx);

        let kv_shared_layer_index = if is_kv_shared_layer {
            let prev_layers = &config.layer_types[..first_kv_shared_idx];
            Some(
                prev_layers
                    .iter()
                    .rposition(|layer_type| layer_type == config.layer_types[layer_idx].as_str())
                    .ok_or_else(|| {
                        format!(
                            "Failed to locate KV-sharing source layer for Gemma4 layer {}",
                            layer_idx
                        )
                    })?,
            )
        } else {
            None
        };

        let store_full_length_kv = if !is_kv_shared_layer && first_kv_shared_idx > 0 {
            let prev_layers = &config.layer_types[..first_kv_shared_idx];
            prev_layers
                .iter()
                .rposition(|layer_type| layer_type == config.layer_types[layer_idx].as_str())
                .map(|idx| idx == layer_idx)
                .unwrap_or(false)
        } else {
            false
        };

        // Gemma 4 matches mlx-lm's separate Q/K/V projection path by default.
        // A concatenated QKV projection reduces QuantizedMatmul count, but on
        // 26B/31B decode it adds slice-heavy graphs and measures slower.
        let enable_fused_qkv = std::env::var_os("MLXCEL_GEMMA4_ENABLE_FUSED_QKV").is_some();
        let projection = if use_k_eq_v {
            AttentionProjection::Separate {
                q_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.q_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?,
                k_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.k_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?,
                v_proj: None,
            }
        } else if enable_fused_qkv {
            AttentionProjection::Fused(FusedQKVLinear::from_weights_separate(
                weights,
                prefix,
                config.group_size(),
                config.bits(),
                n_heads,
                n_kv_heads,
                head_dim,
            )?)
        } else {
            AttentionProjection::Separate {
                q_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.q_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?,
                k_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.k_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?,
                v_proj: Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.v_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?),
            }
        };

        Ok(Self {
            projection,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.o_proj", prefix),
                config.group_size(),
                config.bits(),
            )?,
            q_norm: RMSNorm::new(
                get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?,
                config.rms_norm_eps,
            ),
            k_norm: RMSNorm::new(
                get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?,
                config.rms_norm_eps,
            ),
            v_norm: RMSNormNoScale::new(head_dim, config.rms_norm_eps),
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dims,
            rope_theta: rope_params.rope_theta,
            proportional_rope_freqs,
            proportional_partial_rotary_factor,
            scale: 1.0,
            is_kv_shared_layer,
            kv_shared_layer_index,
            store_full_length_kv,
            use_k_eq_v,
            window_size: if config.is_sliding_layer(layer_idx) {
                config.sliding_window as i32
            } else {
                0
            },
        })
    }
}

fn trim_mask_to_keys(mask: Option<&MlxArray>, keys: &MlxArray) -> Option<UniquePtr<MlxArray>> {
    let mask = mask?;
    let mask_shape = mlxcel_core::array_shape(mask);
    let key_shape = mlxcel_core::array_shape(keys);
    let key_len = key_shape[2];
    let mask_len = *mask_shape.last().unwrap_or(&0);
    if mask_len == key_len {
        Some(mlxcel_core::copy(mask))
    } else if mask_len > key_len {
        let start = mask_len - key_len;
        Some(slice_axis(mask, -1, start, mask_len))
    } else {
        // mask is smaller than key length — this happens when an external
        // mask was created with stale offset (e.g. during chunked prefill on
        // non-batching models).  Discard the undersized caller mask and let
        // the attention kernel fall back to its internal causal handling by
        // returning None, which will cause the caller to pass a null mask.
        tracing::warn!(
            mask_len,
            key_len,
            "trim_mask_to_keys: mask shorter than key length, discarding caller mask"
        );
        None
    }
}

pub struct DecoderLayer {
    pub(crate) self_attn: Attention,
    pub(crate) mlp: MLP,
    pub(crate) input_layernorm: RMSNorm,
    pub(crate) post_attention_layernorm: RMSNorm,
    pub(crate) pre_feedforward_layernorm: RMSNorm,
    pub(crate) post_feedforward_layernorm: RMSNorm,
    pub(crate) router: Option<Router>,
    pub(crate) experts: Option<Experts>,
    pub(crate) post_feedforward_layernorm_1: Option<RMSNorm>,
    pub(crate) pre_feedforward_layernorm_2: Option<RMSNorm>,
    pub(crate) post_feedforward_layernorm_2: Option<RMSNorm>,
    pub(crate) per_layer_input_gate: Option<UnifiedLinear>,
    pub(crate) per_layer_projection: Option<UnifiedLinear>,
    pub(crate) post_per_layer_input_norm: Option<RMSNorm>,
    pub(crate) layer_scalar: UniquePtr<MlxArray>,
    pub(crate) layer_type: String,
}

/// Optional sub-op timer used by `MLXCEL_PROFILE_LAYER_SUBOPS=1` inside
/// `DecoderLayer::forward`. Each `tick` evaluates the tensor (forcing a sync)
/// and prints `[SUBOP ...] layer=.. name=.. ms=..` to stderr.
struct SubopTimer {
    enabled: bool,
    layer_idx: usize,
    last: std::time::Instant,
}

impl SubopTimer {
    fn new(enabled: bool, layer_idx: usize) -> Self {
        Self {
            enabled,
            layer_idx,
            last: std::time::Instant::now(),
        }
    }

    fn tick(&mut self, name: &str, tensor: &MlxArray) {
        if !self.enabled {
            return;
        }
        mlxcel_core::eval(tensor);
        let dt = self.last.elapsed();
        eprintln!(
            "[SUBOP] layer={:02} name={} ms={:.4}",
            self.layer_idx,
            name,
            dt.as_secs_f64() * 1000.0
        );
        self.last = std::time::Instant::now();
    }
}

impl DecoderLayer {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut dyn CacheInterface,
        per_layer_input: Option<&MlxArray>,
        shared_kv: Option<(&MlxArray, &MlxArray)>,
    ) -> (
        UniquePtr<MlxArray>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    ) {
        self.forward_with_profile(
            x,
            mask,
            cache,
            per_layer_input,
            shared_kv,
            usize::MAX,
            false,
        )
    }

    pub(crate) fn forward_with_profile(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut dyn CacheInterface,
        per_layer_input: Option<&MlxArray>,
        shared_kv: Option<(&MlxArray, &MlxArray)>,
        layer_idx: usize,
        profile_subops: bool,
    ) -> (
        UniquePtr<MlxArray>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    ) {
        let mut timer = SubopTimer::new(profile_subops, layer_idx);
        // Mirror mlx-lm's `residual = x` aliasing by holding a reference to
        // the prior stage's tensor instead of copying it. The earlier
        // `mlxcel_core::copy` calls produced a MLX Copy primitive and a fresh
        // UniquePtr<MlxArray> per layer; both are wasted work since the
        // residual is only *read* by the final `add`.
        let h_attn = self.input_layernorm.forward(x);
        timer.tick("input_layernorm", &h_attn);
        let (h_attn, stored_kv) = self.self_attn.forward(&h_attn, mask, cache, shared_kv);
        timer.tick("self_attn", &h_attn);
        let h_attn = self.post_attention_layernorm.forward(&h_attn);
        timer.tick("post_attention_layernorm", &h_attn);
        let after_attn = mlxcel_core::add(x, &h_attn);
        timer.tick("attn_residual_add", &after_attn);

        let ffn_out = if let (Some(router), Some(experts)) = (&self.router, &self.experts) {
            let h1 = self.pre_feedforward_layernorm.forward(&after_attn);
            timer.tick("pre_ffn_ln_shared_mlp", &h1);
            let h1 = self.mlp.forward(&h1);
            timer.tick("shared_mlp", &h1);
            let h1 = self
                .post_feedforward_layernorm_1
                .as_ref()
                .expect("Missing Gemma4 MoE post_feedforward_layernorm_1")
                .forward(&h1);
            timer.tick("post_shared_mlp_ln", &h1);

            let (top_k_indices, top_k_weights) = router.forward(&after_attn);
            timer.tick("router", &top_k_indices);
            let h2 = self
                .pre_feedforward_layernorm_2
                .as_ref()
                .expect("Missing Gemma4 MoE pre_feedforward_layernorm_2")
                .forward(&after_attn);
            timer.tick("pre_moe_ln", &h2);
            let h2 = experts.forward(&h2, &top_k_indices, &top_k_weights);
            timer.tick("experts", &h2);
            let h2 = self
                .post_feedforward_layernorm_2
                .as_ref()
                .expect("Missing Gemma4 MoE post_feedforward_layernorm_2")
                .forward(&h2);
            timer.tick("post_moe_ln", &h2);
            let combined = mlxcel_core::add(&h1, &h2);
            timer.tick("moe_shared_add", &combined);
            combined
        } else {
            let h_norm = self.pre_feedforward_layernorm.forward(&after_attn);
            timer.tick("pre_ffn_ln", &h_norm);
            let out = self.mlp.forward(&h_norm);
            timer.tick("mlp", &out);
            out
        };

        let ffn_out = self.post_feedforward_layernorm.forward(&ffn_out);
        timer.tick("post_ffn_ln", &ffn_out);
        let after_ffn = mlxcel_core::add(&after_attn, &ffn_out);
        timer.tick("ffn_residual_add", &after_ffn);

        let h = if let (Some(gate_proj), Some(proj), Some(post_norm), Some(per_layer_input)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            // Fast path: when both gate and proj are quantized,
            // collapse the whole chain
            //   gate_proj → gelu_approx → mul(per_layer) → proj →
            //   post_norm → add(after_ffn)
            // into a single `mx::core::compile` graph. Falls back
            // to the op-at-a-time chain for non-quantized variants.
            let combined =
                if let (Some(gate_qw), Some(proj_qw)) =
                    (gate_proj.quantized_weight(), proj.quantized_weight())
                {
                    unsafe {
                        mlxcel_core::compiled_per_layer_input_gate(
                            &after_ffn,
                            per_layer_input,
                            &gate_qw.weight,
                            &gate_qw.scales,
                            gate_qw.biases_ptr(),
                            &proj_qw.weight,
                            &proj_qw.scales,
                            proj_qw.biases_ptr(),
                            &post_norm.weight,
                            post_norm.eps,
                            gate_qw.group_size,
                            gate_qw.bits,
                            &gate_qw.mode,
                        )
                    }
                } else {
                    let gate = gate_proj.forward(&after_ffn);
                    let gated =
                        mlxcel_core::compiled_geglu_approx_activation(&gate, per_layer_input);
                    let gate = proj.forward(&gated);
                    let gate = post_norm.forward(&gate);
                    mlxcel_core::add(&after_ffn, &gate)
                };
            timer.tick("per_layer_input_gate", &combined);
            combined
        } else {
            after_ffn
        };

        let h = mlxcel_core::multiply(&h, &self.layer_scalar);
        timer.tick("layer_scalar", &h);
        (h, stored_kv)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let enable_moe = config.enable_moe_block;
        let has_per_layer_input = config.hidden_size_per_layer_input > 0;

        Ok(Self {
            self_attn: Attention::from_weights(
                weights,
                config,
                layer_idx,
                &format!("{}.self_attn", prefix),
            )?,
            mlp: MLP::from_weights(weights, config, layer_idx, &format!("{}.mlp", prefix))?,
            input_layernorm: RMSNorm::new(
                get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?,
                config.rms_norm_eps,
            ),
            post_attention_layernorm: RMSNorm::new(
                get_weight_copy(
                    weights,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                config.rms_norm_eps,
            ),
            pre_feedforward_layernorm: RMSNorm::new(
                get_weight_copy(
                    weights,
                    &format!("{}.pre_feedforward_layernorm.weight", prefix),
                )?,
                config.rms_norm_eps,
            ),
            post_feedforward_layernorm: RMSNorm::new(
                get_weight_copy(
                    weights,
                    &format!("{}.post_feedforward_layernorm.weight", prefix),
                )?,
                config.rms_norm_eps,
            ),
            router: if enable_moe {
                Some(Router::from_weights(
                    weights,
                    config,
                    &format!("{}.router", prefix),
                )?)
            } else {
                None
            },
            experts: if enable_moe {
                Some(Experts::from_weights(
                    weights,
                    config,
                    &format!("{}.experts", prefix),
                )?)
            } else {
                None
            },
            post_feedforward_layernorm_1: if enable_moe {
                Some(RMSNorm::new(
                    get_weight_copy(
                        weights,
                        &format!("{}.post_feedforward_layernorm_1.weight", prefix),
                    )?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            },
            pre_feedforward_layernorm_2: if enable_moe {
                Some(RMSNorm::new(
                    get_weight_copy(
                        weights,
                        &format!("{}.pre_feedforward_layernorm_2.weight", prefix),
                    )?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            },
            post_feedforward_layernorm_2: if enable_moe {
                Some(RMSNorm::new(
                    get_weight_copy(
                        weights,
                        &format!("{}.post_feedforward_layernorm_2.weight", prefix),
                    )?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            },
            per_layer_input_gate: if has_per_layer_input {
                Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.per_layer_input_gate", prefix),
                    config.group_size(),
                    config.bits(),
                )?)
            } else {
                None
            },
            per_layer_projection: if has_per_layer_input {
                Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.per_layer_projection", prefix),
                    config.group_size(),
                    config.bits(),
                )?)
            } else {
                None
            },
            post_per_layer_input_norm: if has_per_layer_input {
                Some(RMSNorm::new(
                    get_weight_copy(
                        weights,
                        &format!("{}.post_per_layer_input_norm.weight", prefix),
                    )?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            },
            layer_scalar: get_weight_copy(weights, &format!("{}.layer_scalar", prefix))?,
            layer_type: config.layer_type(layer_idx).to_string(),
        })
    }
}

pub struct Gemma4TextModel {
    pub(crate) embed_tokens: UnifiedEmbedding,
    pub(crate) embed_tokens_per_layer: Option<UnifiedEmbedding>,
    pub(crate) per_layer_model_projection: Option<UnifiedLinear>,
    pub(crate) per_layer_projection_scale: f32,
    pub(crate) per_layer_projection_norm: Option<RMSNorm>,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) norm: RMSNorm,
    pub(crate) config: TextConfig,
}

impl Gemma4TextModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        mask: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // When `input_embeddings` is supplied (e.g. from the VLM path where
        // vision/audio features have already been merged into the embedding
        // stream), the caller is responsible for applying the
        // `sqrt(hidden_size)` embed scale to the text portion *before*
        // merging. Scaling here would double-scale the text tokens and
        // incorrectly scale image/audio features that are already in the
        // language-model embedding space. See issue #317.
        let mut h = if let Some(embeddings) = input_embeddings {
            mlxcel_core::copy(embeddings)
        } else {
            let embeds = self.embed_tokens.forward(input_ids);
            mlxcel_core::multiply_scalar(&embeds, (self.config.hidden_size as f32).sqrt())
        };

        let shape = mlxcel_core::array_shape(&h);
        let b = shape[0];
        let l = shape[1];

        let per_layer_inputs = if self.config.hidden_size_per_layer_input > 0 {
            let inputs = if let Some(inputs) = per_layer_inputs {
                mlxcel_core::copy(inputs)
            } else {
                self.get_per_layer_inputs(input_ids)
            };
            Some(self.project_per_layer_inputs(&h, Some(&inputs)))
        } else {
            None
        };

        let (global_mask, sliding_mask) = if let Some(mask) = mask {
            (Some(mlxcel_core::copy(mask)), Some(mlxcel_core::copy(mask)))
        } else if l > 1 {
            let global_offset = first_cache_offset(caches, "full_attention");
            let sliding_offset = first_cache_offset(caches, "sliding_attention");
            let sliding_effective_offset =
                sliding_offset.min((self.config.sliding_window as i32 - l).max(0));

            (
                Some(create_causal_mask(l, global_offset)),
                Some(create_causal_mask_with_window(
                    l,
                    sliding_effective_offset,
                    Some(self.config.sliding_window as i32),
                )),
            )
        } else {
            (None, None)
        };

        let mut shared_kv_store: HashMap<usize, (UniquePtr<MlxArray>, UniquePtr<MlxArray>, i32)> =
            HashMap::new();
        let n_layers = self.layers.len();

        // Per-layer profiling: set `MLXCEL_PROFILE_PER_LAYER=1` to dump a
        // `[LAYER xx] type=... ms=...` line to stderr for every layer on
        // every call. Forces `eval` before and after each layer so the
        // measured times include all kernels queued inside that layer.
        // Distorts absolute decode throughput (short-circuits graph
        // fusion), so use it for relative per-layer breakdowns only.
        //
        // `MLXCEL_PROFILE_LAYER_SUBOPS=1` extends this by emitting a
        // `[SUBOP] layer=xx name=.. ms=..` line per sub-step inside the
        // layer (input_ln, self_attn, post_ln, MoE router/experts, etc.).
        let profile_per_layer = std::env::var("MLXCEL_PROFILE_PER_LAYER").is_ok();
        let profile_layer_build = std::env::var("MLXCEL_PROFILE_LAYER_BUILD").is_ok();
        let profile_subops = std::env::var("MLXCEL_PROFILE_LAYER_SUBOPS").is_ok();

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = caches[i].as_interface();
            let mut shared_kv = None;

            if layer.self_attn.is_kv_shared_layer
                && let Some(ref_idx) = layer.self_attn.kv_shared_layer_index
                && let Some((keys, values, ref_offset)) = shared_kv_store.get(&ref_idx)
            {
                cache.set_offset(*ref_offset);
                shared_kv = Some((keys.as_ref().unwrap(), values.as_ref().unwrap()));
            }

            let local_mask = match layer.layer_type.as_str() {
                "full_attention" => global_mask.as_ref().map(|m| m.as_ref().unwrap()),
                _ => sliding_mask.as_ref().map(|m| m.as_ref().unwrap()),
            };

            let layer_input = per_layer_inputs.as_ref().map(|inputs| {
                slice_layer_input(
                    inputs,
                    i as i32,
                    b,
                    l,
                    self.config.hidden_size_per_layer_input as i32,
                )
            });

            let pre_offset = cache.offset();
            let layer_start = if profile_per_layer {
                mlxcel_core::eval(&h);
                Some(std::time::Instant::now())
            } else {
                None
            };
            let layer_build_start = if profile_layer_build {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let (next_h, stored_kv) = layer.forward_with_profile(
                &h,
                local_mask,
                cache,
                layer_input.as_ref().map(|arr| arr.as_ref().unwrap()),
                shared_kv,
                i,
                profile_subops,
            );
            h = next_h;
            if let Some(start) = layer_build_start {
                eprintln!(
                    "[LAYER_BUILD {:02}] type={} seq_len={} ms={:.3}",
                    i,
                    layer.layer_type,
                    l,
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
            if let Some(start) = layer_start {
                mlxcel_core::eval(&h);
                eprintln!(
                    "[LAYER {:02}] type={} seq_len={} ms={:.3}",
                    i,
                    layer.layer_type,
                    l,
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }

            if let Some((keys, values)) = stored_kv {
                shared_kv_store.insert(i, (keys, values, pre_offset));
            }

            pipeline_hint(&h, i, n_layers);
        }

        self.norm.forward(&h)
    }

    pub(crate) fn get_per_layer_inputs(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let embedded = self
            .embed_tokens_per_layer
            .as_ref()
            .expect("Gemma4 per-layer embeddings missing")
            .forward(input_ids);
        let embedded = mlxcel_core::multiply_scalar(
            &embedded,
            (self.config.hidden_size_per_layer_input as f32).sqrt(),
        );

        let shape = mlxcel_core::array_shape(input_ids);
        mlxcel_core::reshape(
            &embedded,
            &[
                shape[0],
                shape[1],
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        )
    }

    pub(crate) fn project_per_layer_inputs(
        &self,
        inputs_embeds: &MlxArray,
        per_layer_inputs: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let projected = self
            .per_layer_model_projection
            .as_ref()
            .expect("Gemma4 per_layer_model_projection missing")
            .forward(inputs_embeds);
        let projected = mlxcel_core::multiply_scalar(&projected, self.per_layer_projection_scale);
        let shape = mlxcel_core::array_shape(inputs_embeds);
        let projected = mlxcel_core::reshape(
            &projected,
            &[
                shape[0],
                shape[1],
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        );
        let projected = self
            .per_layer_projection_norm
            .as_ref()
            .expect("Gemma4 per_layer_projection_norm missing")
            .forward(&projected);

        if let Some(per_layer_inputs) = per_layer_inputs {
            let sum = mlxcel_core::add(&projected, per_layer_inputs);
            mlxcel_core::multiply_scalar(&sum, std::f32::consts::FRAC_1_SQRT_2)
        } else {
            projected
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            &format!("{}.embed_tokens", prefix),
            config.group_size(),
            config.bits(),
        )?;

        let embed_tokens_per_layer = if config.hidden_size_per_layer_input > 0 {
            Some(UnifiedEmbedding::from_weights(
                weights,
                &format!("{}.embed_tokens_per_layer", prefix),
                config.group_size(),
                config.bits(),
            )?)
        } else {
            None
        };

        let per_layer_projection_scale = (config.hidden_size as f32).powf(-0.5);
        let per_layer_model_projection = if config.hidden_size_per_layer_input > 0 {
            Some(UnifiedLinear::from_weights(
                weights,
                &format!("{}.per_layer_model_projection", prefix),
                config.group_size(),
                config.bits(),
            )?)
        } else {
            None
        };

        let per_layer_projection_norm = if config.hidden_size_per_layer_input > 0 {
            Some(RMSNorm::new(
                get_weight_copy(
                    weights,
                    &format!("{}.per_layer_projection_norm.weight", prefix),
                )?,
                config.rms_norm_eps,
            ))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(
                weights,
                config,
                layer_idx,
                &format!("{}.layers.{}", prefix, layer_idx),
            )?);
        }

        let norm = RMSNorm::new(
            get_weight_copy(weights, &format!("{}.norm.weight", prefix))?,
            config.rms_norm_eps,
        );

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_scale,
            per_layer_projection_norm,
            layers,
            norm,
            config: config.clone(),
        })
    }

    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        self.config
            .layer_types
            .iter()
            .map(|layer_type| {
                if layer_type == "full_attention" {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.config.sliding_window as i32))
                }
            })
            .collect()
    }
}

pub(crate) fn first_cache_offset(caches: &mut [Cache], layer_type: &str) -> i32 {
    for cache in caches.iter_mut() {
        match (layer_type, cache) {
            ("full_attention", Cache::Standard(c)) => return c.offset,
            ("sliding_attention", Cache::Rotating(c)) => return c.offset,
            _ => {}
        }
    }
    0
}

pub(crate) fn slice_layer_input(
    layer_inputs: &MlxArray,
    layer_idx: i32,
    batch: i32,
    seq_len: i32,
    hidden_size: i32,
) -> UniquePtr<MlxArray> {
    let sliced = mlxcel_core::slice(
        layer_inputs,
        &[0, 0, layer_idx, 0],
        &[batch, seq_len, layer_idx + 1, hidden_size],
    );
    mlxcel_core::squeeze_axis(&sliced, 2)
}

pub struct Gemma4Model {
    pub(crate) text_model: Gemma4TextModel,
    pub(crate) config: TextConfig,
    pub(crate) eos_token_ids: Vec<i32>,
    _weight_backing: super::sanitize::Gemma4WeightBacking,
}

impl Gemma4Model {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        let config_value: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse sanitized config.json: {}", e))?;

        let is_quantized = config_value.get("quantization").is_some()
            || config_value
                .get("text_config")
                .and_then(|text| text.get("quantization"))
                .is_some();
        let (mut weights, weight_backing) = if is_quantized {
            super::sanitize::load_gemma4_text_weights_with_backing(model_dir)?
        } else {
            (
                crate::models::load_and_sanitize_weights(model_dir)?,
                super::sanitize::Gemma4WeightBacking::default(),
            )
        };
        crate::models::sanitize_tied_embeddings(&mut weights, &config_value);
        let mut model = Self::from_weights(&weights, &args)?;
        model._weight_backing = weight_backing;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let config = args.text_args();
        let text_model = Gemma4TextModel::from_weights(weights, &config, "language_model.model")?;
        let eos_token_ids = {
            let eos = args.eos_token_ids();
            if eos.is_empty() {
                vec![1, 106, 50]
            } else {
                eos
            }
        };

        Ok(Self {
            text_model,
            config,
            eos_token_ids,
            _weight_backing: super::sanitize::Gemma4WeightBacking::default(),
        })
    }

    fn forward_with_caches_and_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        mask: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let hidden =
            self.text_model
                .forward(input_ids, input_embeddings, caches, mask, per_layer_inputs);
        let mut logits = self.text_model.embed_tokens.as_linear(&hidden);
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = mlxcel_core::compiled_softcap(&logits, cap);
        }
        logits
    }

    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        self.text_model.make_caches()
    }
}

pub(crate) struct Gemma4StageModel {
    filter: LayerFilter,
    embed_tokens: Option<UnifiedEmbedding>,
    embed_tokens_per_layer: Option<UnifiedEmbedding>,
    per_layer_model_projection: Option<UnifiedLinear>,
    per_layer_projection_scale: f32,
    per_layer_projection_norm: Option<RMSNorm>,
    layers: Vec<(usize, DecoderLayer)>,
    norm: Option<RMSNorm>,
    config: TextConfig,
    final_logit_softcapping: Option<f32>,
    _weight_backing: super::sanitize::Gemma4WeightBacking,
}

impl Gemma4StageModel {
    pub(crate) fn load(
        model_dir: &Path,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        let config_value: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse sanitized config.json: {}", e))?;

        let is_quantized = config_value.get("quantization").is_some()
            || config_value
                .get("text_config")
                .and_then(|text| text.get("quantization"))
                .is_some();
        let (mut weights, weight_backing) = if is_quantized {
            super::sanitize::load_gemma4_text_weights_with_backing(model_dir)?
        } else {
            (
                crate::models::load_and_sanitize_weights(model_dir)?,
                super::sanitize::Gemma4WeightBacking::default(),
            )
        };
        crate::models::sanitize_tied_embeddings(&mut weights, &config_value);
        let mut effective_filter = filter.clone();
        if filter.has_lm_head {
            effective_filter.has_embedding = true;
        }
        filter_weight_map(&mut weights, &effective_filter);
        Self::from_filtered_weights(&weights, &args, filter, stage_index, weight_backing)
    }

    fn from_filtered_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        filter: &LayerFilter,
        stage_index: usize,
        weight_backing: super::sanitize::Gemma4WeightBacking,
    ) -> Result<Self, String> {
        let config = args.text_args();
        let prefix = "language_model.model";

        let embed_tokens = if filter.has_embedding || filter.has_lm_head {
            Some(UnifiedEmbedding::from_weights(
                weights,
                &format!("{}.embed_tokens", prefix),
                config.group_size(),
                config.bits(),
            )?)
        } else {
            None
        };

        let embed_tokens_per_layer =
            if filter.has_embedding && config.hidden_size_per_layer_input > 0 {
                Some(UnifiedEmbedding::from_weights(
                    weights,
                    &format!("{}.embed_tokens_per_layer", prefix),
                    config.group_size(),
                    config.bits(),
                )?)
            } else {
                None
            };

        let per_layer_projection_scale = (config.hidden_size as f32).powf(-0.5);
        let per_layer_model_projection =
            if filter.has_embedding && config.hidden_size_per_layer_input > 0 {
                Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.per_layer_model_projection", prefix),
                    config.group_size(),
                    config.bits(),
                )?)
            } else {
                None
            };

        let per_layer_projection_norm =
            if filter.has_embedding && config.hidden_size_per_layer_input > 0 {
                Some(RMSNorm::new(
                    get_weight_copy(
                        weights,
                        &format!("{}.per_layer_projection_norm.weight", prefix),
                    )?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            };

        let mut layers = Vec::with_capacity(filter.num_layers());
        for layer_idx in filter.layer_range.clone() {
            let layer = DecoderLayer::from_weights(
                weights,
                &config,
                layer_idx,
                &format!("{}.layers.{}", prefix, layer_idx),
            )?;
            if layer.self_attn.is_kv_shared_layer
                && let Some(shared_idx) = layer.self_attn.kv_shared_layer_index
                && !filter.layer_range.contains(&shared_idx)
            {
                return Err(format!(
                    "stage {} cannot host Gemma4 layer {} because it shares KV with layer {} outside range {}..{}",
                    stage_index,
                    layer_idx,
                    shared_idx,
                    filter.layer_range.start,
                    filter.layer_range.end
                ));
            }
            layers.push((layer_idx, layer));
        }

        if layers.is_empty() {
            return Err(format!(
                "stage {} did not load any layers from range {}..{}",
                stage_index, filter.layer_range.start, filter.layer_range.end
            ));
        }

        let norm = if filter.has_lm_head {
            Some(RMSNorm::new(
                get_weight_copy(weights, &format!("{}.norm.weight", prefix))?,
                config.rms_norm_eps,
            ))
        } else {
            None
        };

        Ok(Self {
            filter: filter.clone(),
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_scale,
            per_layer_projection_norm,
            layers,
            norm,
            final_logit_softcapping: config.final_logit_softcapping,
            config,
            _weight_backing: weight_backing,
        })
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .map(|(layer_idx, _)| {
                if self.config.layer_type(*layer_idx) == "full_attention" {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.config.sliding_window as i32))
                }
            })
            .collect()
    }

    pub(crate) fn execute_from_token_ids(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        let mut hidden = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| {
                "stage does not host embeddings; hidden-state input required".to_string()
            })?
            .forward(input_ids);
        hidden = mlxcel_core::multiply_scalar(&hidden, (self.config.hidden_size as f32).sqrt());

        let stage_inputs = if self.config.hidden_size_per_layer_input > 0 {
            let raw_inputs = self.get_per_layer_inputs(input_ids)?;
            let projected = self.project_per_layer_inputs(
                hidden.as_ref().unwrap(),
                Some(raw_inputs.as_ref().unwrap()),
            )?;
            Some(projected)
        } else {
            None
        };
        let local_inputs = stage_inputs.as_ref().map(|inputs| {
            self.slice_layer_input_range(inputs.as_ref().unwrap(), self.filter.layer_range.clone())
        });
        let downstream_inputs = stage_inputs.as_ref().and_then(|inputs| {
            if self.filter.layer_range.end >= self.config.num_hidden_layers {
                None
            } else {
                Some(self.slice_layer_input_range(
                    inputs.as_ref().unwrap(),
                    self.filter.layer_range.end..self.config.num_hidden_layers,
                ))
            }
        });

        self.execute_hidden(hidden, local_inputs.as_ref(), downstream_inputs, caches)
    }

    pub(crate) fn execute_from_hidden_states(
        &self,
        packed_hidden: UniquePtr<MlxArray>,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        if self.filter.has_embedding {
            return Err("entry stage expects token IDs, not hidden states".to_string());
        }

        let (hidden, local_inputs, downstream_inputs) =
            self.unpack_stage_hidden(packed_hidden.as_ref().unwrap())?;
        self.execute_hidden(hidden, local_inputs.as_ref(), downstream_inputs, caches)
    }

    fn execute_hidden(
        &self,
        mut hidden: UniquePtr<MlxArray>,
        local_per_layer_inputs: Option<&UniquePtr<MlxArray>>,
        downstream_inputs: Option<UniquePtr<MlxArray>>,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        if caches.len() != self.layers.len() {
            return Err(format!(
                "stage cache count mismatch: expected {}, got {}",
                self.layers.len(),
                caches.len()
            ));
        }

        let shape = mlxcel_core::array_shape(hidden.as_ref().unwrap());
        let batch = shape[0];
        let seq_len = shape[1];

        let (global_mask, sliding_mask) = if seq_len > 1 {
            let global_offset = self.first_cache_offset(caches, "full_attention");
            let sliding_offset = self.first_cache_offset(caches, "sliding_attention");
            let sliding_effective_offset =
                sliding_offset.min((self.config.sliding_window as i32 - seq_len).max(0));
            (
                Some(create_causal_mask(seq_len, global_offset)),
                Some(create_causal_mask_with_window(
                    seq_len,
                    sliding_effective_offset,
                    Some(self.config.sliding_window as i32),
                )),
            )
        } else {
            (None, None)
        };

        let mut shared_kv_store: HashMap<usize, (UniquePtr<MlxArray>, UniquePtr<MlxArray>, i32)> =
            HashMap::new();
        let n_layers = self.layers.len();

        for (local_idx, (global_idx, layer)) in self.layers.iter().enumerate() {
            let cache = caches[local_idx].as_interface();
            let mut shared_kv = None;

            if layer.self_attn.is_kv_shared_layer
                && let Some(ref_idx) = layer.self_attn.kv_shared_layer_index
            {
                let (keys, values, ref_offset) =
                    shared_kv_store.get(&ref_idx).ok_or_else(|| {
                        format!(
                            "stage missing shared KV source layer {} for Gemma4 layer {}",
                            ref_idx, global_idx
                        )
                    })?;
                cache.set_offset(*ref_offset);
                shared_kv = Some((keys.as_ref().unwrap(), values.as_ref().unwrap()));
            }

            let local_mask = match layer.layer_type.as_str() {
                "full_attention" => global_mask.as_ref().map(|m| m.as_ref().unwrap()),
                _ => sliding_mask.as_ref().map(|m| m.as_ref().unwrap()),
            };

            let layer_input = local_per_layer_inputs.as_ref().map(|inputs| {
                slice_layer_input(
                    inputs.as_ref().unwrap(),
                    local_idx as i32,
                    batch,
                    seq_len,
                    self.config.hidden_size_per_layer_input as i32,
                )
            });

            let pre_offset = cache.offset();
            let (next_hidden, stored_kv) = layer.forward(
                hidden.as_ref().unwrap(),
                local_mask,
                cache,
                layer_input.as_ref().map(|arr| arr.as_ref().unwrap()),
                shared_kv,
            );
            hidden = next_hidden;

            if let Some((keys, values)) = stored_kv {
                shared_kv_store.insert(*global_idx, (keys, values, pre_offset));
            }

            pipeline_hint(&hidden, local_idx, n_layers);
        }

        if let Some(norm) = &self.norm {
            let hidden = norm.forward(hidden.as_ref().unwrap());
            let mut logits = self
                .embed_tokens
                .as_ref()
                .ok_or_else(|| "final Gemma4 stage missing embeddings".to_string())?
                .as_linear(&hidden);
            if let Some(cap) = self.final_logit_softcapping {
                logits = mlxcel_core::compiled_softcap(&logits, cap);
            }
            return Ok(StageExecutionOutput::Logits(logits));
        }

        Ok(StageExecutionOutput::HiddenStates(
            self.pack_hidden_for_downstream(hidden.as_ref().unwrap(), downstream_inputs.as_ref())?,
        ))
    }

    fn get_per_layer_inputs(&self, input_ids: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let embedded = self
            .embed_tokens_per_layer
            .as_ref()
            .ok_or_else(|| "Gemma4 per-layer embeddings missing".to_string())?
            .forward(input_ids);
        let embedded = mlxcel_core::multiply_scalar(
            &embedded,
            (self.config.hidden_size_per_layer_input as f32).sqrt(),
        );
        let shape = mlxcel_core::array_shape(input_ids);
        Ok(mlxcel_core::reshape(
            &embedded,
            &[
                shape[0],
                shape[1],
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        ))
    }

    fn project_per_layer_inputs(
        &self,
        inputs_embeds: &MlxArray,
        per_layer_inputs: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let projected = self
            .per_layer_model_projection
            .as_ref()
            .ok_or_else(|| "Gemma4 per_layer_model_projection missing".to_string())?
            .forward(inputs_embeds);
        let projected = mlxcel_core::multiply_scalar(&projected, self.per_layer_projection_scale);
        let shape = mlxcel_core::array_shape(inputs_embeds);
        let projected = mlxcel_core::reshape(
            &projected,
            &[
                shape[0],
                shape[1],
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        );
        let projected = self
            .per_layer_projection_norm
            .as_ref()
            .ok_or_else(|| "Gemma4 per_layer_projection_norm missing".to_string())?
            .forward(&projected);

        Ok(if let Some(per_layer_inputs) = per_layer_inputs {
            let sum = mlxcel_core::add(&projected, per_layer_inputs);
            mlxcel_core::multiply_scalar(&sum, std::f32::consts::FRAC_1_SQRT_2)
        } else {
            projected
        })
    }

    fn pack_hidden_for_downstream(
        &self,
        hidden: &MlxArray,
        downstream_inputs: Option<&UniquePtr<MlxArray>>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let Some(downstream_inputs) = downstream_inputs else {
            return Ok(mlxcel_core::copy(hidden));
        };

        let aux_shape = mlxcel_core::array_shape(downstream_inputs.as_ref().unwrap());
        let flat_aux = mlxcel_core::reshape(
            downstream_inputs.as_ref().unwrap(),
            &[aux_shape[0], aux_shape[1], aux_shape[2] * aux_shape[3]],
        );
        Ok(mlxcel_core::concatenate(hidden, &flat_aux, -1))
    }

    fn unpack_stage_hidden(
        &self,
        packed_hidden: &MlxArray,
    ) -> Result<
        (
            UniquePtr<MlxArray>,
            Option<UniquePtr<MlxArray>>,
            Option<UniquePtr<MlxArray>>,
        ),
        String,
    > {
        if self.config.hidden_size_per_layer_input == 0 {
            return Ok((mlxcel_core::copy(packed_hidden), None, None));
        }

        let shape = mlxcel_core::array_shape(packed_hidden);
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = self.config.hidden_size as i32;
        let remaining_layers =
            (self.config.num_hidden_layers - self.filter.layer_range.start) as i32;
        let hidden = slice_axis(packed_hidden, -1, 0, hidden_size);
        let flat_aux = slice_axis(
            packed_hidden,
            -1,
            hidden_size,
            hidden_size + remaining_layers * self.config.hidden_size_per_layer_input as i32,
        );
        let aux = mlxcel_core::reshape(
            &flat_aux,
            &[
                batch,
                seq_len,
                remaining_layers,
                self.config.hidden_size_per_layer_input as i32,
            ],
        );
        let local_layers = self.filter.num_layers() as i32;
        let local_inputs = self.slice_layer_input_range(&aux, 0..local_layers as usize);
        let downstream_inputs =
            if local_layers < remaining_layers {
                Some(self.slice_layer_input_range(
                    &aux,
                    local_layers as usize..remaining_layers as usize,
                ))
            } else {
                None
            };
        Ok((hidden, Some(local_inputs), downstream_inputs))
    }

    fn slice_layer_input_range(
        &self,
        inputs: &MlxArray,
        range: std::ops::Range<usize>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs);
        mlxcel_core::slice(
            inputs,
            &[0, 0, range.start as i32, 0],
            &[
                shape[0],
                shape[1],
                range.end as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        )
    }

    fn first_cache_offset(&self, caches: &[Cache], layer_type: &str) -> i32 {
        for cache in caches {
            match (layer_type, cache) {
                ("full_attention", Cache::Standard(cache)) => return cache.offset,
                ("sliding_attention", Cache::Rotating(cache)) => return cache.offset,
                _ => {}
            }
        }
        0
    }
}

pub struct Gemma4Wrapper {
    model: Gemma4Model,
    caches: RefCell<Vec<Cache>>,
}

impl Gemma4Wrapper {
    pub fn new(model: Gemma4Model) -> Self {
        let caches = model.make_caches();
        Self {
            model,
            caches: RefCell::new(caches),
        }
    }

    fn reset_caches(&self) {
        *self.caches.borrow_mut() = self.model.make_caches();
    }

    pub(crate) fn input_embeddings(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.model.text_model.embed_tokens.forward(input_ids)
    }

    pub(crate) fn get_per_layer_inputs(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        if self.model.config.hidden_size_per_layer_input == 0 {
            None
        } else {
            Some(self.model.text_model.get_per_layer_inputs(input_ids))
        }
    }

    pub(crate) fn project_per_layer_inputs(
        &self,
        inputs_embeds: &MlxArray,
        per_layer_inputs: Option<&MlxArray>,
    ) -> Option<UniquePtr<MlxArray>> {
        if self.model.config.hidden_size_per_layer_input == 0 {
            None
        } else {
            Some(
                self.model
                    .text_model
                    .project_per_layer_inputs(inputs_embeds, per_layer_inputs),
            )
        }
    }

    pub(crate) fn forward_with_inputs(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut caches = self.caches.borrow_mut();
        self.model.forward_with_caches_and_embeddings(
            input_ids,
            input_embeddings,
            &mut caches,
            mask,
            per_layer_inputs,
        )
    }

    pub(crate) fn num_layers_value(&self) -> usize {
        self.model.text_model.layers.len()
    }

    pub(crate) fn eos_token_ids_value(&self) -> Vec<i32> {
        self.model.eos_token_ids.clone()
    }

    /// Returns the text model's hidden size (embedding dimension).
    ///
    /// Used by: `Gemma4VLModel::get_input_embeddings_with_audio` to apply the
    /// `sqrt(hidden_size)` embed scale to text embeddings before merging in
    /// vision / audio features. Vision and audio features must NOT be scaled
    /// again since they are already in the language-model embedding space
    /// (see issue #317).
    pub fn hidden_size(&self) -> usize {
        self.model.config.hidden_size
    }
}

impl LanguageModel for Gemma4Wrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut caches = self.caches.borrow_mut();
        self.model
            .forward_with_caches_and_embeddings(input_ids, None, &mut caches, mask, None)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut caches = self.caches.borrow_mut();
        self.model.forward_with_caches_and_embeddings(
            input_ids,
            input_embeddings,
            &mut caches,
            mask,
            None,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.input_embeddings(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.reset_caches();
        (0..self.model.text_model.layers.len())
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.model.text_model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.model.eos_token_ids.clone()
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_padded_prefill(&self) -> bool {
        // Tried enabling NA tile-aligned padded prefill (both with explicit
        // mask and maskless-causal) for the per_layer_input-free variants
        // (26B-a4b, 31B) and measured no change vs the default bulk forward
        // (26B long prefill 1836 → 1832 tok/s, 31B 430 → 435 tok/s — all
        // within run-to-run noise). Gemma 4's attention mix (sliding window,
        // K=V for full-attention layers, proportional RoPE) does not benefit
        // from the tile-aligned path in the same way plain causal models do,
        // so we keep the conservative default until a profiling-driven
        // rewrite of the prefill path.
        false
    }
}
