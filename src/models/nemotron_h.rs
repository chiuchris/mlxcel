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

// Nemotron-H: NVIDIA's hybrid Mamba2+Transformer model for mlxcel-core
// Reference: mlx-lm/mlx_lm/models/nemotron_h.py
//
// Key features:
// - Hybrid architecture with configurable block types (M/*/−/E)
// - M = Mamba2 mixer, * = Attention, - = MLP, E = MoE
// - MambaRMSNormGated for Mamba blocks
// - Mixed cache: MambaCache for M blocks, KVCache for * blocks
// - relu^2 activation for MLP/MoE

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, Linear, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{
    create_causal_mask, relu_squared, repeat_kv, silu, slice_axis, stack_arrays,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::path::Path;

// Configuration.
// Custom deserializer for hybrid_override_pattern which can be either:
// - A string like "MEMEM*EMEMEM*..." (each char is a block type)
// - A Vec<String> like ["M", "E", "M", ...]
fn deserialize_hybrid_pattern<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};

    struct PatternVisitor;

    impl<'de> Visitor<'de> for PatternVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or array of strings for hybrid_override_pattern")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            // Convert string like "MEMEM*..." to Vec<String>
            Ok(v.chars().map(|c| c.to_string()).collect())
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut vec = Vec::new();
            while let Some(elem) = seq.next_element()? {
                vec.push(elem);
            }
            Ok(vec)
        }
    }

    deserializer.deserialize_any(PatternVisitor)
}

fn deserialize_time_step_limit<'de, D>(deserializer: D) -> Result<(f32, f32), D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{SeqAccess, Visitor};

    struct TimeStepLimitVisitor;

    impl<'de> Visitor<'de> for TimeStepLimitVisitor {
        type Value = (f32, f32);

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a tuple of two numbers")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let first: f32 = seq.next_element()?.unwrap_or(0.0);
            let second: f32 = seq
                .next_element::<serde_json::Value>()?
                .map(|v| match v {
                    serde_json::Value::Number(n) => n.as_f64().unwrap_or(f64::INFINITY) as f32,
                    _ => f32::INFINITY,
                })
                .unwrap_or(f32::INFINITY);
            Ok((first, second))
        }
    }

    deserializer.deserialize_seq(TimeStepLimitVisitor)
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NemotronHConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    #[serde(default)]
    pub attention_bias: bool,

    pub mamba_num_heads: usize,
    pub mamba_head_dim: usize,

    #[serde(default)]
    pub mamba_proj_bias: bool,

    pub ssm_state_size: usize,
    pub conv_kernel: usize,
    pub n_groups: usize,

    #[serde(
        default = "default_time_step_limit",
        deserialize_with = "deserialize_time_step_limit"
    )]
    pub time_step_limit: (f32, f32),

    #[serde(default)]
    pub mlp_bias: bool,

    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,

    #[serde(default)]
    pub use_bias: bool,

    #[serde(default = "default_true")]
    pub use_conv_bias: bool,

    #[serde(deserialize_with = "deserialize_hybrid_pattern")]
    pub hybrid_override_pattern: Vec<String>,

    #[serde(default)]
    pub head_dim: Option<usize>,

    // MoE parameters
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,

    #[serde(default)]
    pub moe_shared_expert_intermediate_size: Option<usize>,

    #[serde(default)]
    pub n_group: Option<usize>,

    #[serde(default)]
    pub n_routed_experts: Option<usize>,

    #[serde(default)]
    pub n_shared_experts: Option<usize>,

    #[serde(default)]
    pub topk_group: Option<usize>,

    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,

    #[serde(default)]
    pub norm_topk_prob: Option<bool>,

    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

impl NemotronHConfig {
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

fn default_time_step_limit() -> (f32, f32) {
    (0.0, f32::INFINITY)
}

fn default_layer_norm_epsilon() -> f32 {
    1e-5
}

fn default_true() -> bool {
    true
}

impl NemotronHConfig {
    pub fn get_mamba_intermediate_size(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }

    pub fn get_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn get_conv_dim(&self) -> usize {
        let intermediate_size = self.get_mamba_intermediate_size();
        intermediate_size + 2 * self.n_groups * self.ssm_state_size
    }
}

// Block Types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockType {
    Mamba,     // "M"
    Attention, // "*"
    MLP,       // "-"
    MoE,       // "E"
}

impl BlockType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "M" => BlockType::Mamba,
            "*" => BlockType::Attention,
            "-" => BlockType::MLP,
            "E" => BlockType::MoE,
            _ => panic!("Unknown block type: {}", s),
        }
    }

    pub fn needs_cache(&self) -> bool {
        matches!(self, BlockType::Mamba | BlockType::Attention)
    }
}

// Cache Types.
pub struct NemotronMambaCache {
    pub conv_state: Option<UniquePtr<MlxArray>>,
    pub ssm_state: Option<UniquePtr<MlxArray>>,
}

impl NemotronMambaCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
        }
    }
}

impl Default for NemotronMambaCache {
    fn default() -> Self {
        Self::new()
    }
}

pub enum NemotronLayerCache {
    Attention(KVCache),
    Mamba(NemotronMambaCache),
}

impl NemotronLayerCache {
    pub fn offset(&self) -> i32 {
        match self {
            NemotronLayerCache::Attention(kv) => kv.offset,
            NemotronLayerCache::Mamba(_) => 0,
        }
    }
}

// MambaRMSNormGated - RMS norm with gating and group structure.
struct MambaRMSNormGated {
    weight: UniquePtr<MlxArray>,
    eps: f32,
    group_size: usize,
}

impl MambaRMSNormGated {
    fn forward(&self, x: &MlxArray, gate: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        // Apply gating first if provided
        let x = if let Some(g) = gate {
            mlxcel_core::multiply(x, &silu(g))
        } else {
            mlxcel_core::copy(x)
        };

        // Unflatten last dim into groups
        let shape = mlxcel_core::array_shape(&x);
        let ndim = shape.len();
        let last_dim = shape[ndim - 1] as usize;
        let num_groups = last_dim / self.group_size;

        let mut new_shape: Vec<i32> = shape[..ndim - 1].to_vec();
        new_shape.push(num_groups as i32);
        new_shape.push(self.group_size as i32);
        let x_grouped = mlxcel_core::reshape(&x, &new_shape);

        // Apply RMS norm per group
        let group_weight =
            mlxcel_core::ones(&[self.group_size as i32], mlxcel_core::dtype::FLOAT32);
        let x_normed = mlxcel_core::fast_rms_norm(&x_grouped, &group_weight, self.eps);

        // Flatten back and apply weight
        let x_flat = mlxcel_core::reshape(&x_normed, &shape);
        mlxcel_core::multiply(&self.weight, &x_flat)
    }
}

// Mamba2 Mixer for Nemotron-H.
struct NemotronHMamba2Mixer {
    num_heads: usize,
    #[allow(dead_code)]
    hidden_size: usize,
    ssm_state_size: usize,
    conv_kernel_size: usize,
    intermediate_size: usize,
    n_groups: usize,
    head_dim: usize,
    time_step_limit: (f32, f32),
    conv_dim: usize,

    conv_weight: UniquePtr<MlxArray>,
    conv_bias: Option<UniquePtr<MlxArray>>,
    in_proj: UnifiedLinear,
    dt_bias: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    d_param: UniquePtr<MlxArray>,
    norm: MambaRMSNormGated,
    out_proj: UnifiedLinear,
}

impl NemotronHMamba2Mixer {
    fn forward(
        &self,
        hidden_states: &MlxArray,
        mut cache: Option<&mut NemotronMambaCache>,
    ) -> UniquePtr<MlxArray> {
        let projected = self.in_proj.forward(hidden_states);

        // Split: gate, conv_input, dt
        let gate = slice_axis(&projected, -1, 0, self.intermediate_size as i32);
        let conv_input = slice_axis(
            &projected,
            -1,
            self.intermediate_size as i32,
            (self.intermediate_size + self.conv_dim) as i32,
        );
        let dt = slice_axis(
            &projected,
            -1,
            (self.intermediate_size + self.conv_dim) as i32,
            -1,
        );

        // Conv with padding
        let shape = mlxcel_core::array_shape(&conv_input);
        let batch = shape[0];
        let seq_len = shape[1];
        let k = self.conv_kernel_size;

        let conv_state = cache.as_ref().and_then(|c| {
            c.conv_state
                .as_ref()
                .and_then(|s| s.as_ref().map(mlxcel_core::copy))
        });

        let padded_input = if let Some(ref conv_st) = conv_state {
            concatenate(conv_st, &conv_input, 1)
        } else {
            let pad_arr = mlxcel_core::zeros(
                &[batch, (k - 1) as i32, self.conv_dim as i32],
                mlxcel_core::dtype::FLOAT32,
            );
            concatenate(&pad_arr, &conv_input, 1)
        };

        // Update conv cache
        if let Some(c) = cache.as_deref_mut() {
            let padded_shape = mlxcel_core::array_shape(&padded_input);
            let len = padded_shape[1] as usize;
            c.conv_state = Some(slice_axis(
                &padded_input,
                1,
                (len - (k - 1)) as i32,
                len as i32,
            ));
        }

        // Depthwise conv1d
        let conv_out = mlxcel_core::conv1d(
            &padded_input,
            &self.conv_weight,
            1,
            0,
            1,
            self.conv_dim as i32,
        );
        let conv_out = if let Some(ref b) = self.conv_bias {
            let b_reshaped = mlxcel_core::reshape(b, &[1, 1, -1]);
            mlxcel_core::add(&conv_out, &b_reshaped)
        } else {
            conv_out
        };

        let conv_output = silu(&conv_out);

        // Split conv output: hidden_states_ssm, B, C
        let bc_size = (self.n_groups * self.ssm_state_size) as i32;
        let hidden_ssm = slice_axis(&conv_output, -1, 0, self.intermediate_size as i32);
        let b = slice_axis(
            &conv_output,
            -1,
            self.intermediate_size as i32,
            self.intermediate_size as i32 + bc_size,
        );
        let c = slice_axis(
            &conv_output,
            -1,
            self.intermediate_size as i32 + bc_size,
            -1,
        );

        // SSM computation
        let ssm_state = cache.as_ref().and_then(|c| {
            c.ssm_state
                .as_ref()
                .and_then(|s| s.as_ref().map(mlxcel_core::copy))
        });

        // Use fused Metal kernel for single-token decode when state is available
        let (y, new_state) =
            if seq_len == 1 && ssm_state.is_some() && mlxcel_core::ssm_kernel_available() {
                self.ssm_step_kernel(&hidden_ssm, &b, &c, &dt, ssm_state.as_ref().unwrap())
            } else {
                self.ssm_step(&hidden_ssm, &b, &c, &dt, ssm_state.as_deref())
            };

        // Update SSM state
        if let Some(c) = cache {
            c.ssm_state = Some(new_state);
        }

        // Reshape y back to [batch, seq, intermediate_size]
        let y = mlxcel_core::reshape(&y, &[batch, seq_len, self.intermediate_size as i32]);

        // Apply gated norm and output projection
        let y_normed = self.norm.forward(&y, Some(&gate));
        self.out_proj.forward(&y_normed)
    }

    /// SSM computation using the same approach as Mamba2's ssm_attn
    fn ssm_step(
        &self,
        hidden_states: &MlxArray,
        b: &MlxArray,
        c: &MlxArray,
        dt: &MlxArray,
        state: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(hidden_states);
        let batch = shape[0];
        let seq_len = shape[1];

        // Reshape inputs for multi-head processing
        let x = mlxcel_core::reshape(
            hidden_states,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );
        let b = mlxcel_core::reshape(
            b,
            &[
                batch,
                seq_len,
                self.n_groups as i32,
                self.ssm_state_size as i32,
            ],
        );
        let c = mlxcel_core::reshape(
            c,
            &[
                batch,
                seq_len,
                self.n_groups as i32,
                self.ssm_state_size as i32,
            ],
        );

        let num_heads = self.num_heads as i32;
        let n_groups = self.n_groups as i32;
        let state_dim = self.ssm_state_size as i32;
        let head_dim = self.head_dim as i32;
        let repeats = num_heads / n_groups;

        // Compute dt with bias and limits (same as Mamba2's compute_dt)
        let dt_biased = mlxcel_core::add(dt, &self.dt_bias);
        let dt_soft = mlxcel_core::softplus(&dt_biased);
        let min_val =
            mlxcel_core::full_f32(&[1], self.time_step_limit.0, mlxcel_core::dtype::FLOAT32);
        let max_val =
            mlxcel_core::full_f32(&[1], self.time_step_limit.1, mlxcel_core::dtype::FLOAT32);
        let dt = mlxcel_core::clip(&dt_soft, &min_val, &max_val);

        // A = -exp(A_log)
        let a = mlxcel_core::negative(&mlxcel_core::exp(&self.a_log));
        let a_reshaped = mlxcel_core::reshape(&a, &[1, 1, num_heads]);

        // dtA = dt * A
        let dt_a = mlxcel_core::multiply(&dt, &a_reshaped);

        // dtx = dt * x (expand dt for broadcasting)
        let dt_exp = mlxcel_core::reshape(&dt, &[batch, seq_len, num_heads, 1]);
        let dtx = mlxcel_core::multiply(&dt_exp, &x);

        // B: [batch, seq, groups, state_dim] -> [batch, groups, state_dim, seq]
        let b_t = mlxcel_core::transpose_axes(&b, &[0, 2, 3, 1]);

        // CB = C.swapaxes(1, 2) @ B
        let c_t = mlxcel_core::swap_axes(&c, 1, 2);
        let cb = mlxcel_core::matmul(&c_t, &b_t);

        // Repeat CB for each head in group
        let cb = repeat_axis_nemotron(&cb, repeats, 1);

        // decay = exp(segsum(dtA.swapaxes(1, 2)))
        let dt_a_t = mlxcel_core::swap_axes(&dt_a, 1, 2);
        let seg = segsum_nemotron(&dt_a_t);
        let decay = mlxcel_core::exp(&seg);

        // surrogate_attention_matrix = tril(CB * decay)
        let attn_matrix = mlxcel_core::multiply(&cb, &decay);
        let attn_matrix = mlxcel_core::tril(&attn_matrix, 0);

        // y = attn_matrix @ dtx.swapaxes(1, 2)
        let dtx_t = mlxcel_core::swap_axes(&dtx, 1, 2);
        let y = mlxcel_core::matmul(&attn_matrix, &dtx_t);
        let y = mlxcel_core::swap_axes(&y, 1, 2);

        // Compute next state
        // decay_last = decay[:, :, -1:, :]
        let decay_shape = mlxcel_core::array_shape(&decay);
        let decay_last = slice_axis(&decay, 2, decay_shape[2] - 1, decay_shape[2]);
        let decay_t = mlxcel_core::transpose_axes(&decay_last, &[0, 3, 1, 2]);

        // B for state update
        let b_rep = repeat_axis_nemotron(&b_t, repeats, 1);
        let b_sw = mlxcel_core::swap_axes(&b_rep, 2, 3);

        let dtx_decay = mlxcel_core::multiply(&dtx, &decay_t);
        let dtx_decay_t = mlxcel_core::swap_axes(&dtx_decay, 1, 2);
        let dtx_decay_t = mlxcel_core::swap_axes(&dtx_decay_t, 2, 3);

        let mut next_state = mlxcel_core::matmul(&dtx_decay_t, &b_sw);

        // Handle previous state if present (using Mamba2's approach)
        let y = if let Some(prev_state) = state {
            // exp_dtA_cumsum = exp(cumsum(dtA, axis=-2))
            // Python uses inclusive cumsum (exclusive=false)
            let dta_cumsum = mlxcel_core::cumsum(&dt_a, -2, false, false);
            let exp_dta_cumsum = mlxcel_core::exp(&dta_cumsum);

            // Update next_state with previous state
            let exp_shape = mlxcel_core::array_shape(&exp_dta_cumsum);
            let last_exp = slice_axis(&exp_dta_cumsum, 1, exp_shape[1] - 1, exp_shape[1]);

            // Expand for broadcasting with state (Mamba2's approach)
            let mut exp_shape_new = mlxcel_core::array_shape(&last_exp);
            exp_shape_new.push(1);
            exp_shape_new.push(1);
            let last_exp = mlxcel_core::reshape(&last_exp, &exp_shape_new);

            let state_contrib = mlxcel_core::multiply(&last_exp, prev_state);
            next_state = mlxcel_core::add(&next_state, &state_contrib);

            // y_prev contribution
            let c_reshaped = mlxcel_core::reshape(&c, &[batch, seq_len, n_groups, 1, state_dim, 1]);
            let state_reshaped = mlxcel_core::reshape(
                prev_state,
                &[batch, 1, n_groups, repeats, head_dim, state_dim],
            );
            let y_prev = mlxcel_core::matmul(&state_reshaped, &c_reshaped);
            let y_prev = mlxcel_core::squeeze_axis(&y_prev, -1);
            let y_prev = mlxcel_core::reshape(&y_prev, &[batch, seq_len, num_heads, head_dim]);

            // Expand exp_dta_cumsum for y_prev multiplication
            let mut exp_shape_y = mlxcel_core::array_shape(&exp_dta_cumsum);
            exp_shape_y.push(1);
            let exp_dta_exp = mlxcel_core::reshape(&exp_dta_cumsum, &exp_shape_y);

            let y_prev_scaled = mlxcel_core::multiply(&exp_dta_exp, &y_prev);
            mlxcel_core::add(&y, &y_prev_scaled)
        } else {
            y
        };

        // Add D term: y = y + x * D
        let d_reshaped = mlxcel_core::reshape(&self.d_param, &[1, 1, num_heads, 1]);
        let d_contrib = mlxcel_core::multiply(&x, &d_reshaped);
        let y = mlxcel_core::add(&y, &d_contrib);

        (y, next_state)
    }

    /// Fused SSM step using Metal kernel (single-token decode only)
    /// Replaces ~55 individual FFI calls with a single Metal kernel invocation
    fn ssm_step_kernel(
        &self,
        hidden_states: &MlxArray,
        b: &MlxArray,
        c: &MlxArray,
        dt: &MlxArray,
        state: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(hidden_states);
        let batch = shape[0];
        let seq_len = shape[1];

        // Reshape inputs for multi-head processing (matching Python's shapes)
        let x = mlxcel_core::reshape(
            hidden_states,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );
        let b = mlxcel_core::reshape(
            b,
            &[
                batch,
                seq_len,
                self.n_groups as i32,
                self.ssm_state_size as i32,
            ],
        );
        let c = mlxcel_core::reshape(
            c,
            &[
                batch,
                seq_len,
                self.n_groups as i32,
                self.ssm_state_size as i32,
            ],
        );

        let mut output = mlxcel_core::UniquePtr::null();
        let mut next_state = mlxcel_core::UniquePtr::null();

        mlxcel_core::ssm_update_kernel(
            &x,
            &self.a_log,
            &b,
            &c,
            &self.d_param,
            &dt,
            &self.dt_bias,
            state,
            self.time_step_limit.0,
            self.time_step_limit.1,
            &mut output,
            &mut next_state,
        );

        (output, next_state)
    }
}

/// Repeat array along axis by broadcasting (for Nemotron)
fn repeat_axis_nemotron(x: &MlxArray, repeats: i32, axis: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len() as i32;
    let axis = if axis < 0 { ndim + axis } else { axis };

    let mut new_shape: Vec<i32> = shape.iter().take(axis as usize + 1).copied().collect();
    new_shape.push(1);
    new_shape.extend(shape.iter().skip(axis as usize + 1));
    let x_exp = mlxcel_core::reshape(x, &new_shape);

    new_shape[axis as usize + 1] = repeats;
    let x_broad = mlxcel_core::broadcast_to(&x_exp, &new_shape);

    let mut final_shape: Vec<i32> = shape.clone();
    final_shape[axis as usize] *= repeats;
    mlxcel_core::reshape(&x_broad, &final_shape)
}

/// Segmented cumulative sum for SSM attention (for Nemotron)
fn segsum_nemotron(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let l = shape[shape.len() - 1];

    let mut new_shape = shape.clone();
    new_shape.push(1);
    let x_exp = mlxcel_core::reshape(x, &new_shape);

    let last_idx = new_shape.len() - 1;
    new_shape[last_idx] = l;
    let x_rep = mlxcel_core::broadcast_to(&x_exp, &new_shape);

    let x_tril = mlxcel_core::tril(&x_rep, -1);
    // Python uses inclusive cumsum (exclusive=false)
    mlxcel_core::cumsum(&x_tril, -2, false, false)
}

// Attention.
struct NemotronHAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl NemotronHAttention {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: Option<&mut KVCache>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];

        let queries = self.q_proj.forward(x);
        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);

        let queries = mlxcel_core::reshape(
            &queries,
            &[batch, seq_len, self.n_heads as i32, self.head_dim as i32],
        );
        let keys = mlxcel_core::reshape(
            &keys,
            &[batch, seq_len, self.n_kv_heads as i32, self.head_dim as i32],
        );
        let values = mlxcel_core::reshape(
            &values,
            &[batch, seq_len, self.n_kv_heads as i32, self.head_dim as i32],
        );

        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(keys, values)
        } else {
            (keys, values)
        };

        // Repeat KV for GQA
        let n_rep = self.n_heads / self.n_kv_heads;
        let (keys, values) = if n_rep > 1 {
            (
                repeat_kv(&keys, n_rep as i32),
                repeat_kv(&values, n_rep as i32),
            )
        } else {
            (keys, values)
        };

        // Scaled dot-product attention
        let keys_t = mlxcel_core::transpose_axes(&keys, &[0, 1, 3, 2]);
        let mut scores = mlxcel_core::matmul(&queries, &keys_t);
        let scale_arr = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::dtype::FLOAT32);
        scores = mlxcel_core::multiply(&scores, &scale_arr);

        if let Some(m) = mask {
            scores = mlxcel_core::add(&scores, m);
        }

        let weights = mlxcel_core::softmax(&scores, -1);
        let output = mlxcel_core::matmul(&weights, &values);

        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[batch, seq_len, -1]);

        self.o_proj.forward(&output)
    }
}

// MLP with relu^2 activation.
struct NemotronHMLP {
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl NemotronHMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let up = self.up_proj.forward(x);
        let relu_sq = relu_squared(&up);
        self.down_proj.forward(&relu_sq)
    }
}

/// Helper to extract raw weight/scales/biases pointers from UnifiedLinear
/// Returns null pointers for non-quantized linear layers
trait QuantizedWeightPtrs {
    fn weight_ptr(&self) -> *const MlxArray;
    fn scales_ptr(&self) -> *const MlxArray;
    fn biases_ptr(&self) -> *const MlxArray;
}

impl QuantizedWeightPtrs for UnifiedLinear {
    fn weight_ptr(&self) -> *const MlxArray {
        match self {
            UnifiedLinear::Quantized { weight, .. } => {
                weight.weight.as_ref().unwrap() as *const MlxArray
            }
            UnifiedLinear::Regular(l) => l.weight.as_ref().unwrap() as *const MlxArray,
        }
    }
    fn scales_ptr(&self) -> *const MlxArray {
        match self {
            UnifiedLinear::Quantized { weight, .. } => {
                weight.scales.as_ref().unwrap() as *const MlxArray
            }
            _ => std::ptr::null(),
        }
    }
    fn biases_ptr(&self) -> *const MlxArray {
        match self {
            UnifiedLinear::Quantized { weight, .. } => weight.biases_ptr(),
            _ => std::ptr::null(),
        }
    }
}

// QuantizedSwitchLinear and SwitchMLP for MoE (using gather_qmm).
// Kept local: uses weights.remove() (ownership), different naming (QuantizedSwitchLinear)
enum QuantizedSwitchLinear {
    Quantized {
        weight: UniquePtr<MlxArray>, // [num_experts, out_features, packed_in_features]
        scales: UniquePtr<MlxArray>, // [num_experts, out_features, groups]
        biases: UniquePtr<MlxArray>, // [num_experts, out_features, groups]
        group_size: i32,
        bits: i32,
    },
    Regular {
        weight: UniquePtr<MlxArray>, // [num_experts, out_features, in_features]
    },
}

impl QuantizedSwitchLinear {
    fn forward(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
        sorted_indices: bool,
    ) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                let biases_ptr = biases.as_ref().unwrap() as *const MlxArray;
                let indices_ptr = indices as *const MlxArray;
                unsafe {
                    mlxcel_core::gather_qmm(
                        x,
                        weight,
                        scales,
                        biases_ptr,
                        std::ptr::null(),
                        indices_ptr,
                        true,
                        *group_size,
                        *bits,
                        sorted_indices,
                        "affine",
                    )
                }
            }
            Self::Regular { weight } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                unsafe {
                    mlxcel_core::gather_mm(
                        x,
                        &wt,
                        std::ptr::null(),
                        indices as *const _,
                        sorted_indices,
                    )
                }
            }
        }
    }
}

struct SwitchMLP {
    fc1: QuantizedSwitchLinear,
    fc2: QuantizedSwitchLinear,
}

impl SwitchMLP {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        // x: [tokens, hidden], indices: [tokens, top_k]
        // Expand x to [tokens, 1, 1, hidden] for gather_qmm
        let x_shape = mlxcel_core::array_shape(x);
        let x_exp = mlxcel_core::reshape(x, &[x_shape[0], 1, 1, x_shape[1]]);

        // Note: sorted_indices optimization skipped for simplicity
        let sorted = false;

        let h = self.fc1.forward(&x_exp, indices, sorted);
        let h = relu_squared(&h);
        let out = self.fc2.forward(&h, indices, sorted);

        // Squeeze middle dimension: [tokens, top_k, 1, hidden] -> [tokens, top_k, hidden]
        mlxcel_core::squeeze_axis(&out, -2)
    }
}

// MoE Gate and Block.
struct NemotronHMoEGate {
    weight: UniquePtr<MlxArray>,
    e_score_correction_bias: UniquePtr<MlxArray>,
    top_k: usize,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
}

impl NemotronHMoEGate {
    fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // Gate projection: x @ W^T
        let w_t = mlxcel_core::transpose(&self.weight);
        let gates = mlxcel_core::matmul(x, &w_t);

        // Use fused C++ gate function (matches Python @mx.compile group_expert_select)
        let mut indices = mlxcel_core::UniquePtr::null();
        let mut scores = mlxcel_core::UniquePtr::null();
        mlxcel_core::compiled_moe_gate(
            &gates,
            &self.e_score_correction_bias,
            self.top_k as i32,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            &mut indices,
            &mut scores,
        );

        (indices, scores)
    }
}

struct NemotronHMoE {
    gate: NemotronHMoEGate,
    switch_mlp: SwitchMLP,
    shared_experts: Option<NemotronHMLP>,
}

impl NemotronHMoE {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);

        // Flatten to [n_tokens, hidden]
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, orig_shape[orig_shape.len() - 1]])
        } else {
            mlxcel_core::copy(x)
        };

        // Try fused MoE forward (quantized path only)
        let result = if let (
            QuantizedSwitchLinear::Quantized {
                weight: fc1_w,
                scales: fc1_s,
                biases: fc1_b,
                group_size,
                bits,
            },
            QuantizedSwitchLinear::Quantized {
                weight: fc2_w,
                scales: fc2_s,
                biases: fc2_b,
                ..
            },
        ) = (&self.switch_mlp.fc1, &self.switch_mlp.fc2)
        {
            // Get shared expert weight pointers (nullable)
            let (sup_w, sup_s, sup_b, sdn_w, sdn_s, sdn_b) =
                if let Some(ref shared) = self.shared_experts {
                    (
                        shared.up_proj.weight_ptr(),
                        shared.up_proj.scales_ptr(),
                        shared.up_proj.biases_ptr(),
                        shared.down_proj.weight_ptr(),
                        shared.down_proj.scales_ptr(),
                        shared.down_proj.biases_ptr(),
                    )
                } else {
                    (
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                    )
                };

            unsafe {
                mlxcel_core::fused_moe_forward(
                    &x_flat,
                    &self.gate.weight,
                    &self.gate.e_score_correction_bias,
                    fc1_w,
                    fc1_s,
                    fc1_b,
                    fc2_w,
                    fc2_s,
                    fc2_b,
                    sup_w,
                    sup_s,
                    sup_b,
                    sdn_w,
                    sdn_s,
                    sdn_b,
                    self.gate.top_k as i32,
                    self.gate.routed_scaling_factor,
                    self.gate.norm_topk_prob,
                    *group_size,
                    *bits,
                )
            }
        } else {
            // Fallback: non-quantized path (individual calls)
            let (indices, scores) = self.gate.forward(&x_flat);
            let y = self.switch_mlp.forward(&x_flat, &indices);
            let mut scores_shape = mlxcel_core::array_shape(&scores);
            scores_shape.push(1);
            let scores_exp = mlxcel_core::reshape(&scores, &scores_shape);
            let weighted = mlxcel_core::multiply(&y, &scores_exp);
            let mut result = mlxcel_core::sum_axis(&weighted, -2, false);
            if let Some(ref shared) = self.shared_experts {
                let shared_out = shared.forward(&x_flat);
                result = mlxcel_core::add(&result, &shared_out);
            }
            result
        };

        // Reshape back
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }
}

// Hybrid Block.
enum NemotronHMixer {
    Mamba(NemotronHMamba2Mixer),
    Attention(NemotronHAttention),
    MLP(NemotronHMLP),
    MoE(NemotronHMoE),
}

#[allow(dead_code)]
struct NemotronHBlock {
    norm: RMSNorm,
    block_type: BlockType,
    mixer: NemotronHMixer,
}

impl NemotronHBlock {
    fn forward(
        &self,
        x: &MlxArray,
        attn_mask: Option<&MlxArray>,
        mamba_cache: Option<&mut NemotronMambaCache>,
        kv_cache: Option<&mut KVCache>,
    ) -> UniquePtr<MlxArray> {
        let h = self.norm.forward(x);

        let out = match &self.mixer {
            NemotronHMixer::Mamba(m) => m.forward(&h, mamba_cache),
            NemotronHMixer::Attention(a) => a.forward(&h, attn_mask, kv_cache),
            NemotronHMixer::MLP(m) => m.forward(&h),
            NemotronHMixer::MoE(m) => m.forward(&h),
        };

        mlxcel_core::add(x, &out)
    }
}

// Full Model.
use std::cell::RefCell;

pub struct NemotronHModel {
    config: NemotronHConfig,
    embeddings: UnifiedEmbedding,
    layers: Vec<NemotronHBlock>,
    norm_f: RMSNorm,
    lm_head: UnifiedLinear,
    block_types: Vec<BlockType>,
    /// Internal caches for LanguageModel trait compatibility
    internal_caches: RefCell<Vec<NemotronLayerCache>>,
}

impl NemotronHModel {
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn make_caches(&self) -> Vec<NemotronLayerCache> {
        self.block_types
            .iter()
            .filter(|bt| bt.needs_cache())
            .map(|bt| match bt {
                BlockType::Mamba => NemotronLayerCache::Mamba(NemotronMambaCache::new()),
                BlockType::Attention => NemotronLayerCache::Attention(KVCache::new()),
                _ => unreachable!(),
            })
            .collect()
    }

    pub fn forward_with_caches(
        &self,
        inputs: &MlxArray,
        caches: &mut [NemotronLayerCache],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embeddings.forward(inputs);

        // Find first attention cache offset
        let attn_offset = caches
            .iter()
            .find(|c| matches!(c, NemotronLayerCache::Attention(_)))
            .map(|c| c.offset())
            .unwrap_or(0);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];
        let attn_mask = if seq_len > 1 {
            Some(create_causal_mask(seq_len, attn_offset))
        } else {
            None
        };

        let mut cache_idx = 0;
        for (layer, &block_type) in self.layers.iter().zip(self.block_types.iter()) {
            if block_type.needs_cache() {
                let cache_entry = &mut caches[cache_idx];
                match (block_type, cache_entry) {
                    (BlockType::Mamba, NemotronLayerCache::Mamba(mc)) => {
                        h = layer.forward(&h, None, Some(mc), None);
                    }
                    (BlockType::Attention, NemotronLayerCache::Attention(kv)) => {
                        h = layer.forward(&h, attn_mask.as_deref(), None, Some(kv));
                    }
                    _ => {}
                }
                cache_idx += 1;
            } else {
                h = layer.forward(&h, None, None, None);
            }
        }

        let h = self.norm_f.forward(&h);
        self.lm_head.forward(&h)
    }

    pub fn load(model_path: &str) -> Result<(Self, NemotronHConfig), Box<dyn std::error::Error>> {
        let path = Path::new(model_path);

        println!("[NemotronH] Loading config...");
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_str = super::sanitize_config_json(&config_str);
        let config: NemotronHConfig = serde_json::from_str(&config_str)?;

        let block_types: Vec<BlockType> = config
            .hybrid_override_pattern
            .iter()
            .map(|s| BlockType::from_str(s))
            .collect();

        println!(
            "[NemotronH] Config loaded: {} layers ({} mamba, {} attention, {} mlp, {} moe)",
            config.num_hidden_layers,
            block_types
                .iter()
                .filter(|t| **t == BlockType::Mamba)
                .count(),
            block_types
                .iter()
                .filter(|t| **t == BlockType::Attention)
                .count(),
            block_types.iter().filter(|t| **t == BlockType::MLP).count(),
            block_types.iter().filter(|t| **t == BlockType::MoE).count()
        );

        println!("[NemotronH] Loading weights from safetensors...");
        let weights = crate::models::load_and_sanitize_weights(path)?;
        let weights = Self::sanitize_weights(weights, &config);

        println!("[NemotronH] Building model...");
        let model = Self::from_weights(config.clone(), weights, block_types)?;

        println!("[NemotronH] Model loaded successfully");
        Ok((model, config))
    }

    pub fn sanitize_weights(mut weights: WeightMap, config: &NemotronHConfig) -> WeightMap {
        // Handle conv1d weight transpose
        let keys: Vec<String> = weights.keys().cloned().collect();
        for k in keys {
            if k.contains("conv1d.weight")
                && let Some(v) = weights.get(&k)
            {
                let shape = mlxcel_core::array_shape(v);
                if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                    let transposed = mlxcel_core::swap_axes(v, -1, -2);
                    weights.insert(k, transposed);
                }
            }
        }

        // Stack expert weights
        let n_routed = config.n_routed_experts.unwrap_or(0);
        if n_routed > 0 {
            for l in 0..config.num_hidden_layers {
                let prefix = format!("backbone.layers.{}.mixer", l);

                for (m, n) in [("down_proj", "fc2"), ("up_proj", "fc1")] {
                    let first_expert_key = format!("{}.experts.0.{}.weight", prefix, m);
                    if weights.contains_key(&first_expert_key) {
                        let mut expert_tensors: Vec<UniquePtr<MlxArray>> = Vec::new();
                        for e in 0..n_routed {
                            if let Some(w) =
                                weights.remove(&format!("{}.experts.{}.{}.weight", prefix, e, m))
                            {
                                expert_tensors.push(w);
                            }
                        }
                        if !expert_tensors.is_empty() {
                            let stacked = stack_arrays(&expert_tensors, 0);
                            weights.insert(format!("{}.switch_mlp.{}.weight", prefix, n), stacked);
                        }
                    }
                }
            }
        }

        weights
    }

    pub fn from_weights(
        config: NemotronHConfig,
        mut weights: WeightMap,
        block_types: Vec<BlockType>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Get quantization parameters
        let group_size = config.group_size();
        let bits = config.bits();

        // Quantized Embeddings
        let embeddings =
            UnifiedEmbedding::from_weights(&weights, "backbone.embeddings", group_size, bits)?;

        // Build layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (i, &block_type) in block_types.iter().enumerate() {
            let prefix = format!("backbone.layers.{}", i);

            // Norm
            let norm_weight = weights
                .remove(&format!("{}.norm.weight", prefix))
                .ok_or(format!("Missing norm weight for layer {}", i))?;
            let norm = RMSNorm::new(norm_weight, config.layer_norm_epsilon);

            // Mixer based on block type
            let mixer = match block_type {
                BlockType::Mamba => {
                    let mixer_prefix = format!("{}.mixer", prefix);
                    let intermediate_size = config.get_mamba_intermediate_size();
                    let conv_dim = config.get_conv_dim();
                    let mamba_group_size = intermediate_size / config.n_groups;

                    let conv_weight = weights
                        .remove(&format!("{}.conv1d.weight", mixer_prefix))
                        .ok_or(format!("Missing conv1d weight for layer {}", i))?;
                    let conv_bias = weights.remove(&format!("{}.conv1d.bias", mixer_prefix));

                    let in_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.in_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;
                    let out_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.out_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;

                    let dt_bias = weights
                        .remove(&format!("{}.dt_bias", mixer_prefix))
                        .ok_or(format!("Missing dt_bias for layer {}", i))?;
                    let a_log = weights
                        .remove(&format!("{}.A_log", mixer_prefix))
                        .ok_or(format!("Missing A_log for layer {}", i))?;
                    let d_param = weights
                        .remove(&format!("{}.D", mixer_prefix))
                        .ok_or(format!("Missing D for layer {}", i))?;

                    let norm_weight = weights
                        .remove(&format!("{}.norm.weight", mixer_prefix))
                        .ok_or(format!("Missing mixer norm weight for layer {}", i))?;

                    NemotronHMixer::Mamba(NemotronHMamba2Mixer {
                        num_heads: config.mamba_num_heads,
                        hidden_size: config.hidden_size,
                        ssm_state_size: config.ssm_state_size,
                        conv_kernel_size: config.conv_kernel,
                        intermediate_size,
                        n_groups: config.n_groups,
                        head_dim: config.mamba_head_dim,
                        time_step_limit: config.time_step_limit,
                        conv_dim,
                        conv_weight,
                        conv_bias,
                        in_proj,
                        dt_bias,
                        a_log,
                        d_param,
                        norm: MambaRMSNormGated {
                            weight: norm_weight,
                            eps: config.layer_norm_epsilon,
                            group_size: mamba_group_size,
                        },
                        out_proj,
                    })
                }
                BlockType::Attention => {
                    let mixer_prefix = format!("{}.mixer", prefix);
                    let head_dim = config.get_head_dim();

                    let q_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.q_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;
                    let k_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.k_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;
                    let v_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.v_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;
                    let o_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.o_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;

                    NemotronHMixer::Attention(NemotronHAttention {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        n_heads: config.num_attention_heads,
                        n_kv_heads: config.num_key_value_heads,
                        head_dim,
                        scale: (head_dim as f32).powf(-0.5),
                    })
                }
                BlockType::MLP => {
                    let mixer_prefix = format!("{}.mixer", prefix);
                    let up_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.up_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;
                    let down_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.down_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;

                    NemotronHMixer::MLP(NemotronHMLP { up_proj, down_proj })
                }
                BlockType::MoE => {
                    let mixer_prefix = format!("{}.mixer", prefix);
                    let n_routed = config.n_routed_experts.unwrap_or(1);
                    let _moe_hidden = config
                        .moe_intermediate_size
                        .unwrap_or(config.intermediate_size);
                    let top_k = config.num_experts_per_tok.unwrap_or(1);

                    // Gate
                    let gate_weight = weights
                        .remove(&format!("{}.gate.weight", mixer_prefix))
                        .ok_or(format!("Missing gate weight for layer {}", i))?;
                    let e_score_bias = weights
                        .remove(&format!("{}.gate.e_score_correction_bias", mixer_prefix))
                        .unwrap_or_else(|| {
                            mlxcel_core::zeros(&[n_routed as i32], mlxcel_core::dtype::FLOAT32)
                        });

                    // Switch MLP (quantized or regular)
                    let fc1_weight = weights
                        .remove(&format!("{}.switch_mlp.fc1.weight", mixer_prefix))
                        .ok_or(format!("Missing switch_mlp.fc1.weight for layer {}", i))?;
                    let fc1_scales =
                        weights.remove(&format!("{}.switch_mlp.fc1.scales", mixer_prefix));
                    let fc1_biases =
                        weights.remove(&format!("{}.switch_mlp.fc1.biases", mixer_prefix));
                    let fc2_weight = weights
                        .remove(&format!("{}.switch_mlp.fc2.weight", mixer_prefix))
                        .ok_or(format!("Missing switch_mlp.fc2.weight for layer {}", i))?;
                    let fc2_scales =
                        weights.remove(&format!("{}.switch_mlp.fc2.scales", mixer_prefix));
                    let fc2_biases =
                        weights.remove(&format!("{}.switch_mlp.fc2.biases", mixer_prefix));

                    // Shared experts (optional, quantized)
                    let shared_experts = if config.n_shared_experts.is_some() {
                        let up_proj = UnifiedLinear::from_weights(
                            &weights,
                            &format!("{}.shared_experts.up_proj", mixer_prefix),
                            group_size,
                            bits,
                        )
                        .ok();
                        let down_proj = UnifiedLinear::from_weights(
                            &weights,
                            &format!("{}.shared_experts.down_proj", mixer_prefix),
                            group_size,
                            bits,
                        )
                        .ok();

                        if let (Some(up), Some(down)) = (up_proj, down_proj) {
                            Some(NemotronHMLP {
                                up_proj: up,
                                down_proj: down,
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    NemotronHMixer::MoE(NemotronHMoE {
                        gate: NemotronHMoEGate {
                            weight: gate_weight,
                            e_score_correction_bias: e_score_bias,
                            top_k,
                            routed_scaling_factor: config.routed_scaling_factor.unwrap_or(1.0),
                            norm_topk_prob: config.norm_topk_prob.unwrap_or(false),
                        },
                        switch_mlp: SwitchMLP {
                            fc1: if let Some(scales) = fc1_scales {
                                QuantizedSwitchLinear::Quantized {
                                    weight: fc1_weight,
                                    scales,
                                    biases: fc1_biases.ok_or(format!(
                                        "Missing switch_mlp.fc1.biases for layer {}",
                                        i
                                    ))?,
                                    group_size,
                                    bits,
                                }
                            } else {
                                QuantizedSwitchLinear::Regular { weight: fc1_weight }
                            },
                            fc2: if let Some(scales) = fc2_scales {
                                QuantizedSwitchLinear::Quantized {
                                    weight: fc2_weight,
                                    scales,
                                    biases: fc2_biases.ok_or(format!(
                                        "Missing switch_mlp.fc2.biases for layer {}",
                                        i
                                    ))?,
                                    group_size,
                                    bits,
                                }
                            } else {
                                QuantizedSwitchLinear::Regular { weight: fc2_weight }
                            },
                        },
                        shared_experts,
                    })
                }
            };

            layers.push(NemotronHBlock {
                norm,
                block_type,
                mixer,
            });
        }

        // Final norm
        let norm_f_weight = weights
            .remove("backbone.norm_f.weight")
            .ok_or("Missing final norm weight")?;
        let norm_f = RMSNorm::new(norm_f_weight, config.layer_norm_epsilon);

        // LM head (quantized)
        let lm_head = UnifiedLinear::from_weights(&weights, "lm_head", group_size, bits)?;

        // Create internal caches for LanguageModel trait compatibility
        let internal_caches: Vec<NemotronLayerCache> = block_types
            .iter()
            .filter(|bt| bt.needs_cache())
            .map(|bt| match bt {
                BlockType::Mamba => NemotronLayerCache::Mamba(NemotronMambaCache::new()),
                BlockType::Attention => NemotronLayerCache::Attention(KVCache::new()),
                _ => unreachable!(),
            })
            .collect();

        Ok(Self {
            config,
            embeddings,
            layers,
            norm_f,
            lm_head,
            block_types,
            internal_caches: RefCell::new(internal_caches),
        })
    }
}

#[allow(dead_code)]
fn load_linear(
    weights: &mut WeightMap,
    prefix: &str,
) -> Result<Linear, Box<dyn std::error::Error>> {
    let weight = weights
        .remove(&format!("{}.weight", prefix))
        .ok_or(format!("Missing weight for {}", prefix))?;
    let bias = weights.remove(&format!("{}.bias", prefix));
    Ok(Linear::new(weight, bias))
}

// LanguageModel trait implementation.
impl LanguageModel for NemotronHModel {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // NemotronH uses mixed cache types (KVCache + MambaCache)
        // We use internal RefCell caches to maintain state through shared reference
        let mut internal = self.internal_caches.borrow_mut();
        self.forward_with_caches(input, &mut internal)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Reset internal caches for new generation session
        let new_internal_caches: Vec<NemotronLayerCache> = self
            .block_types
            .iter()
            .filter(|bt| bt.needs_cache())
            .map(|bt| match bt {
                BlockType::Mamba => NemotronLayerCache::Mamba(NemotronMambaCache::new()),
                BlockType::Attention => NemotronLayerCache::Attention(KVCache::new()),
                _ => unreachable!(),
            })
            .collect();
        *self.internal_caches.borrow_mut() = new_internal_caches;

        // Return empty KV caches for trait compatibility
        // Actual caching uses internal_caches
        (0..self
            .block_types
            .iter()
            .filter(|bt| bt.needs_cache())
            .count())
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn supports_batching(&self) -> bool {
        false // NemotronH is a hybrid Mamba+Transformer, internal caches not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![11] // <|im_end|>
    }
}
