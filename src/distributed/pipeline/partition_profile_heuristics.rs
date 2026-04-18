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

//! Per-model-family heuristics for the pipeline [`ModelProfile`] builder.
//!
//! Each model family either keeps the uniform dense-layer cost (default
//! `match` arm) or overrides it to account for MoE expert weights,
//! Gemma 4's double-wide MLP on KV-shared consumers, or similar layer
//! heterogeneity. See [`super::partition_profile::build_profile_from_json`]
//! for the entry point.
//!
//! Used by: `super::partition_profile::build_per_layer_bytes`,
//! `super::partition_profile::build_adjacency`

use serde_json::Value;

use super::partition::LayerAdjacencyGroup;

/// Populate the per-layer byte vector, accounting for MoE expert layers
/// and Gemma 4's double-wide MLP on KV-shared consumers.
pub(super) fn build_per_layer_bytes(
    root: &Value,
    text: Option<&Value>,
    model_type: &str,
    num_layers: usize,
    dense_layer_bytes: u64,
    hidden_size: u64,
    bits_per_weight: u64,
) -> Vec<u64> {
    let mut out = vec![dense_layer_bytes; num_layers];

    match model_type {
        "mixtral" | "qwen3_5_moe" | "qwen3_5_moe_vlm" | "exaone_moe" | "gpt_oss" | "glm4_moe"
        | "glm4_moe_lite" | "glm_moe_dsa" | "phi_moe" => {
            let experts = expert_count(text, root);
            let moe_intermediate = moe_intermediate_size(text, root).unwrap_or_else(|| {
                text.and_then(|t| t.get("intermediate_size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(hidden_size * 4)
            });
            let expert_block =
                moe_expert_block_bytes(hidden_size, moe_intermediate, experts, bits_per_weight);
            let mlp_only = dense_mlp_bytes(hidden_size, moe_intermediate, bits_per_weight);
            let moe_layer_bytes = dense_layer_bytes
                .saturating_sub(mlp_only)
                .saturating_add(expert_block);
            for slot in out.iter_mut() {
                *slot = moe_layer_bytes;
            }
        }
        "deepseek_v3" | "deepseek_v2" | "deepseek" | "nemotron_h" => {
            let experts = expert_count(text, root);
            let first_dense = text
                .and_then(|t| t.get("first_k_dense_replace"))
                .and_then(|v| v.as_u64())
                .or_else(|| root.get("first_k_dense_replace").and_then(|v| v.as_u64()))
                .unwrap_or(0) as usize;
            let moe_freq = text
                .and_then(|t| t.get("moe_layer_freq"))
                .and_then(|v| v.as_u64())
                .or_else(|| root.get("moe_layer_freq").and_then(|v| v.as_u64()))
                .unwrap_or(1) as usize;
            let moe_intermediate = moe_intermediate_size(text, root).unwrap_or_else(|| {
                text.and_then(|t| t.get("intermediate_size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(hidden_size * 4)
            });
            let expert_block =
                moe_expert_block_bytes(hidden_size, moe_intermediate, experts, bits_per_weight);
            let mlp_only = dense_mlp_bytes(hidden_size, moe_intermediate, bits_per_weight);
            for (i, slot) in out.iter_mut().enumerate() {
                let is_moe =
                    experts > 0 && i >= first_dense && moe_freq > 0 && i.is_multiple_of(moe_freq);
                if is_moe {
                    *slot = dense_layer_bytes
                        .saturating_sub(mlp_only)
                        .saturating_add(expert_block);
                }
            }
        }
        "llama4" | "llama4_vlm" => {
            let experts = expert_count(text, root);
            let step = text
                .and_then(|t| t.get("interleave_moe_layer_step"))
                .and_then(|v| v.as_u64())
                .or_else(|| {
                    root.get("interleave_moe_layer_step")
                        .and_then(|v| v.as_u64())
                })
                .unwrap_or(1) as usize;
            let moe_intermediate = text
                .and_then(|t| t.get("intermediate_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(hidden_size * 4);
            let dense_intermediate = text
                .and_then(|t| t.get("intermediate_size_mlp"))
                .and_then(|v| v.as_u64())
                .unwrap_or(moe_intermediate);
            let expert_block =
                moe_expert_block_bytes(hidden_size, moe_intermediate, experts, bits_per_weight);
            let dense_mlp_only = dense_mlp_bytes(hidden_size, dense_intermediate, bits_per_weight);
            let moe_mlp_only = dense_mlp_bytes(hidden_size, moe_intermediate, bits_per_weight);
            for (i, slot) in out.iter_mut().enumerate() {
                let is_moe_step = step > 0 && (i % step) == (step - 1);
                if is_moe_step {
                    *slot = dense_layer_bytes
                        .saturating_sub(moe_mlp_only)
                        .saturating_add(expert_block);
                } else {
                    *slot = dense_layer_bytes
                        .saturating_sub(moe_mlp_only)
                        .saturating_add(dense_mlp_only);
                }
            }
        }
        "jamba" => {
            let experts = expert_count(text, root);
            let expert_period = text
                .and_then(|t| t.get("expert_layer_period"))
                .and_then(|v| v.as_u64())
                .or_else(|| root.get("expert_layer_period").and_then(|v| v.as_u64()))
                .unwrap_or(2) as usize;
            let expert_offset = text
                .and_then(|t| t.get("expert_layer_offset"))
                .and_then(|v| v.as_u64())
                .or_else(|| root.get("expert_layer_offset").and_then(|v| v.as_u64()))
                .unwrap_or(0) as usize;
            let moe_intermediate = text
                .and_then(|t| t.get("intermediate_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(hidden_size * 4);
            let expert_block =
                moe_expert_block_bytes(hidden_size, moe_intermediate, experts, bits_per_weight);
            let mlp_only = dense_mlp_bytes(hidden_size, moe_intermediate, bits_per_weight);
            for (i, slot) in out.iter_mut().enumerate() {
                let is_moe = experts > 0
                    && expert_period > 0
                    && i >= expert_offset
                    && (i - expert_offset).is_multiple_of(expert_period);
                if is_moe {
                    *slot = dense_layer_bytes
                        .saturating_sub(mlp_only)
                        .saturating_add(expert_block);
                }
            }
        }
        "gemma4" | "gemma4_vlm" | "gemma3" | "gemma3_text" => {
            let num_shared = text
                .and_then(|t| t.get("num_kv_shared_layers"))
                .and_then(|v| v.as_u64())
                .or_else(|| root.get("num_kv_shared_layers").and_then(|v| v.as_u64()))
                .unwrap_or(0) as usize;
            let use_double = text
                .and_then(|t| t.get("use_double_wide_mlp"))
                .and_then(|v| v.as_bool())
                .or_else(|| root.get("use_double_wide_mlp").and_then(|v| v.as_bool()))
                .unwrap_or(false);
            if use_double && num_shared > 0 {
                let intermediate = text
                    .and_then(|t| t.get("intermediate_size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(hidden_size * 4);
                let mlp_base = dense_mlp_bytes(hidden_size, intermediate, bits_per_weight);
                let mlp_double =
                    dense_mlp_bytes(hidden_size, intermediate.saturating_mul(2), bits_per_weight);
                let first_shared = num_layers.saturating_sub(num_shared);
                for (i, slot) in out.iter_mut().enumerate() {
                    if i >= first_shared {
                        *slot = dense_layer_bytes
                            .saturating_sub(mlp_base)
                            .saturating_add(mlp_double);
                    }
                }
            }
        }
        _ => {}
    }

    out
}

/// Build the adjacency constraints implied by the model type.
///
/// Currently Gemma 4 is the only supported model with a mandatory adjacency
/// invariant: each of its `num_kv_shared_layers` consumers reads its keys
/// and values from the most recent earlier layer with the same
/// `layer_types[i]` value.
pub(super) fn build_adjacency(
    root: &Value,
    text: Option<&Value>,
    model_type: &str,
    num_layers: usize,
) -> Vec<LayerAdjacencyGroup> {
    let mut out = Vec::new();
    if !(model_type == "gemma4" || model_type == "gemma4_vlm") {
        return out;
    }
    let num_shared = text
        .and_then(|t| t.get("num_kv_shared_layers"))
        .and_then(|v| v.as_u64())
        .or_else(|| root.get("num_kv_shared_layers").and_then(|v| v.as_u64()))
        .unwrap_or(0) as usize;
    if num_shared == 0 || num_layers == 0 {
        return out;
    }
    let layer_types: Vec<String> = text
        .and_then(|t| t.get("layer_types"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default();
    if layer_types.len() < num_layers {
        return out;
    }
    let first_shared = num_layers.saturating_sub(num_shared);
    for consumer in first_shared..num_layers {
        let ltype = &layer_types[consumer];
        if let Some(source_rel) = layer_types[..first_shared].iter().rposition(|t| t == ltype) {
            let source = source_rel;
            out.push(LayerAdjacencyGroup {
                layers: source..(consumer + 1),
                reason: format!(
                    "gemma4 KV-shared layer {} reads keys/values from layer {}",
                    consumer, source
                ),
            });
        }
    }
    out
}

pub(super) fn expert_count(text: Option<&Value>, root: &Value) -> u64 {
    for key in [
        "num_local_experts",
        "n_routed_experts",
        "num_experts",
        "num_routed_experts",
    ] {
        if let Some(v) = text
            .and_then(|t| t.get(key))
            .and_then(|v| v.as_u64())
            .or_else(|| root.get(key).and_then(|v| v.as_u64()))
        {
            return v;
        }
    }
    0
}

pub(super) fn moe_intermediate_size(text: Option<&Value>, root: &Value) -> Option<u64> {
    for key in ["moe_intermediate_size", "expert_intermediate_size"] {
        if let Some(v) = text
            .and_then(|t| t.get(key))
            .and_then(|v| v.as_u64())
            .or_else(|| root.get(key).and_then(|v| v.as_u64()))
        {
            return Some(v);
        }
    }
    None
}

pub(super) fn moe_expert_block_bytes(
    hidden_size: u64,
    intermediate: u64,
    experts: u64,
    bits_per_weight: u64,
) -> u64 {
    let per_expert = hidden_size.saturating_mul(intermediate).saturating_mul(3);
    super::partition_profile::param_bytes(per_expert.saturating_mul(experts), bits_per_weight)
}

pub(super) fn dense_mlp_bytes(hidden_size: u64, intermediate: u64, bits_per_weight: u64) -> u64 {
    super::partition_profile::param_bytes(
        hidden_size.saturating_mul(intermediate).saturating_mul(3),
        bits_per_weight,
    )
}
