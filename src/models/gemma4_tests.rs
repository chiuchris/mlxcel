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

//! Unit tests for Gemma 4 configuration parsing and RoPE handling.
//!
//! These tests lock in the behavior described in GitHub issue #321:
//!
//! 1. Real Gemma 4 checkpoints declare `rope_type: "proportional"` on every
//!    `full_attention` layer and `rope_type: "default"` on every
//!    `sliding_attention` layer.
//! 2. Under `rope_type: "proportional"`, RoPE exponents MUST be normalized by
//!    the full head dimension (not the rotated-only slice) — matching the
//!    upstream `mlx_vlm.models.gemma4.rope_utils.ProportionalRoPE` semantics.
//! 3. Under `rope_type: "default"` (sliding-attention layers), mlxcel keeps
//!    the historical `nn.RoPE(head_dim * partial_rotary_factor)` path.

use super::gemma4::{RopeParameters, TextConfig};

fn parse_text_config(json: serde_json::Value) -> TextConfig {
    serde_json::from_value(json).expect("TextConfig must deserialize")
}

/// Minimal text_config mirroring the Gemma 4 E2B real checkpoint
/// (trimmed to fields relevant for RoPE / layer-type dispatch).
fn real_gemma4_e2b_text_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma4_text",
        "hidden_size": 1536,
        "num_hidden_layers": 35,
        "intermediate_size": 6144,
        "num_attention_heads": 8,
        "head_dim": 256,
        "global_head_dim": 512,
        "rms_norm_eps": 1e-6,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 20,
        "hidden_size_per_layer_input": 256,
        "sliding_window": 512,
        "max_position_embeddings": 131072,
        "use_double_wide_mlp": true,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional"
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default"
            }
        },
        // Real layer pattern from the checkpoint — 4 sliding then 1 full, repeated.
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention"
        ]
    })
}

#[test]
fn gemma4_config_parses_real_checkpoint_rope_parameters() {
    // The primary regression target for issue #321: make sure we can in fact
    // read `rope_type` out of the real checkpoint config without erroring,
    // and that both per-layer-type entries deserialize correctly.
    let cfg = parse_text_config(real_gemma4_e2b_text_config());

    let full = cfg
        .rope_parameters
        .get("full_attention")
        .expect("full_attention rope params must be present");
    assert_eq!(full.rope_type, "proportional");
    assert!((full.partial_rotary_factor - 0.25).abs() < 1e-6);
    assert!((full.rope_theta - 1_000_000.0).abs() < 1e-3);

    let sliding = cfg
        .rope_parameters
        .get("sliding_attention")
        .expect("sliding_attention rope params must be present");
    assert_eq!(sliding.rope_type, "default");
    // Sliding entries in the real checkpoint omit `partial_rotary_factor`;
    // the serde default should be 1.0.
    assert!((sliding.partial_rotary_factor - 1.0).abs() < 1e-6);
    assert!((sliding.rope_theta - 10_000.0).abs() < 1e-3);
}

#[test]
fn gemma4_rope_parameters_rope_type_defaults_when_absent() {
    // Older / simpler configs that omit `rope_type` entirely must still
    // deserialize and default to "default".
    let params: RopeParameters = serde_json::from_value(serde_json::json!({
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0
    }))
    .expect("RopeParameters must deserialize without rope_type");
    assert_eq!(params.rope_type, "default");
}

#[test]
fn gemma4_proportional_rope_freqs_match_python_semantics() {
    // Lock in the numerical semantics of issue #321 Case A:
    //
    //   freqs[i] = base^(2 * i / head_dim)   for i in [0, rope_angles)
    //
    // with rope_angles = int(partial_rotary_factor * head_dim / 2), followed
    // by an `inf` tail that disables rotation for the remaining pairs. The
    // denominator is the FULL head_dim — this is what distinguishes
    // "proportional" RoPE from the default `nn.RoPE(rope_dims)` form.
    //
    // If this test regresses, it means the RoPE frequencies diverged from
    // upstream `mlx_vlm.models.gemma4.rope_utils.ProportionalRoPE`, which
    // is exactly the hazard that motivated issue #321.
    let head_dim = 256_i32;
    let prf = 0.25_f32;
    let base = 1_000_000.0_f32;
    let factor = 1.0_f32;

    let freqs = mlxcel_core::rope_proportional::compute_proportional_rope_freqs(
        head_dim, prf, base, factor,
    )
    .expect("freqs must exist for prf=0.25");
    mlxcel_core::eval(&freqs);

    // For head_dim=256 and prf=0.25, rope_angles = 32, but upstream pads the
    // table to head_dim/2 with `inf`.
    assert_eq!(
        mlxcel_core::array_shape(&freqs),
        vec![128],
        "freqs length must equal head_dim / 2"
    );

    // Pull the values back to host and spot-check a handful of entries.
    let freqs_f32 = mlxcel_core::astype(&freqs, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&freqs_f32);
    let freq_bytes = mlxcel_core::array_to_raw_bytes(&freqs_f32);
    assert_eq!(freq_bytes.len(), 128 * 4);
    let freq_values: Vec<f32> = freq_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    assert_eq!(freq_values.len(), 128);

    for (i, &got) in freq_values.iter().take(32).enumerate() {
        let expected = base.powf((2 * i) as f32 / head_dim as f32);
        let rel = (got - expected).abs() / expected.max(1.0);
        assert!(
            rel < 1e-4,
            "freqs[{i}] expected {expected}, got {got} (rel err {rel})"
        );
    }
    for (i, &got) in freq_values.iter().enumerate().skip(32) {
        assert!(
            got.is_infinite() && got.is_sign_positive(),
            "freqs[{i}] must be +inf"
        );
    }

    // Sanity: the second-to-last entry is noticeably smaller than
    // `base^(rotated_dims/head_dim)` in the default (non-proportional) form.
    // If mlxcel regressed to the default form, exponents would be normalized
    // by `rope_dims = 64` instead of `head_dim = 256`, giving
    //     default[i=15] = base^(30/64)  ≈ 1148
    //     proportional[i=15] = base^(30/256) ≈ 14.7
    // i.e. a ~78x larger value — a regression would be immediately obvious.
    let default_formula = base.powf(30.0 / 64.0);
    assert!(
        freq_values[15] < default_formula / 10.0,
        "freqs[15]={}, should be far smaller than default-RoPE formula ({}); \
         likely regression to non-proportional semantics",
        freq_values[15],
        default_formula,
    );
}
