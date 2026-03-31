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

//! Quantization recommendation helpers for the CLI and server.
//!
//! This module bridges the low-level [`mlxcel_core::hardware`] recommendation
//! engine with the higher-level model loading path. It can estimate model
//! parameter counts from a `config.json` and emit human-readable advice.

use std::path::Path;

use mlxcel_core::hardware::{HardwareCapabilities, QuantRecommendation, recommend_quantization};

// ── Model-size estimation ─────────────────────────────────────────────────────

/// Try to estimate the approximate number of model parameters (in billions)
/// from the model's `config.json`.
///
/// This is a best-effort approximation that works for the most common dense
/// and MoE transformer architectures. Returns `None` when the config cannot be
/// read or when the required fields are absent.
///
/// The estimation formula mirrors what mlx-lm tooling uses for rough sizing:
/// ```text
/// params ≈ vocab_size * hidden_size          (embedding table)
///         + num_hidden_layers * (
///             4 * hidden_size²               (attention Q/K/V/O)
///             + 3 * hidden_size * ffn_size   (MLP gate/up/down, approx)
///           )
/// ```
///
/// For MoE models (`num_experts` > 1), the FFN term is scaled by the number
/// of experts, giving a rough total parameter count including inactive experts.
pub fn estimate_model_params_billions(model_path: &Path) -> Option<f64> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config_str).ok()?;

    estimate_params_from_config(&config)
}

fn estimate_params_from_config(config: &serde_json::Value) -> Option<f64> {
    // Support models that wrap text config under a "text_config" key (VLMs).
    let text_cfg = config
        .get("text_config")
        .unwrap_or(config);

    let hidden_size = text_cfg
        .get("hidden_size")
        .or_else(|| text_cfg.get("d_model"))
        .and_then(|v| v.as_f64())?;

    let num_layers = text_cfg
        .get("num_hidden_layers")
        .or_else(|| text_cfg.get("n_layers"))
        .or_else(|| text_cfg.get("num_layers"))
        .and_then(|v| v.as_f64())?;

    let vocab_size = text_cfg
        .get("vocab_size")
        .and_then(|v| v.as_f64())
        .unwrap_or(32_000.0);

    // FFN intermediate size — try several field names used across architectures.
    let ffn_size = text_cfg
        .get("intermediate_size")
        .or_else(|| text_cfg.get("ffn_dim"))
        .or_else(|| text_cfg.get("ffn_hidden_size"))
        .and_then(|v| v.as_f64())
        .unwrap_or(hidden_size * 4.0); // fallback: 4× hidden as a rule of thumb

    // Number of experts (MoE). For dense models this stays at 1.
    let num_experts = text_cfg
        .get("num_experts")
        .or_else(|| text_cfg.get("num_local_experts"))
        .or_else(|| text_cfg.get("n_routed_experts"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0)
        .max(1.0);

    // Attention heads (used for GQA parameter correction, optional).
    let num_heads = text_cfg
        .get("num_attention_heads")
        .or_else(|| text_cfg.get("num_heads"))
        .and_then(|v| v.as_f64())
        .unwrap_or(hidden_size / 64.0);

    let kv_heads = text_cfg
        .get("num_key_value_heads")
        .and_then(|v| v.as_f64())
        .unwrap_or(num_heads);

    // head_dim inferred from hidden_size / num_heads.
    let head_dim = (hidden_size / num_heads).round();

    // Parameter count estimate:
    //   embedding:  vocab_size × hidden_size (input + output embedding shared)
    //   per-layer attention:
    //     Q: hidden_size × (num_heads × head_dim)
    //     K: hidden_size × (kv_heads × head_dim)
    //     V: hidden_size × (kv_heads × head_dim)
    //     O: (num_heads × head_dim) × hidden_size
    //   per-layer MLP: 3 × hidden_size × ffn_size × num_experts
    //   norms: 2 × num_layers × hidden_size (small, included for accuracy)
    let embedding_params = vocab_size * hidden_size;

    let attn_params_per_layer =
        hidden_size * (num_heads * head_dim)          // Q proj
        + hidden_size * (kv_heads * head_dim)         // K proj
        + hidden_size * (kv_heads * head_dim)         // V proj
        + (num_heads * head_dim) * hidden_size;       // O proj

    let ffn_params_per_layer = 3.0 * hidden_size * ffn_size * num_experts;
    let norm_params_per_layer = 2.0 * hidden_size;

    let total_params = embedding_params
        + num_layers * (attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer);

    Some(total_params / 1e9)
}

// ── BFloat16 detection ────────────────────────────────────────────────────────

/// Torch dtype strings that indicate BFloat16 weights.
const BFLOAT16_DTYPE_STRINGS: &[&str] = &["bfloat16", "bf16", "torch.bfloat16"];

/// Return `true` if the model's `config.json` declares BFloat16 as the
/// default weight dtype.
///
/// The M5 Neural Accelerator does **not** support BFloat16 computation, so
/// models with BF16 weights should be converted to FP16 or quantized before
/// running to avoid a fallback to the GPU shader pipeline.
pub fn model_uses_bfloat16(model_path: &Path) -> bool {
    let config_path = model_path.join("config.json");
    let Ok(config_str) = std::fs::read_to_string(&config_path) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) else {
        return false;
    };

    // Check both the top-level and text_config for the dtype field.
    let top_level_dtype = config
        .get("torch_dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let text_cfg_dtype = config
        .get("text_config")
        .and_then(|tc| tc.get("torch_dtype"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    BFLOAT16_DTYPE_STRINGS.contains(&top_level_dtype)
        || BFLOAT16_DTYPE_STRINGS.contains(&text_cfg_dtype)
}

// ── High-level advisor ────────────────────────────────────────────────────────

/// Advice produced by [`advise_quantization`].
#[derive(Debug)]
pub struct QuantAdvice {
    /// The recommended quantization mode.
    pub recommendation: QuantRecommendation,
    /// Estimated parameter count in billions (None if config parsing failed).
    pub estimated_params_billions: Option<f64>,
    /// True when the model config declares BFloat16 weights.
    pub model_uses_bfloat16: bool,
}

/// Produce a complete quantization recommendation for a model directory.
///
/// Uses the model's `config.json` to estimate the parameter count, then calls
/// [`recommend_quantization`] against the provided hardware capabilities.
///
/// When `model_params_override` is `Some(n)`, that value is used instead of
/// the estimate from `config.json`.
pub fn advise_quantization(
    model_path: &Path,
    hw: &HardwareCapabilities,
    model_params_override: Option<f64>,
) -> QuantAdvice {
    let estimated_params = estimate_model_params_billions(model_path);
    let params = model_params_override
        .or(estimated_params)
        .unwrap_or(7.0); // safe fallback: assume 7B when unknown

    let recommendation =
        recommend_quantization(params, hw.unified_memory_gb, hw);

    let uses_bf16 = model_uses_bfloat16(model_path);

    QuantAdvice {
        recommendation,
        estimated_params_billions: estimated_params,
        model_uses_bfloat16: uses_bf16,
    }
}

/// Print a human-readable quantization recommendation to stdout.
pub fn print_quant_advice(advice: &QuantAdvice, hw: &HardwareCapabilities) {
    println!();
    println!("=== Quantization Recommendation ===");
    println!("  Hardware:   {} ({})", hw.silicon_gen, {
        if hw.has_neural_accelerator && hw.macos_supports_na {
            "Neural Accelerator active"
        } else if hw.has_neural_accelerator {
            "Neural Accelerator requires macOS 26.2+"
        } else {
            "no Neural Accelerator"
        }
    });
    println!("  Memory:     {} GB unified", hw.unified_memory_gb);

    if let Some(params) = advice.estimated_params_billions {
        println!("  Model size: ~{:.1}B parameters (estimated)", params);
    } else {
        println!("  Model size: unknown (could not parse config.json)");
    }

    println!();
    println!("  Recommendation: {}", advice.recommendation.label().to_uppercase());
    println!("  Reason:         {}", advice.recommendation.reason());

    if advice.model_uses_bfloat16 && hw.has_neural_accelerator {
        println!();
        println!("  WARNING: This model uses BFloat16 weights.");
        println!("  The M5 Neural Accelerator does not support BFloat16 computation.");
        println!("  For best performance, use an INT8 or FP16 quantized variant of this model.");
    }

    println!();
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(hidden: u64, layers: u64, vocab: u64, ffn: u64) -> serde_json::Value {
        serde_json::json!({
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "vocab_size": vocab,
            "intermediate_size": ffn,
            "num_attention_heads": 32,
            "num_key_value_heads": 8
        })
    }

    #[test]
    fn estimates_llama3_8b_in_correct_range() {
        // Llama 3 8B: 4096 hidden, 32 layers, 128256 vocab, 14336 ffn, GQA 32/8.
        let config = make_config(4096, 32, 128256, 14336);
        let params = estimate_params_from_config(&config).unwrap();
        // Should estimate between 7B and 10B.
        assert!(
            params >= 7.0 && params <= 10.0,
            "Expected 7-10B for Llama3-8B config, got {:.2}B",
            params
        );
    }

    #[test]
    fn estimates_qwen2_0_5b_in_correct_range() {
        // Qwen2 0.5B: 896 hidden, 24 layers, 151936 vocab, 4864 ffn.
        let config = make_config(896, 24, 151936, 4864);
        let params = estimate_params_from_config(&config).unwrap();
        assert!(
            params >= 0.3 && params <= 1.0,
            "Expected 0.3-1.0B for Qwen2-0.5B config, got {:.2}B",
            params
        );
    }

    #[test]
    fn text_config_nested_works() {
        // VLM style: text model params under "text_config".
        let config = serde_json::json!({
            "model_type": "llava",
            "text_config": {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
                "intermediate_size": 11008,
                "num_attention_heads": 32
            }
        });
        let params = estimate_params_from_config(&config).unwrap();
        assert!(params > 5.0, "Expected >5B for LLaVA-7B config, got {:.2}B", params);
    }

    #[test]
    fn returns_none_for_empty_config() {
        let config = serde_json::json!({});
        let result = estimate_params_from_config(&config);
        assert!(result.is_none());
    }

    #[test]
    fn detects_bfloat16_from_torch_dtype() {
        // model_uses_bfloat16 reads from disk, so we test the underlying logic.
        let bfloat16_dtypes = ["bfloat16", "bf16", "torch.bfloat16"];
        for dtype in &bfloat16_dtypes {
            assert!(
                BFLOAT16_DTYPE_STRINGS.contains(dtype),
                "{dtype} should be recognized as bfloat16"
            );
        }
        assert!(!BFLOAT16_DTYPE_STRINGS.contains(&"float16"));
        assert!(!BFLOAT16_DTYPE_STRINGS.contains(&"fp16"));
    }

    #[test]
    fn advise_quantization_uses_override_params() {
        use mlxcel_core::hardware::{AppleSiliconGen, HardwareCapabilities};

        let hw = HardwareCapabilities {
            silicon_gen: AppleSiliconGen::M5,
            gpu_core_count: 10,
            has_neural_accelerator: true,
            metal_version: 4,
            macos_supports_na: true,
            memory_bandwidth_gbps: 150.0,
            unified_memory_gb: 32,
        };

        // Pass a temp dir — override_params bypasses config.json read.
        let tmp = std::env::temp_dir();
        let advice = advise_quantization(&tmp, &hw, Some(7.0));

        // With 7B model on 32 GB M5 with NA: expect INT8.
        assert_eq!(
            advice.recommendation,
            QuantRecommendation::Int8 {
                reason: "M5 NA delivers 2x throughput for INT8 vs FP16",
            }
        );
    }
}
