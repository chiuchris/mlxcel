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

use super::sanitize::{load_text_weights, sanitize_config_json, sanitize_tied_embeddings};
use mlxcel_core::weights::{WeightMap, WeightTransform};
use mlxcel_core::{self, dtype};
use safetensors::tensor::{Dtype as SafeTensorDtype, View};
use serde_json::json;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

fn sample_weight_map(key: &str) -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert(key.to_string(), mlxcel_core::ones(&[2, 2], dtype::FLOAT32));
    weights
}

#[derive(Clone)]
struct OwnedTensor {
    dtype: SafeTensorDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
    fn dtype(&self) -> SafeTensorDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.as_slice().into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl View for OwnedTensor {
    fn dtype(&self) -> SafeTensorDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.as_slice().into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn temp_model_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("mlxcel_sanitize_test_{name}_{nanos}"))
}

fn write_safetensors(path: &Path, tensors: &[(&str, OwnedTensor)]) {
    let mut views: HashMap<String, OwnedTensor> = HashMap::new();
    for (name, tensor) in tensors {
        views.insert((*name).to_string(), tensor.clone());
    }
    safetensors::serialize_to_file(&views, None, path).unwrap();
}

#[test]
fn sanitize_config_json_replaces_non_standard_values() {
    let sanitized = sanitize_config_json("{\"a\": Infinity, \"b\": -Infinity, \"c\": NaN}");
    assert_eq!(sanitized, "{\"a\": 1e38, \"b\": -1e38, \"c\": 0.0}");
}

#[test]
fn sanitize_tied_embeddings_copies_standard_embed_tokens_when_missing() {
    let mut weights = sample_weight_map("model.embed_tokens.weight");
    sanitize_tied_embeddings(&mut weights, &json!({}));

    assert!(weights.contains_key("lm_head.weight"));
}

#[test]
fn sanitize_tied_embeddings_copies_prefixed_language_model_keys() {
    let mut weights = sample_weight_map("language_model.model.embed_tokens.weight");
    sanitize_tied_embeddings(&mut weights, &json!({}));

    assert!(weights.contains_key("language_model.lm_head.weight"));
}

#[test]
fn sanitize_tied_embeddings_respects_explicit_untied_config() {
    let mut weights = sample_weight_map("model.embed_tokens.weight");
    sanitize_tied_embeddings(&mut weights, &json!({ "tie_word_embeddings": false }));

    assert!(!weights.contains_key("lm_head.weight"));
}

#[test]
fn load_and_sanitize_weights_selectively_keeps_gemma4_text_tensors() {
    let dir = temp_model_dir("gemma4_selective");
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "gemma4",
            "tie_word_embeddings": false,
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "text_config": {
                "model_type": "gemma4",
                "quantization": {
                    "group_size": 64,
                    "bits": 4
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();

    write_safetensors(
        &dir.join("model-00001-of-00002.safetensors"),
        &[(
            "language_model.model.layers.0.self_attn.q_proj.weight",
            OwnedTensor {
                dtype: SafeTensorDtype::F32,
                shape: vec![2],
                data: vec![0, 0, 128, 63, 0, 0, 0, 64],
            },
        )],
    );
    write_safetensors(
        &dir.join("model-00002-of-00002.safetensors"),
        &[
            (
                "language_model.model.per_layer_projection_norm.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::BF16,
                    shape: vec![1],
                    data: vec![0x80, 0x3F],
                },
            ),
            (
                "language_model.model.embed_tokens_per_layer.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::U32,
                    shape: vec![1],
                    data: 7_u32.to_le_bytes().to_vec(),
                },
            ),
            (
                "vision_tower.vision_model.embeddings.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::F32,
                    shape: vec![1],
                    data: vec![0, 0, 64, 64],
                },
            ),
        ],
    );

    let weights = super::sanitize::load_and_sanitize_weights(&dir).unwrap();

    assert!(weights.contains_key("language_model.model.layers.0.self_attn.q_proj.weight"));
    assert!(weights.contains_key("language_model.model.per_layer_projection_norm.weight"));
    assert!(weights.contains_key("language_model.model.embed_tokens_per_layer.weight"));
    assert!(!weights.contains_key("vision_tower.vision_model.embeddings.weight"));

    let bf16 = weights
        .get("language_model.model.per_layer_projection_norm.weight")
        .unwrap();
    let expected_bf16_dtype = if mlxcel_core::hardware::get_hardware().silicon_gen
        != mlxcel_core::hardware::AppleSiliconGen::Unknown
    {
        dtype::FLOAT16
    } else {
        dtype::BFLOAT16
    };
    assert_eq!(mlxcel_core::array_dtype(bf16), expected_bf16_dtype);
    let bf16_f32 = mlxcel_core::astype(bf16, dtype::FLOAT32);
    mlxcel_core::eval(&bf16_f32);
    assert!((mlxcel_core::item_f32(&bf16_f32) - 1.0).abs() < 0.01);

    let quant = weights
        .get("language_model.model.embed_tokens_per_layer.weight")
        .unwrap();
    assert_eq!(mlxcel_core::array_dtype(quant), dtype::UINT32);
    let quant_i64 = mlxcel_core::astype(quant, dtype::INT64);
    mlxcel_core::eval(&quant_i64);
    assert_eq!(mlxcel_core::item_i64(&quant_i64), 7);

    std::fs::remove_dir_all(&dir).unwrap();
}

/// Verify that loading an nvfp4 Gemma 4 checkpoint:
/// 1. Remaps `model.language_model.X` → `language_model.model.X`
/// 2. Dequantizes the packed U8 FP4 weight tensor to F16
///
/// Test data: 2×16 packed U8 weights (nibble 0x2 = FP4 E2M1 value 1.0),
/// 2×2 F16 block scales (all 1.0 = 0x3C00), global F32 scale 1.0.
/// Expected output: 2×32 F16 tensor with all values = 1.0.
#[test]
fn load_and_sanitize_weights_dequantizes_nvfp4_gemma4_checkpoint() {
    let dir = temp_model_dir("gemma4_nvfp4");
    std::fs::create_dir_all(&dir).unwrap();

    // Non-quantized Gemma 4 config so the bf16→f16 path is active but
    // no quantization field blocks nvfp4 dequantization.
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "gemma4",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "gemma4"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    // Shape: [out_dim=2, packed_dim=16]. Each byte encodes two FP4 E2M1
    // nibbles. Nibble 0x2 = 1.0, so byte 0x22 → [1.0, 1.0]. Expected
    // dequantized shape: [2, 32].
    let out_dim: usize = 2;
    let packed_dim: usize = 16; // in_dim / 2 = 32 / 2
    let weight_data = vec![0x22u8; out_dim * packed_dim];

    // F16 block scales: shape [out_dim=2, num_groups=2]. 1.0 in F16 = 0x3C00
    // (little-endian bytes [0x00, 0x3C]).
    let f16_one: [u8; 2] = [0x00, 0x3C];
    let num_groups = 2usize;
    let mut scale_data = Vec::with_capacity(out_dim * num_groups * 2);
    for _ in 0..(out_dim * num_groups) {
        scale_data.extend_from_slice(&f16_one);
    }

    // Global F32 scale: 1-element F32 tensor with value 1.0.
    // 1.0 in F32 little-endian = [0x00, 0x00, 0x80, 0x3F].
    let scale2_data: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3F];

    // Use nvfp4-style key prefix: `model.language_model.layers.0.mlp.gate_proj`
    let prefix = "model.language_model.layers.0.mlp.gate_proj";
    write_safetensors(
        &dir.join("model.safetensors"),
        &[
            (
                &format!("{prefix}.weight"),
                OwnedTensor {
                    dtype: SafeTensorDtype::U8,
                    shape: vec![out_dim, packed_dim],
                    data: weight_data,
                },
            ),
            (
                &format!("{prefix}.weight_scale"),
                OwnedTensor {
                    dtype: SafeTensorDtype::F16,
                    shape: vec![out_dim, num_groups],
                    data: scale_data,
                },
            ),
            (
                &format!("{prefix}.weight_scale_2"),
                OwnedTensor {
                    dtype: SafeTensorDtype::F32,
                    shape: vec![1],
                    data: scale2_data,
                },
            ),
        ],
    );

    let weights = super::sanitize::load_and_sanitize_weights(&dir).unwrap();

    // After normalize_nvfp4_keys, the key should be remapped.
    let expected_key = "language_model.model.layers.0.mlp.gate_proj.weight";
    assert!(
        weights.contains_key(expected_key),
        "Expected dequantized key '{expected_key}' not found; keys: {:?}",
        weights.keys().collect::<Vec<_>>()
    );

    // Auxiliary scale keys must have been removed.
    assert!(!weights.contains_key(&format!("{prefix}.weight_scale")));
    assert!(!weights.contains_key(&format!("{prefix}.weight_scale_2")));

    // The dequantized weight must be F16 with shape [out_dim, in_dim].
    let w = weights.get(expected_key).unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(w),
        dtype::FLOAT16,
        "Expected F16 after dequantization"
    );
    let w_f32 = mlxcel_core::astype(w, dtype::FLOAT32);
    mlxcel_core::eval(&w_f32);
    let shape = mlxcel_core::array_shape(&w_f32);
    assert_eq!(shape, vec![out_dim as i32, 32i32], "Expected shape [2, 32]");

    // All values should be 1.0 * 1.0 * 1.0 = 1.0 within f16 precision.
    let w_bytes = mlxcel_core::array_to_raw_bytes(&w_f32);
    for chunk in w_bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(
            (v - 1.0f32).abs() < 1e-3,
            "Expected 1.0, got {v}; nibble=0x2 (1.0) * scale=1.0 * scale2=1.0 should equal 1.0"
        );
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

// --- Tests for the consolidated `load_text_weights` entry point and the
// optional `WeightTransform` hook introduced for Axis A (issue #365). ---

/// Build a minimal text model fixture: tiny config.json with no
/// quantization plus a single safetensors shard containing
/// `model.embed_tokens.weight`. Enough to exercise the consolidated
/// loader's sanitize → optional transform → precision-cast pipeline.
fn write_text_model_fixture(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "llama",
            "tie_word_embeddings": true,
        }))
        .unwrap(),
    )
    .unwrap();

    // Two F32 weights so we can verify both that sanitize copies
    // `embed_tokens` into `lm_head` (via `tie_word_embeddings: true`)
    // and that the transform observes the right keys.
    let weight_data = vec![0u8; 4 * 4]; // 4 x F32 zeros
    write_safetensors(
        &dir.join("model.safetensors"),
        &[
            (
                "model.embed_tokens.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::F32,
                    shape: vec![2, 2],
                    data: weight_data,
                },
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::F32,
                    shape: vec![2, 2],
                    data: vec![0u8; 4 * 4],
                },
            ),
        ],
    );
}

#[test]
fn load_text_weights_with_none_transform_matches_legacy_path() {
    // Bit-exactness (a) — when `transform = None` the consolidated
    // entry point must produce exactly the same `WeightMap` (same
    // keys, same dtypes, same shapes, byte-identical contents) as the
    // legacy `load_and_sanitize_weights` alias. The alias itself is
    // implemented as `load_text_weights(p, None)`, so this test
    // doubles as a regression guard against accidental divergence
    // between the alias and the new entry point.
    let dir_a = temp_model_dir("text_load_none_transform_a");
    let dir_b = temp_model_dir("text_load_none_transform_b");
    write_text_model_fixture(&dir_a);
    write_text_model_fixture(&dir_b);

    let legacy = super::sanitize::load_and_sanitize_weights(&dir_a).unwrap();
    let consolidated = load_text_weights(&dir_b, None).unwrap();

    assert_eq!(
        legacy.len(),
        consolidated.len(),
        "WeightMap entry count must match"
    );
    let mut keys_a: Vec<&String> = legacy.keys().collect();
    let mut keys_b: Vec<&String> = consolidated.keys().collect();
    keys_a.sort();
    keys_b.sort();
    assert_eq!(keys_a, keys_b, "key sets must match");

    for k in keys_a {
        let a = legacy.get(k).unwrap();
        let b = consolidated.get(k).unwrap();
        assert_eq!(
            mlxcel_core::array_dtype(a),
            mlxcel_core::array_dtype(b),
            "dtype mismatch for key {k}"
        );
        assert_eq!(
            mlxcel_core::array_shape(a),
            mlxcel_core::array_shape(b),
            "shape mismatch for key {k}"
        );
        mlxcel_core::eval(a);
        mlxcel_core::eval(b);
        let a_bytes = mlxcel_core::array_to_raw_bytes(a);
        let b_bytes = mlxcel_core::array_to_raw_bytes(b);
        assert_eq!(a_bytes, b_bytes, "byte mismatch for key {k}");
    }

    std::fs::remove_dir_all(&dir_a).unwrap();
    std::fs::remove_dir_all(&dir_b).unwrap();
}

/// Counter-based `WeightTransform` that records how many times
/// `apply` ran. Used to prove the hook is actually invoked by
/// `load_text_weights` when a non-`None` transform is supplied.
struct CountingTransform {
    calls: AtomicUsize,
}

impl CountingTransform {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl WeightTransform for CountingTransform {
    fn apply(&self, _weights: &mut WeightMap, _cfg: &serde_json::Value) -> Result<(), String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn load_text_weights_invokes_transform_when_supplied() {
    // Integration (b) — the optional hook must be invoked exactly
    // once per call when a transform is supplied. With a no-op
    // transform the produced `WeightMap` must match the `None` path
    // bit-for-bit.
    let dir_a = temp_model_dir("text_load_transform_invoked_a");
    let dir_b = temp_model_dir("text_load_transform_invoked_b");
    write_text_model_fixture(&dir_a);
    write_text_model_fixture(&dir_b);

    let baseline = load_text_weights(&dir_a, None).unwrap();
    let transform = CountingTransform::new();
    let with_hook = load_text_weights(&dir_b, Some(&transform)).unwrap();

    assert_eq!(
        transform.call_count(),
        1,
        "WeightTransform::apply must run exactly once per load"
    );
    assert_eq!(baseline.len(), with_hook.len());

    // Verify byte-level equivalence between the no-transform path and
    // the empty-transform path (this is the bit-exactness guarantee
    // a no-op pipeline must uphold).
    for (k, base) in &baseline {
        let other = with_hook
            .get(k)
            .unwrap_or_else(|| panic!("missing key {k}"));
        mlxcel_core::eval(base);
        mlxcel_core::eval(other);
        assert_eq!(
            mlxcel_core::array_to_raw_bytes(base),
            mlxcel_core::array_to_raw_bytes(other),
            "no-op transform must preserve weights bit-for-bit: key={k}"
        );
    }

    std::fs::remove_dir_all(&dir_a).unwrap();
    std::fs::remove_dir_all(&dir_b).unwrap();
}

/// `WeightTransform` that returns an error so we can verify error
/// propagation from the consolidated loader.
struct FailingTransform;

impl WeightTransform for FailingTransform {
    fn apply(&self, _weights: &mut WeightMap, _cfg: &serde_json::Value) -> Result<(), String> {
        Err("intentional failure for test".to_string())
    }
}

#[test]
fn load_text_weights_propagates_transform_error() {
    let dir = temp_model_dir("text_load_transform_error");
    write_text_model_fixture(&dir);

    let transform = FailingTransform;
    let result = load_text_weights(&dir, Some(&transform));
    match result {
        Ok(_) => panic!("expected failing transform to surface an error"),
        Err(err) => assert!(
            err.contains("intentional failure for test"),
            "expected transform error to be surfaced, got: {err}"
        ),
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

// --- Integration tests for the Axis A `mlxcel-surgery` crate (#367).
// Gated on the `surgery` feature so the default build still passes
// without pulling the new crate into the dependency graph. ---

#[cfg(feature = "surgery")]
mod surgery_integration {
    use super::*;
    use mlxcel_surgery::{SharedSurgeryOp, SurgeryError, SurgeryOp, SurgeryPipeline};
    use std::sync::Arc;

    /// `SurgeryOp` that simply notes that it ran. Used to prove the
    /// `SurgeryPipeline` is actually invoked by A1's hook, without
    /// touching the weight values (so the bit-exactness assertion
    /// still holds).
    struct RecordingOp {
        counter: Arc<AtomicUsize>,
    }

    impl SurgeryOp for RecordingOp {
        fn apply(
            &self,
            _weights: &mut WeightMap,
            _cfg: &serde_json::Value,
        ) -> Result<(), SurgeryError> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn name(&self) -> &'static str {
            "recording-noop"
        }
    }

    #[test]
    fn empty_surgery_pipeline_is_bit_exact_with_none() {
        // Acceptance criterion (b) / (e) — wiring a `SurgeryPipeline`
        // through the consolidated loader with no registered ops must
        // produce the same `WeightMap` as `transform = None`. This
        // doubles as proof that the new crate's `SurgeryPipeline`
        // really does implement A1's `WeightTransform` trait at the
        // type level (the call would not type-check otherwise).
        let dir_a = temp_model_dir("surgery_empty_a");
        let dir_b = temp_model_dir("surgery_empty_b");
        write_text_model_fixture(&dir_a);
        write_text_model_fixture(&dir_b);

        let baseline = load_text_weights(&dir_a, None).unwrap();
        let pipeline = SurgeryPipeline::new();
        let with_empty = load_text_weights(&dir_b, Some(&pipeline)).unwrap();

        assert_eq!(baseline.len(), with_empty.len());
        for (k, base) in &baseline {
            let other = with_empty
                .get(k)
                .unwrap_or_else(|| panic!("missing key {k} in surgery path"));
            mlxcel_core::eval(base);
            mlxcel_core::eval(other);
            assert_eq!(
                mlxcel_core::array_to_raw_bytes(base),
                mlxcel_core::array_to_raw_bytes(other),
                "empty surgery pipeline must preserve key {k} bit-for-bit",
            );
        }

        std::fs::remove_dir_all(&dir_a).unwrap();
        std::fs::remove_dir_all(&dir_b).unwrap();
    }

    #[test]
    fn surgery_pipeline_runs_registered_ops_during_load() {
        // Acceptance criterion (b) — a non-empty pipeline routed
        // through A1's hook must actually invoke each registered op.
        // We use a recording no-op so the resulting `WeightMap` is
        // still bit-identical to the baseline (because the op does
        // not mutate weights), which proves that "no-op" semantics
        // hold end-to-end.
        let dir_a = temp_model_dir("surgery_recording_a");
        let dir_b = temp_model_dir("surgery_recording_b");
        write_text_model_fixture(&dir_a);
        write_text_model_fixture(&dir_b);

        let counter = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SurgeryPipeline::new();
        let op: SharedSurgeryOp = Arc::new(RecordingOp {
            counter: counter.clone(),
        });
        pipeline.push(op);

        let baseline = load_text_weights(&dir_a, None).unwrap();
        let with_pipeline = load_text_weights(&dir_b, Some(&pipeline)).unwrap();

        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "recording op must have run exactly once during load",
        );

        for (k, base) in &baseline {
            let other = with_pipeline
                .get(k)
                .unwrap_or_else(|| panic!("missing key {k} in surgery path"));
            mlxcel_core::eval(base);
            mlxcel_core::eval(other);
            assert_eq!(
                mlxcel_core::array_to_raw_bytes(base),
                mlxcel_core::array_to_raw_bytes(other),
                "recording no-op surgery must preserve key {k} bit-for-bit",
            );
        }

        std::fs::remove_dir_all(&dir_a).unwrap();
        std::fs::remove_dir_all(&dir_b).unwrap();
    }

    #[test]
    fn surgery_error_propagates_through_load_path() {
        // Acceptance criterion (b) — when a surgery op fails, the
        // error must surface out of `load_text_weights` rather than
        // being silently swallowed. Use the existing
        // `SurgeryError::TensorNotFound` variant since it exercises
        // the `Display` impl that the pipeline funnels through
        // `WeightTransform::apply`'s `Result<_, String>` return.
        struct FailingOp;
        impl SurgeryOp for FailingOp {
            fn apply(
                &self,
                _weights: &mut WeightMap,
                _cfg: &serde_json::Value,
            ) -> Result<(), SurgeryError> {
                Err(SurgeryError::TensorNotFound(
                    "expected.key.for.test".to_string(),
                ))
            }
            fn name(&self) -> &'static str {
                "failing"
            }
        }

        let dir = temp_model_dir("surgery_failing");
        write_text_model_fixture(&dir);

        let mut pipeline = SurgeryPipeline::new();
        pipeline.push(Arc::new(FailingOp));

        let result = load_text_weights(&dir, Some(&pipeline));
        match result {
            Ok(_) => panic!("expected failing surgery op to abort the load"),
            Err(err) => {
                assert!(
                    err.contains("failing") && err.contains("expected.key.for.test"),
                    "expected error to identify failing op + key, got: {err}",
                );
            }
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
