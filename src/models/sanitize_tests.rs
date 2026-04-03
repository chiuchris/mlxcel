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

use super::sanitize::{sanitize_config_json, sanitize_tied_embeddings};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{self, dtype};
use safetensors::tensor::{Dtype as SafeTensorDtype, View};
use serde_json::json;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
