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

use super::ModelType;
use super::detection::{detect_hunyuan_model_type, detect_text_or_vlm, has_vision_config};
use serde_json::json;

#[test]
fn has_vision_config_detects_vlm_configs() {
    assert!(has_vision_config(&json!({ "vision_config": {} })));
    assert!(!has_vision_config(&json!({ "text_config": {} })));
}

#[test]
fn detect_text_or_vlm_prefers_vlm_when_vision_config_exists() {
    let vlm = detect_text_or_vlm(
        &json!({ "vision_config": {} }),
        ModelType::Gemma3,
        ModelType::Gemma3VLM,
    );
    let text = detect_text_or_vlm(&json!({}), ModelType::Gemma3, ModelType::Gemma3VLM);

    assert_eq!(vlm, ModelType::Gemma3VLM);
    assert_eq!(text, ModelType::Gemma3);
}

#[test]
fn detect_hunyuan_model_type_uses_num_experts() {
    assert_eq!(
        detect_hunyuan_model_type(&json!({ "num_experts": 4 })),
        ModelType::HunyuanMoe
    );
    assert_eq!(
        detect_hunyuan_model_type(&json!({ "num_experts": 1 })),
        ModelType::HunyuanV1Dense
    );
    assert_eq!(
        detect_hunyuan_model_type(&json!({})),
        ModelType::HunyuanV1Dense
    );
}
