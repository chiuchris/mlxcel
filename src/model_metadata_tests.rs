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

use crate::model_metadata::{
    DirectoryLoadRoute, ModelKind, WeightLoadRoute, is_vlm_model_type, model_load_policy,
};
use crate::models::ModelType;
use serde_json::json;

#[test]
fn model_metadata_distinguishes_text_and_vlm_routes() {
    let text_policy = model_load_policy(ModelType::Qwen35, None).unwrap();
    assert_eq!(text_policy.capabilities.kind, ModelKind::Text);
    assert_eq!(text_policy.directory_route, DirectoryLoadRoute::Nonstandard);
    assert_eq!(text_policy.weight_route, Some(WeightLoadRoute::Special));

    let vlm_policy = model_load_policy(ModelType::Qwen3VL, None).unwrap();
    assert_eq!(vlm_policy.capabilities.kind, ModelKind::Vlm);
    assert_eq!(vlm_policy.directory_route, DirectoryLoadRoute::Vlm);
    assert_eq!(vlm_policy.weight_route, None);
}

#[test]
fn model_metadata_preserves_mistral3_nested_text_wrapper_rule() {
    let config = json!({
        "text_config": {
            "model_type": "ministral3"
        }
    });

    let policy = model_load_policy(ModelType::Mistral3, Some(&config)).unwrap();
    assert_eq!(
        policy.directory_route,
        DirectoryLoadRoute::Mistral3TextWrapper
    );
    assert_eq!(policy.weight_route, Some(WeightLoadRoute::LlamaFamily));
}

#[test]
fn is_vlm_model_type_matches_control_plane_capabilities() {
    assert!(is_vlm_model_type(ModelType::Gemma3VLM));
    assert!(is_vlm_model_type(ModelType::Phi3VLM));
    assert!(!is_vlm_model_type(ModelType::Gemma3));
    assert!(!is_vlm_model_type(ModelType::Mamba2));
}
