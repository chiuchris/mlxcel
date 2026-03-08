use super::{
    ModelType, detect_hunyuan_model_type, detect_text_or_vlm, has_vision_config,
    sanitize_config_json,
};
use serde_json::json;

#[test]
fn sanitize_config_json_replaces_non_standard_values() {
    let sanitized = sanitize_config_json("{\"a\": Infinity, \"b\": -Infinity, \"c\": NaN}");

    assert_eq!(sanitized, "{\"a\": 1e38, \"b\": -1e38, \"c\": 0.0}");
}

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
