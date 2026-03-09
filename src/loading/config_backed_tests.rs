use super::is_config_backed_model_type;
use crate::models::ModelType;

#[test]
fn config_backed_registry_covers_standard_text_models() {
    assert!(is_config_backed_model_type(ModelType::Llama));
    assert!(is_config_backed_model_type(ModelType::Gemma3));
    assert!(is_config_backed_model_type(ModelType::Step3p5));
}

#[test]
fn config_backed_registry_excludes_special_and_vlm_models() {
    assert!(!is_config_backed_model_type(ModelType::Qwen35));
    assert!(!is_config_backed_model_type(ModelType::Gemma3n));
    assert!(!is_config_backed_model_type(ModelType::Qwen3VL));
}
