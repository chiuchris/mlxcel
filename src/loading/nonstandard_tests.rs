use super::is_nonstandard_model_type;
use crate::models::ModelType;

#[test]
fn nonstandard_registry_covers_special_directory_families() {
    assert!(is_nonstandard_model_type(ModelType::Qwen35));
    assert!(is_nonstandard_model_type(ModelType::Gemma3n));
    assert!(is_nonstandard_model_type(ModelType::Rwkv7));
}

#[test]
fn nonstandard_registry_excludes_standard_and_vlm_families() {
    assert!(!is_nonstandard_model_type(ModelType::Llama));
    assert!(!is_nonstandard_model_type(ModelType::Qwen3VL));
    assert!(!is_nonstandard_model_type(ModelType::Step3p5));
}
