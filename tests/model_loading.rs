use mlxcel::models::{ModelType, get_model_type};
use std::path::PathBuf;

#[test]
fn test_detect_llama_model_type() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("models/Meta-Llama-3.1-8B-Instruct-4bit");

    if !d.exists() {
        eprintln!("Skipping test: Model directory not found at {:?}", d);
        return;
    }

    let model_type = get_model_type(&d).expect("Failed to detect model type");
    assert_eq!(model_type, ModelType::Llama);
}
