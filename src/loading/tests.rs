use super::{
    Qwen35VlmKind, is_ministral3_config, parse_eos_token_ids, qwen35_vlm_kind, read_eos_token_ids,
    resolve_model_dir,
};
use crate::models::ModelType;
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("mlxcel_loader_test_{name}_{nanos}"))
}

#[test]
fn parse_eos_token_ids_supports_single_number() {
    let config = json!({ "eos_token_id": 42 });
    assert_eq!(parse_eos_token_ids(&config), vec![42]);
}

#[test]
fn parse_eos_token_ids_supports_number_arrays() {
    let config = json!({ "eos_token_id": [1, 2, 3] });
    assert_eq!(parse_eos_token_ids(&config), vec![1, 2, 3]);
}

#[test]
fn parse_eos_token_ids_ignores_invalid_entries() {
    let config = json!({ "eos_token_id": [7, "bad", null, 9] });
    assert_eq!(parse_eos_token_ids(&config), vec![7, 9]);
}

#[test]
fn read_eos_token_ids_returns_empty_for_missing_file() {
    let missing_dir = temp_path("missing_generation_config");
    assert!(read_eos_token_ids(&missing_dir).is_empty());
}

#[test]
fn read_eos_token_ids_reads_generation_config() {
    let model_dir = temp_path("generation_config");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("generation_config.json"),
        r#"{ "eos_token_id": [10, 11] }"#,
    )
    .unwrap();

    assert_eq!(read_eos_token_ids(&model_dir), vec![10, 11]);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn resolve_model_dir_uses_parent_for_model_files() {
    let model_dir = temp_path("model_dir");
    fs::create_dir_all(&model_dir).unwrap();
    let model_file = model_dir.join("model.safetensors");
    fs::write(&model_file, b"").unwrap();

    assert_eq!(resolve_model_dir(&model_file), model_dir);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn resolve_model_dir_keeps_directory_paths() {
    let model_dir = temp_path("directory_passthrough");
    fs::create_dir_all(&model_dir).unwrap();

    assert_eq!(resolve_model_dir(&model_dir), model_dir);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn is_ministral3_config_detects_nested_model_type() {
    let config = json!({
        "text_config": {
            "model_type": "ministral3"
        }
    });

    assert!(is_ministral3_config(&config));
}

#[test]
fn is_ministral3_config_returns_false_without_matching_text_model() {
    let config = json!({
        "text_config": {
            "model_type": "llama"
        }
    });

    assert!(!is_ministral3_config(&config));
}

#[test]
fn qwen35_vlm_kind_matches_supported_model_types() {
    assert_eq!(
        qwen35_vlm_kind(ModelType::Qwen35VLM),
        Some(Qwen35VlmKind::Dense)
    );
    assert_eq!(
        qwen35_vlm_kind(ModelType::Qwen35MoeVLM),
        Some(Qwen35VlmKind::Moe)
    );
    assert_eq!(qwen35_vlm_kind(ModelType::Qwen35), None);
}
