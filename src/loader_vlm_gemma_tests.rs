use super::{
    gemma3n_language_model_prefix, gemma3n_metadata, gemma3n_needs_conv_transpose,
    sanitize_gemma3n_weights,
};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

#[test]
fn gemma3n_metadata_applies_defaults_and_overrides() {
    let defaults = gemma3n_metadata(&json!({}));
    assert_eq!(defaults.vision_hidden_size, 2048);
    assert_eq!(defaults.image_size, 256);
    assert_eq!(defaults.image_token_id, 262_145);
    assert_eq!(defaults.boi_token_id, 255_999);
    assert_eq!(defaults.eoi_token_id, 262_144);
    assert!((defaults.vision_rms_eps - 1e-6).abs() < f32::EPSILON);

    let overrides = gemma3n_metadata(&json!({
        "vision_config": {
            "hidden_size": 3072,
            "image_size": 384,
            "rms_norm_eps": 1e-5
        },
        "image_token_index": 9,
        "boi_token_id": 10,
        "eoi_token_id": 11
    }));
    assert_eq!(overrides.vision_hidden_size, 3072);
    assert_eq!(overrides.image_size, 384);
    assert_eq!(overrides.image_token_id, 9);
    assert_eq!(overrides.boi_token_id, 10);
    assert_eq!(overrides.eoi_token_id, 11);
    assert!((overrides.vision_rms_eps - 1e-5).abs() < f32::EPSILON);
}

#[test]
fn gemma3n_language_model_prefix_prefers_quantized_prefix_when_present() {
    let mut quantized = WeightMap::new();
    quantized.insert(
        "language_model.model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    assert_eq!(
        gemma3n_language_model_prefix(&quantized),
        "language_model.model"
    );

    let mut dense = WeightMap::new();
    dense.insert(
        "language_model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    assert_eq!(gemma3n_language_model_prefix(&dense), "language_model");
}

#[test]
fn gemma3n_needs_conv_transpose_checks_reference_conv_shape() {
    let mut raw_weights = WeightMap::new();
    raw_weights.insert(
        "model.vision_tower.timm_model.blocks.0.0.conv_exp.weight".to_string(),
        mlxcel_core::ones(&[8, 16, 3, 3], dtype::FLOAT32),
    );
    assert!(gemma3n_needs_conv_transpose(&raw_weights));

    raw_weights.insert(
        "model.vision_tower.timm_model.blocks.0.0.conv_exp.weight".to_string(),
        mlxcel_core::ones(&[8, 3, 3, 16], dtype::FLOAT32),
    );
    assert!(!gemma3n_needs_conv_transpose(&raw_weights));
}

#[test]
fn sanitize_gemma3n_weights_strips_model_prefix_and_transposes_conv_weights() {
    let mut raw_weights = WeightMap::new();
    raw_weights.insert(
        "model.vision_tower.timm_model.blocks.0.0.conv_exp.weight".to_string(),
        mlxcel_core::ones(&[8, 16, 3, 3], dtype::FLOAT32),
    );
    raw_weights.insert(
        "model.language_model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );

    let sanitized = sanitize_gemma3n_weights(raw_weights);

    assert!(sanitized.contains_key("vision_tower.timm_model.blocks.0.0.conv_exp.weight"));
    assert!(sanitized.contains_key("language_model.embed_tokens.weight"));
    assert_eq!(
        mlxcel_core::array_shape(
            sanitized
                .get("vision_tower.timm_model.blocks.0.0.conv_exp.weight")
                .unwrap()
        ),
        vec![8, 3, 3, 16]
    );
}
