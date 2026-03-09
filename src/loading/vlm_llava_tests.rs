use super::{
    LlavaTextBackend, detect_bunny_text_backend, infer_llama_config_from_weights,
    inherit_text_quantization_if_missing, llava_text_backend, parse_bunny_vision_config,
    rewrite_bunny_weight_key,
};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

fn test_weight_map() -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::ones(&[256], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        mlxcel_core::ones(&[512, 256], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.2.self_attn.q_proj.weight".to_string(),
        mlxcel_core::ones(&[256, 256], dtype::FLOAT32),
    );
    weights
}

#[test]
fn infer_llama_config_from_weights_populates_missing_fields() {
    let mut config = json!({});

    infer_llama_config_from_weights(&mut config, &test_weight_map());

    assert_eq!(config["hidden_size"], 256);
    assert_eq!(config["num_hidden_layers"], 3);
    assert_eq!(config["intermediate_size"], 512);
    assert_eq!(config["num_attention_heads"], 2);
}

#[test]
fn infer_llama_config_from_weights_preserves_existing_values() {
    let mut config = json!({
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "intermediate_size": 4096,
        "num_attention_heads": 16
    });

    infer_llama_config_from_weights(&mut config, &test_weight_map());

    assert_eq!(config["hidden_size"], 1024);
    assert_eq!(config["num_hidden_layers"], 8);
    assert_eq!(config["intermediate_size"], 4096);
    assert_eq!(config["num_attention_heads"], 16);
}

#[test]
fn rewrite_bunny_weight_key_rewrites_expected_prefixes() {
    assert_eq!(
        rewrite_bunny_weight_key("vision_tower.vision_tower.vision_model.encoder.layers.0"),
        Some("vision_tower.vision_model.encoder.layers.0".to_string())
    );
    assert_eq!(
        rewrite_bunny_weight_key("mm_projector.0.weight"),
        Some("multi_modal_projector.0.weight".to_string())
    );
    assert_eq!(
        rewrite_bunny_weight_key("model.lm_head.weight"),
        Some("lm_head.weight".to_string())
    );
    assert_eq!(
        rewrite_bunny_weight_key("model.layers.0.self_attn.q_proj"),
        None
    );
}

#[test]
fn detect_bunny_text_backend_prefers_nested_text_config_then_top_level_hint() {
    assert_eq!(
        detect_bunny_text_backend(&json!({
            "text_config": { "model_type": "qwen2" },
            "model_type": "llama"
        })),
        LlavaTextBackend::Qwen2
    );
    assert_eq!(
        detect_bunny_text_backend(&json!({
            "model_type": "bunny-qwen2"
        })),
        LlavaTextBackend::Qwen2
    );
    assert_eq!(
        detect_bunny_text_backend(&json!({
            "model_type": "bunny-llama"
        })),
        LlavaTextBackend::Llama
    );
}

#[test]
fn llava_text_backend_rejects_unknown_backends() {
    let err = llava_text_backend("gemma", "LLaVA").unwrap_err();
    assert!(err.to_string().contains("Unsupported LLaVA text backend"));
}

#[test]
fn parse_bunny_vision_config_uses_defaults_for_sparse_configs() {
    let config = parse_bunny_vision_config(
        &json!({
            "mm_hidden_size": 1408
        }),
        &json!({
            "intermediate_size": 5120
        }),
    );

    assert_eq!(config.model_type, "siglip_vision_model");
    assert_eq!(config.hidden_size, 1408);
    assert_eq!(config.intermediate_size, 5120);
    assert_eq!(config.patch_size, 14);
    assert_eq!(config.image_size, 384);
}

#[test]
fn inherit_text_quantization_if_missing_rejects_non_object_configs() {
    let mut text_config = json!("bad");
    let full_config = json!({
        "quantization": {"group_size": 128, "bits": 8}
    });

    let err = inherit_text_quantization_if_missing(&mut text_config, &full_config)
        .unwrap_err()
        .to_string();
    assert!(err.contains("LLaVA text_config"));
}

#[test]
fn parse_bunny_vision_config_preserves_full_configs() {
    let config = parse_bunny_vision_config(
        &json!({}),
        &json!({
            "model_type": "siglip_vision_model",
            "num_hidden_layers": 2,
            "hidden_size": 768,
            "intermediate_size": 1536,
            "num_attention_heads": 12,
            "patch_size": 16,
            "image_size": 224,
            "num_channels": 3,
            "layer_norm_eps": 1e-5
        }),
    );

    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.intermediate_size, 1536);
    assert_eq!(config.patch_size, 16);
    assert_eq!(config.image_size, 224);
}
