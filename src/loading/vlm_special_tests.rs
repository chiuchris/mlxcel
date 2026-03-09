use super::{
    cap_molmo2_vit_num_layers, inherit_quantization_if_missing, llama4_mm_tokens_per_image,
    llama4_quantization_params, llama4_token_ids, llama4_vision_prefix, molmo2_max_crops,
    parse_molmo2_vit_layers, phi3_num_crops, rewrite_molmo2_weight_key, rewrite_phi3_weight_key,
    should_transpose_phi3_patch_embedding,
};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

#[test]
fn rewrite_phi3_weight_key_skips_position_ids_and_maps_known_prefixes() {
    assert_eq!(
        rewrite_phi3_weight_key("model.embed_tokens.weight"),
        Some("model.embed_tokens.weight".to_string())
    );
    assert_eq!(
        rewrite_phi3_weight_key(
            "model.vision_embed_tokens.img_processor.vision_model.embeddings.patch_embedding.weight"
        ),
        Some("vision_tower.vision_model.embeddings.patch_embedding.weight".to_string())
    );
    assert_eq!(
        rewrite_phi3_weight_key("model.vision_embed_tokens.img_projection.0.weight"),
        Some("img_projection.0.weight".to_string())
    );
    assert_eq!(
        rewrite_phi3_weight_key("model.vision_embed_tokens.glb_GN"),
        Some("glb_GN".to_string())
    );
    assert_eq!(rewrite_phi3_weight_key("model.position_ids"), None);
}

#[test]
fn phi3_patch_embedding_transpose_detection_matches_layout_expectations() {
    assert!(!should_transpose_phi3_patch_embedding(&[1024, 14, 14, 3]));
    assert!(should_transpose_phi3_patch_embedding(&[14, 14, 3, 1024]));
    assert!(!should_transpose_phi3_patch_embedding(&[1024, 196]));
}

#[test]
fn phi3_num_crops_prefers_preprocessor_then_config_then_default() {
    assert_eq!(
        phi3_num_crops(
            &json!({"vision_config": {"num_crops": 8}}),
            Some(&json!({"num_crops": 4}))
        ),
        4
    );
    assert_eq!(
        phi3_num_crops(
            &json!({"vision_config": {"num_crops": 8}}),
            Some(&json!({}))
        ),
        4
    );
    assert_eq!(
        phi3_num_crops(&json!({"vision_config": {"num_crops": 8}}), None),
        8
    );
    assert_eq!(phi3_num_crops(&json!({}), None), 16);
}

#[test]
fn molmo2_helpers_clamp_layer_count_and_parse_defaults() {
    assert_eq!(cap_molmo2_vit_num_layers(27), 25);
    assert_eq!(cap_molmo2_vit_num_layers(12), 12);
    assert_eq!(parse_molmo2_vit_layers(&json!({})), vec![-3, -9]);
    assert_eq!(
        parse_molmo2_vit_layers(&json!({"vit_layers": [-1, -7, 3]})),
        vec![-1, -7, 3]
    );
}

#[test]
fn rewrite_molmo2_weight_key_maps_text_vision_and_lm_head_prefixes() {
    assert_eq!(
        rewrite_molmo2_weight_key("model.transformer.layers.0.self_attn.q_proj.weight"),
        "language_model.model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        rewrite_molmo2_weight_key(
            "model.vision_backbone.transformer.resblocks.0.attn.q_proj.weight"
        ),
        "vision_tower.transformer.0.attn.q_proj.weight"
    );
    assert_eq!(
        rewrite_molmo2_weight_key("lm_head.weight"),
        "language_model.lm_head.weight"
    );
}

#[test]
fn molmo2_max_crops_uses_default_when_preprocessor_is_missing() {
    assert_eq!(molmo2_max_crops(None), 8);
    assert_eq!(molmo2_max_crops(Some(&json!({"max_crops": 12}))), 12);
}

#[test]
fn inherit_quantization_if_missing_copies_top_level_quantization_once() {
    let mut text_config = json!({
        "hidden_size": 4096
    });
    let full_config = json!({
        "quantization": {"group_size": 128, "bits": 8}
    });

    inherit_quantization_if_missing(&mut text_config, &full_config);
    assert_eq!(text_config["quantization"]["group_size"], 128);
    assert_eq!(text_config["quantization"]["bits"], 8);

    let mut explicit = json!({
        "quantization": {"group_size": 64, "bits": 4}
    });
    inherit_quantization_if_missing(&mut explicit, &full_config);
    assert_eq!(explicit["quantization"]["group_size"], 64);
    assert_eq!(explicit["quantization"]["bits"], 4);
}

#[test]
fn llama4_helpers_cover_prefix_detection_defaults_and_token_math() {
    let mut vision_model_weights = WeightMap::new();
    vision_model_weights.insert(
        "vision_model.patch_embedding.linear.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );
    assert_eq!(llama4_vision_prefix(&vision_model_weights), "vision_model");

    let mut tower_weights = WeightMap::new();
    tower_weights.insert(
        "vision_tower.patch_embedding.linear.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );
    assert_eq!(llama4_vision_prefix(&tower_weights), "vision_tower");

    assert_eq!(llama4_quantization_params(&json!({})), (64, 4));
    assert_eq!(
        llama4_quantization_params(&json!({"quantization": {"group_size": 128, "bits": 6}})),
        (128, 6)
    );
    assert_eq!(llama4_token_ids(&json!({})), (200092, 200018));
    assert_eq!(
        llama4_token_ids(&json!({"image_token_index": 7, "text_config": {"pad_token_id": 9}})),
        (7, 9)
    );
}

#[test]
fn llama4_mm_tokens_per_image_applies_pixel_shuffle_ratio() {
    let config: crate::vision::encoders::llama4::Llama4VisionConfig =
        serde_json::from_value(json!({
            "hidden_size": 1024,
            "image_size": 1120,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "patch_size": 14,
            "pixel_shuffle_ratio": 0.5
        }))
        .unwrap();

    assert_eq!(llama4_mm_tokens_per_image(&config), 1600);
}
