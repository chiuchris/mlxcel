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

use super::{
    cap_molmo2_vit_num_layers, dequantize_moondream3_weight, inherit_quantization_if_missing,
    llama4_mm_tokens_per_image, llama4_quantization_params, llama4_token_ids, llama4_vision_prefix,
    molmo2_max_crops, moondream3_text_config_value, moondream3_vision_config_value,
    parse_molmo2_vit_layers, phi3_num_crops, phi4_siglip_text_config_value,
    phi4mm_text_config_value, phi4mm_vision_config_value, remap_minicpmo_text_weights,
    rewrite_molmo2_weight_key, rewrite_moondream3_weight_key, rewrite_phi3_weight_key,
    rewrite_phi4_siglip_weight_key, rewrite_phi4mm_vision_key,
    should_transpose_phi3_patch_embedding,
};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

#[test]
fn remap_minicpmo_text_weights_strips_language_model_prefix() {
    let mut weights = WeightMap::new();
    weights.insert(
        "language_model.model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );
    weights.insert(
        "language_model.lm_head.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );
    weights.insert(
        "vision_tower.embeddings.patch_embedding.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );

    let remapped = remap_minicpmo_text_weights(&weights);
    assert!(remapped.contains_key("model.embed_tokens.weight"));
    assert!(remapped.contains_key("lm_head.weight"));
    assert!(remapped.contains_key("vision_tower.embeddings.patch_embedding.weight"));
}

#[test]
fn rewrite_moondream3_weight_key_strips_model_prefix_and_skips_region_branch() {
    assert_eq!(
        rewrite_moondream3_weight_key("model.text.wte"),
        Some("text.wte.weight".to_string())
    );
    assert_eq!(
        rewrite_moondream3_weight_key("model.text.blocks.4.attn.qkv.weight.packed"),
        Some("text.blocks.4.attn.qkv.weight.packed".to_string())
    );
    assert_eq!(
        rewrite_moondream3_weight_key("model.region.coord_encoder.weight"),
        None
    );
}

#[test]
fn moondream3_text_and_vision_config_helpers_fill_default_shapes() {
    let text = moondream3_text_config_value(&json!({
        "text_group_size": 64,
        "expert_group_size": 32,
        "quantization_config": {"quant_method": "int4"}
    }));
    let vision = moondream3_vision_config_value(&json!({}));

    assert_eq!(text["model_type"], "moondream3");
    assert_eq!(text["group_size"], 64);
    assert_eq!(text["moe"]["expert_group_size"], 32);
    assert_eq!(text["bits"], 4);
    assert_eq!(vision["crop_size"], 378);
    assert_eq!(vision["enc_patch_size"], 14);
}

#[test]
fn dequantize_moondream3_weight_restores_interleaved_uint4_rows() {
    let mut packed_bytes = [0u8; 128];
    packed_bytes[0] = 0x1F;
    packed_bytes[1] = 0x20;
    let packed_i32: Vec<i32> = packed_bytes.iter().map(|&value| value as i32).collect();
    let packed = mlxcel_core::from_slice_i32(&packed_i32, &[1, 128]);
    let packed = mlxcel_core::astype(&packed, dtype::UINT8);
    let scale = mlxcel_core::ones(&[2, 1], dtype::FLOAT32);
    let zero = mlxcel_core::zeros(&[2, 1], dtype::FLOAT32);

    let dequantized = dequantize_moondream3_weight(&packed, &scale, &zero, &[2, 128]);
    assert_eq!(mlxcel_core::array_shape(&dequantized), vec![2, 128]);
    let total = mlxcel_core::sum_all(&dequantized);
    mlxcel_core::eval(&total);
    assert!(mlxcel_core::item_f32(&total) > 0.0);
}

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
fn rewrite_phi4_siglip_weight_key_keeps_text_keys_and_remaps_multimodal_prefixes() {
    assert_eq!(
        rewrite_phi4_siglip_weight_key("model.layers.0.self_attn.qkv_proj.weight"),
        Some("model.layers.0.self_attn.qkv_proj.weight".to_string())
    );
    assert_eq!(
        rewrite_phi4_siglip_weight_key(
            "model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight"
        ),
        Some(
            "vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight".to_string()
        )
    );
    assert_eq!(
        rewrite_phi4_siglip_weight_key("model.mm_projector.0.weight"),
        Some("mm_projector_linear1.weight".to_string())
    );
    assert_eq!(rewrite_phi4_siglip_weight_key("model.position_ids"), None);
}

#[test]
fn phi4_siglip_text_config_value_inherits_root_text_fields() {
    let text_config = phi4_siglip_text_config_value(&json!({
        "model_type": "phi4-siglip",
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "intermediate_size": 17920,
        "vocab_size": 100352,
        "rope_theta": 500000.0,
        "quantization": {"group_size": 64, "bits": 4},
        "vision_config": {"hidden_size": 1152}
    }))
    .unwrap();

    assert_eq!(text_config["hidden_size"], 5120);
    assert_eq!(text_config["num_attention_heads"], 40);
    assert_eq!(text_config["quantization"]["group_size"], 64);
}

#[test]
fn rewrite_phi4mm_vision_key_maps_multimodal_prefixes_and_skips_audio() {
    assert_eq!(
        rewrite_phi4mm_vision_key(
            "model.embed_tokens_extend.image_embed.img_processor.embeddings.patch_embedding.weight"
        ),
        Some(
            "vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight".to_string()
        )
    );
    assert_eq!(
        rewrite_phi4mm_vision_key("model.embed_tokens_extend.image_embed.img_projection.0.weight"),
        Some("mm_projector_linear1.weight".to_string())
    );
    assert_eq!(
        rewrite_phi4mm_vision_key("model.layers.0.self_attn.qkv_proj.base_layer.weight"),
        Some("model.layers.0.self_attn.qkv_proj.base_layer.weight".to_string())
    );
    assert_eq!(
        rewrite_phi4mm_vision_key(
            "model.embed_tokens_extend.audio_embed.audio_projection.speech.0.weight"
        ),
        None
    );
}

#[test]
fn phi4mm_text_config_value_inherits_root_text_fields() {
    let text_config = phi4mm_text_config_value(&json!({
        "model_type": "phi4mm",
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_hidden_layers": 32,
        "intermediate_size": 8192,
        "vocab_size": 200064,
        "partial_rotary_factor": 0.75,
        "tie_word_embeddings": true
    }))
    .unwrap();

    assert_eq!(text_config["model_type"], "phi4mm");
    assert_eq!(text_config["partial_rotary_factor"], 0.75);
    assert_eq!(text_config["tie_word_embeddings"], true);
}

#[test]
fn phi4mm_vision_config_value_uses_crop_size_defaults() {
    let vision_config = phi4mm_vision_config_value(&json!({
        "embd_layer": {
            "image_embd_layer": {
                "crop_size": 448
            }
        }
    }));

    assert_eq!(vision_config["patch_size"], 14);
    assert_eq!(vision_config["image_size"], 448);
    assert_eq!(vision_config["num_patches"], 1024);
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

    inherit_quantization_if_missing(&mut text_config, &full_config).unwrap();
    assert_eq!(text_config["quantization"]["group_size"], 128);
    assert_eq!(text_config["quantization"]["bits"], 8);

    let mut explicit = json!({
        "quantization": {"group_size": 64, "bits": 4}
    });
    inherit_quantization_if_missing(&mut explicit, &full_config).unwrap();
    assert_eq!(explicit["quantization"]["group_size"], 64);
    assert_eq!(explicit["quantization"]["bits"], 4);
}

#[test]
fn inherit_quantization_if_missing_rejects_non_object_text_config() {
    let mut text_config = json!(5);
    let full_config = json!({
        "quantization": {"group_size": 128, "bits": 8}
    });

    let err = inherit_quantization_if_missing(&mut text_config, &full_config)
        .unwrap_err()
        .to_string();
    assert!(err.contains("special VLM text_config"));
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
