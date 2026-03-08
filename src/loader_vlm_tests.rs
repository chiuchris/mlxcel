use super::{
    QwenVisionTokenIds, inherit_qwen_vision_quantization, qwen_vl_token_ids,
    rewrite_qwen3_vl_weight_key,
};
use crate::vision::encoders::qwen3_vl::Qwen3VLVisionConfig;
use serde_json::json;

#[test]
fn inherit_qwen_vision_quantization_uses_top_level_defaults() {
    let mut vision_config: Qwen3VLVisionConfig = serde_json::from_value(json!({
        "hidden_size": 1536
    }))
    .unwrap();
    let full_config = json!({
        "quantization": {
            "group_size": 128,
            "bits": 8
        }
    });

    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    assert_eq!(vision_config.quant_group_size, 128);
    assert_eq!(vision_config.quant_bits, 8);
}

#[test]
fn inherit_qwen_vision_quantization_preserves_existing_values() {
    let mut vision_config: Qwen3VLVisionConfig = serde_json::from_value(json!({
        "hidden_size": 1536,
        "quant_group_size": 32,
        "quant_bits": 6
    }))
    .unwrap();
    let full_config = json!({
        "quantization": {
            "group_size": 128,
            "bits": 8
        }
    });

    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    assert_eq!(vision_config.quant_group_size, 32);
    assert_eq!(vision_config.quant_bits, 6);
}

#[test]
fn rewrite_qwen3_vl_weight_key_rewrites_language_and_visual_prefixes() {
    assert_eq!(
        rewrite_qwen3_vl_weight_key(
            "model.language_model.layers.0.self_attn.q_proj.weight".into(),
            false
        ),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        rewrite_qwen3_vl_weight_key("model.visual.blocks.0.attn.qkv.weight".into(), false),
        "vision_tower.blocks.0.attn.qkv.weight"
    );
    assert_eq!(
        rewrite_qwen3_vl_weight_key("language_model.lm_head.weight".into(), false),
        "lm_head.weight"
    );
}

#[test]
fn rewrite_qwen3_vl_weight_key_sanitizes_moe_expert_weights() {
    assert_eq!(
        rewrite_qwen3_vl_weight_key(
            "model.language_model.layers.0.mlp.experts.up_proj".into(),
            true
        ),
        "model.layers.0.mlp.switch_mlp.up_proj.weight"
    );
    assert_eq!(
        rewrite_qwen3_vl_weight_key(
            "model.language_model.layers.0.mlp.experts.up_proj.weight".into(),
            true
        ),
        "model.layers.0.mlp.switch_mlp.up_proj.weight"
    );
}

#[test]
fn qwen_vl_token_ids_applies_defaults_and_overrides() {
    let defaults = QwenVisionTokenIds {
        image_token_id: 10,
        video_token_id: 11,
        vision_start_token_id: 12,
    };

    let ids = qwen_vl_token_ids(
        &json!({
            "image_token_id": 20,
            "vision_start_token_id": 22
        }),
        defaults,
    );

    assert_eq!(
        ids,
        QwenVisionTokenIds {
            image_token_id: 20,
            video_token_id: 11,
            vision_start_token_id: 22,
        }
    );
}
