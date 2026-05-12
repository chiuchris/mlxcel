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

//! Unit tests for the Gemma 4 MTP assistant drafter (`Gemma4AssistantDraftModel`).
//!
//! The tests build a deliberately tiny synthetic config (`hidden_size = 64`,
//! `num_hidden_layers = 2`) and feed in zero-initialised weights so that the
//! whole load / forward / sample path can be exercised on a unit-test budget.
//! Output magnitude is not asserted — the tests verify shapes, lifecycle
//! ordering, and error propagation.

use super::config::{DrafterRopeParameters, DrafterTextConfig, Gemma4AssistantConfig};
use super::model::Gemma4AssistantDraftModel;
use crate::drafter::{Drafter, DrafterError, DrafterKind, SharedKv};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::collections::HashMap;

/// Minimal drafter config used by every test. Uses `hidden_size=64`, 2 layers
/// in a SWA / full pattern, head_dim=32 → 2 heads, all KV-shared. Field
/// defaults match upstream Python where applicable.
fn make_test_config(num_hidden_layers: usize, tie_word_embeddings: bool) -> Gemma4AssistantConfig {
    let mut rope_parameters = HashMap::new();
    rope_parameters.insert(
        "full_attention".to_string(),
        DrafterRopeParameters {
            rope_theta: 1_000_000.0,
            partial_rotary_factor: 1.0,
            rope_type: "proportional".to_string(),
        },
    );
    rope_parameters.insert(
        "sliding_attention".to_string(),
        DrafterRopeParameters {
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rope_type: "default".to_string(),
        },
    );
    let layer_types: Vec<String> = (0..num_hidden_layers)
        .map(|i| {
            if i + 1 == num_hidden_layers {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect();
    let text_config = DrafterTextConfig {
        model_type: "gemma4_text".into(),
        hidden_size: 64,
        num_hidden_layers,
        intermediate_size: 128,
        num_attention_heads: 2,
        head_dim: 32,
        global_head_dim: None,
        rms_norm_eps: 1e-6,
        vocab_size: 16,
        num_key_value_heads: 1,
        num_global_key_value_heads: None,
        num_kv_shared_layers: num_hidden_layers,
        rope_parameters,
        sliding_window: 64,
        sliding_window_pattern: 2,
        max_position_embeddings: 256,
        layer_types,
        attention_k_eq_v: false,
        final_logit_softcapping: None,
        use_double_wide_mlp: false,
        quantization: None,
    };
    Gemma4AssistantConfig {
        model_type: "gemma4_assistant".into(),
        backbone_hidden_size: 32,
        use_ordered_embeddings: false,
        num_centroids: 8,
        centroid_intermediate_top_k: 2,
        tie_word_embeddings,
        block_size: 4,
        target_layer_ids: vec![],
        target_layer_types: vec![],
        text_config: Some(text_config),
    }
    .normalize()
    .expect("normalize")
}

/// Build a fp32 zero-initialised tensor of the given shape.
fn zeros_f32(shape: &[i32]) -> UniquePtr<MlxArray> {
    ffi::zeros(shape, crate::dtype::FLOAT32)
}

/// Insert a fp32 zero-initialised tensor at the given weight-map key.
fn insert_zeros(weights: &mut WeightMap, key: &str, shape: &[i32]) {
    weights.insert(key.to_string(), zeros_f32(shape));
}

/// Build a complete weight map for the test config above. Every tensor is
/// fp32 zeros — values are not asserted by the tests, only shapes and
/// lifecycle.
fn make_test_weights(config: &Gemma4AssistantConfig) -> WeightMap {
    let tc = config.text_config();
    let hidden = tc.hidden_size as i32;
    let inter = tc.intermediate_size as i32;
    let vocab = tc.vocab_size as i32;
    let n_heads = tc.num_attention_heads as i32;
    let head_dim = tc.head_dim as i32;
    let n_kv = tc.num_key_value_heads as i32;
    let backbone = config.backbone_hidden_size as i32;

    let mut w = WeightMap::new();

    // model.embed_tokens.weight: [vocab, hidden].
    insert_zeros(&mut w, "model.embed_tokens.weight", &[vocab, hidden]);

    // model.norm.weight: [hidden].
    insert_zeros(&mut w, "model.norm.weight", &[hidden]);

    // pre_projection.weight: [hidden, 2 * backbone] (Linear.forward
    // transposes weight, so weight shape is [out, in] = [hidden,
    // 2*backbone]).
    insert_zeros(&mut w, "pre_projection.weight", &[hidden, 2 * backbone]);

    // post_projection.weight: [backbone, hidden].
    insert_zeros(&mut w, "post_projection.weight", &[backbone, hidden]);

    if !config.tie_word_embeddings {
        // lm_head.weight: [vocab, hidden].
        insert_zeros(&mut w, "lm_head.weight", &[vocab, hidden]);
    }

    // Per-layer weights.
    for i in 0..tc.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let kv_dim = n_kv * head_dim;
        // Q proj: out = n_heads * head_dim
        insert_zeros(
            &mut w,
            &format!("{p}.self_attn.q_proj.weight"),
            &[n_heads * head_dim, hidden],
        );
        let _ = kv_dim; // KV-shared layer never has its own K/V proj.
        insert_zeros(&mut w, &format!("{p}.self_attn.q_norm.weight"), &[head_dim]);
        // o_proj: out = hidden, in = n_heads * head_dim
        insert_zeros(
            &mut w,
            &format!("{p}.self_attn.o_proj.weight"),
            &[hidden, n_heads * head_dim],
        );

        // MLP
        insert_zeros(
            &mut w,
            &format!("{p}.mlp.gate_proj.weight"),
            &[inter, hidden],
        );
        insert_zeros(
            &mut w,
            &format!("{p}.mlp.up_proj.weight"),
            &[inter, hidden],
        );
        insert_zeros(
            &mut w,
            &format!("{p}.mlp.down_proj.weight"),
            &[hidden, inter],
        );

        // Norms
        insert_zeros(&mut w, &format!("{p}.input_layernorm.weight"), &[hidden]);
        insert_zeros(
            &mut w,
            &format!("{p}.post_attention_layernorm.weight"),
            &[hidden],
        );
        insert_zeros(
            &mut w,
            &format!("{p}.pre_feedforward_layernorm.weight"),
            &[hidden],
        );
        insert_zeros(
            &mut w,
            &format!("{p}.post_feedforward_layernorm.weight"),
            &[hidden],
        );
    }

    w
}

/// Mock `LanguageModel` for `bind()` testing. Only `embed_tokens` is
/// meaningful; everything else falls through to the trait defaults or
/// `unreachable!()` panics if the drafter inadvertently calls them.
struct MockLanguageModel {
    hidden_size: i32,
}

impl MockLanguageModel {
    fn new(_vocab_size: i32, hidden_size: i32) -> Self {
        Self { hidden_size }
    }
}

impl LanguageModel for MockLanguageModel {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [crate::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        unreachable!("draft tests do not invoke target forward")
    }

    fn make_caches(&self) -> Vec<crate::layers::KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        0
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        // Simulate the gemma4 target's embed_tokens: gather rows from
        // `embed_weight` at the given `input_ids`. For unit tests we don't
        // need correctness of the lookup — just a non-None return so the
        // drafter accepts the bind.
        let shape = ffi::array_shape(input_ids);
        let b = shape[0];
        let l = shape[1];
        Some(ffi::zeros(
            &[b, l, self.hidden_size],
            crate::dtype::FLOAT32,
        ))
    }
}

// ── Sanitize tests ────────────────────────────────────────────────────────

#[test]
fn sanitize_drops_lm_head_when_tied() {
    let cfg = make_test_config(2, true);
    let mut weights = make_test_weights(&cfg);
    // Add an extra lm_head.weight as if the checkpoint shipped one despite
    // tie_word_embeddings=true (upstream Python explicitly handles this case).
    insert_zeros(&mut weights, "lm_head.weight", &[16, 64]);
    assert!(weights.contains_key("lm_head.weight"));
    Gemma4AssistantDraftModel::sanitize_weights(&mut weights, &cfg);
    assert!(
        !weights.contains_key("lm_head.weight"),
        "sanitize must drop lm_head.weight when tie_word_embeddings=true"
    );
}

#[test]
fn sanitize_keeps_lm_head_when_not_tied() {
    let cfg = make_test_config(2, false);
    let mut weights = make_test_weights(&cfg);
    assert!(weights.contains_key("lm_head.weight"));
    Gemma4AssistantDraftModel::sanitize_weights(&mut weights, &cfg);
    assert!(
        weights.contains_key("lm_head.weight"),
        "sanitize must keep lm_head.weight when tie_word_embeddings=false"
    );
}

// ── Construction / weight-loading tests ──────────────────────────────────

#[test]
fn from_weights_loads_tied_dense_drafter() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let model = Gemma4AssistantDraftModel::from_weights(weights, cfg);
    assert!(model.is_ok(), "tied-dense drafter must load: {:?}", model.err());
}

#[test]
fn from_weights_loads_non_tied_drafter_with_explicit_lm_head() {
    let cfg = make_test_config(2, false);
    let weights = make_test_weights(&cfg);
    let model = Gemma4AssistantDraftModel::from_weights(weights, cfg);
    assert!(
        model.is_ok(),
        "non-tied drafter must load with explicit lm_head: {:?}",
        model.err()
    );
}

#[test]
fn from_weights_errors_on_missing_pre_projection() {
    let cfg = make_test_config(2, true);
    let mut weights = make_test_weights(&cfg);
    weights.remove("pre_projection.weight");
    let err = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect_err("must fail");
    match err {
        DrafterError::WeightLoad { reason } => {
            assert!(
                reason.contains("pre_projection"),
                "error must point at the missing pre_projection: {reason}"
            );
        }
        other => panic!("expected WeightLoad, got {other:?}"),
    }
}

// ── Drafter trait surface tests ──────────────────────────────────────────

#[test]
fn drafter_kind_returns_mtp() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    assert_eq!(model.kind(), DrafterKind::Mtp);
}

#[test]
fn make_cache_returns_empty_vec() {
    // MTP drafter has no own KV cache (matches upstream Python
    // `Gemma4AssistantDraftModel.make_cache(self) -> []`).
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    assert!(model.make_cache().is_empty());
}

#[test]
fn draft_block_rejects_call_before_set_shared_kv() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    let target = MockLanguageModel::new(16, 64);
    model.bind(&target).expect("bind");

    let sampler = SamplingConfig::greedy();
    let err = model.draft_block(0, None, 4, &sampler).expect_err("must fail");
    match err {
        DrafterError::SetSharedKvNotCalled => {}
        other => panic!("expected SetSharedKvNotCalled, got {other:?}"),
    }
}

#[test]
fn draft_block_rejects_call_before_bind() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");

    // Skip bind, set_shared_kv first (the order doesn't matter; both
    // pre-conditions must hold). draft_block must fail before it runs.
    let sampler = SamplingConfig::greedy();
    let err = model.draft_block(0, None, 4, &sampler).expect_err("must fail");
    // Order check: set_shared_kv runs first inside draft_block, so the
    // first guard to trigger is SetSharedKvNotCalled. After that fixes,
    // BindNotCalled fires. The test pins the first-failure ordering.
    match err {
        DrafterError::SetSharedKvNotCalled => {}
        other => panic!("expected SetSharedKvNotCalled, got {other:?}"),
    }
}

#[test]
fn bind_rejects_target_without_embed_tokens_override() {
    /// LanguageModel that does NOT override embed_tokens (returns None).
    struct BareTarget;
    impl LanguageModel for BareTarget {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [crate::layers::KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            unreachable!()
        }
        fn make_caches(&self) -> Vec<crate::layers::KVCache> {
            Vec::new()
        }
        fn num_layers(&self) -> usize {
            0
        }
        fn eos_token_ids(&self) -> Vec<i32> {
            Vec::new()
        }
        // No `embed_tokens` override; falls through to the trait default
        // which returns None.
    }

    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");

    let target = BareTarget;
    let err = model.bind(&target).expect_err("must fail");
    match err {
        DrafterError::TargetMissingFeature { feature } => {
            assert_eq!(feature, "embed_tokens");
        }
        other => panic!("expected TargetMissingFeature, got {other:?}"),
    }
}

// ── Centroid LM head gating (sibling PR #627) ────────────────────────────

#[test]
fn bind_rejects_centroid_lm_head_until_627_lands() {
    // E2B / E4B drafters: `use_ordered_embeddings=true` selects the centroid
    // (MaskedEmbedder) LM head. Until #627 lands, this path returns
    // NotYetImplemented { issue: 627 } so callers get a clear actionable
    // message.
    let mut cfg = make_test_config(2, true);
    cfg.use_ordered_embeddings = true;
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");

    let target = MockLanguageModel::new(16, 64);
    let err = model.bind(&target).expect_err("must fail until #627 lands");
    match err {
        DrafterError::NotYetImplemented { kind, issue } => {
            assert_eq!(kind, DrafterKind::Mtp);
            assert_eq!(issue, 627);
        }
        other => panic!("expected NotYetImplemented(627), got {other:?}"),
    }
}

// ── SharedKv shape validation ────────────────────────────────────────────

#[test]
fn set_shared_kv_rejects_unexpected_tensor_count() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let mut model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");

    let target = MockLanguageModel::new(16, 64);
    model.bind(&target).expect("bind");

    // 3 tensors — not a valid count for Gemma 4 shared K/V.
    let t0 = ffi::zeros(&[1, 1, 2, 32], crate::dtype::FLOAT32);
    let t1 = ffi::zeros(&[1, 1, 2, 32], crate::dtype::FLOAT32);
    let t2 = ffi::zeros(&[1, 1, 2, 32], crate::dtype::FLOAT32);
    let tensors: Vec<&MlxArray> = vec![
        t0.as_ref().unwrap(),
        t1.as_ref().unwrap(),
        t2.as_ref().unwrap(),
    ];
    let shared = SharedKv::new(&tensors);
    let err = model.set_shared_kv(shared, 0, 0, 0).expect_err("must fail");
    match err {
        DrafterError::SharedKvShape { got, expected } => {
            assert_eq!(got, 3);
            assert_eq!(expected, &[2, 4]);
        }
        other => panic!("expected SharedKvShape, got {other:?}"),
    }
}

// ── Object-safety / dispatch test ────────────────────────────────────────

#[test]
fn gemma4_assistant_drafter_is_object_safe_via_box_dyn() {
    let cfg = make_test_config(2, true);
    let weights = make_test_weights(&cfg);
    let model = Gemma4AssistantDraftModel::from_weights(weights, cfg).expect("load");
    let boxed: Box<dyn Drafter> = Box::new(model);
    assert_eq!(boxed.kind(), DrafterKind::Mtp);
}
