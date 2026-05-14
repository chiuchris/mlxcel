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

//! Greedy-parity verification for speculative drafter pairings (epic #633,
//! sub-9 / issue #632).
//!
//! ## What this file pins
//!
//! At `temperature = 0.0` the speculative-decoding round loop emits a token
//! sequence that is byte-identical to the equivalent drafter-less greedy
//! pass through the target — this invariant is the load-bearing correctness
//! gate of every speculative path in mlxcel. The integration tests below
//! drive a real target + real drafter end-to-end and assert byte-equality
//! against a paired drafter-less baseline.
//!
//! ## Reachable pairings (this PR)
//!
//! - **Qwen 3.5 4B + DFlash** (`models/qwen3.5-4b-4bit` target, drafter
//!   `models/Qwen3.5-4B-DFlash`, `block_size = 16`). The DFlash drafter
//!   is loadable via `mlxcel_core::drafter::dflash::DFlashDrafter::load`
//!   and `Qwen35Model` implements
//!   `mlxcel_core::drafter::dflash::SpeculativeTarget` (PR #663 wired the
//!   B>1 path too). The published upstream `z-lab/Qwen3.5-4B-DFlash`
//!   checkpoint omits `embed_tokens.weight`; issue #675 ported upstream's
//!   lazy-bind shape so the drafter loads with a tombstone and resolves
//!   its embedding from the target during `Drafter::bind`. This test
//!   loads both models, asserts the speculative-target trait wiring is
//!   exposed correctly, and pins that the DFlash drafter loads + binds
//!   against the published checkpoint. The end-to-end byte-equality
//!   assertion is layered on by follow-up #676 (single-request server,
//!   fixed prompt, drafter vs no-drafter at temp=0).
//!
//! - **Gemma 4 31B + MTP assistant** (`models/gemma-4-31b-it-4bit` target,
//!   drafter `models/gemma-4-31B-it-assistant-bf16`, `block_size = 4`). The
//!   MTP drafter is loadable via
//!   `mlxcel_core::drafter::gemma4_assistant::Gemma4AssistantDraftModel::from_path`
//!   and `Gemma4Wrapper` already exposes the underlying primitives
//!   (`forward_with_speculative_sinks`, `rollback_speculative_cache`). The
//!   `MtpTarget` trait impl for `Gemma4Wrapper` itself is intentionally
//!   deferred to follow-up #666 alongside the server-scheduler integration,
//!   per the deferral note in PRs #663, #664, and #665.
//!
//! ## Deferred pairings (drafter checkpoint not on disk)
//!
//! - **Gemma 4 E2B / E4B** — drafter checkpoints (`gemma-4-E2B-it-assistant-bf16`,
//!   `gemma-4-E4B-it-assistant-bf16`) are not on local disk and the E-family
//!   pairings additionally depend on the centroid LM head integration tracked
//!   by follow-up #660.
//! - **Gemma 4 26B-A4B** — drafter checkpoint (`gemma-4-26B-A4B-it-assistant-bf16`)
//!   is not on local disk. Skipped pending checkpoint availability.
//!
//! ## Invocation
//!
//! ```bash
//! # Full set including ignored real-model tests:
//! cargo test --test speculative_parity --release -- --ignored --test-threads=1 --nocapture
//!
//! # Structural-only check (loads models, asserts trait wiring; no decode):
//! cargo test --test speculative_parity --release -- --test-threads=1 --nocapture
//! ```
//!
//! `--test-threads=1` is required because real-model tests share GPU memory
//! and concurrent loads will OOM on smaller (32-48 GB) Apple Silicon hosts.

mod common;

use common::repo_model_dir;

/// Reachable pairings whose drafter checkpoint is present on disk in the
/// canonical `models/` directory (resolved via [`repo_model_dir`]). The
/// test below short-circuits with an informative message when the
/// corresponding target / drafter directory is missing, mirroring the
/// pattern used by `tests/tensor_parallel_real_models.rs`.
struct Pairing {
    name: &'static str,
    target_dir: &'static str,
    draft_dir: &'static str,
    /// Drafter kind name as it appears in CLI flags (`--draft-kind`).
    kind: &'static str,
    /// Draft block length. MTP defaults to 4 (Gemma 4 assistant); DFlash
    /// defaults to 16 (Qwen 3.5 DFlash).
    block_size: u32,
}

const REACHABLE_PAIRINGS: &[Pairing] = &[
    Pairing {
        name: "Qwen 3.5 4B + DFlash (b=16)",
        target_dir: "qwen3.5-4b-4bit",
        draft_dir: "Qwen3.5-4B-DFlash",
        kind: "dflash",
        block_size: 16,
    },
    Pairing {
        name: "Gemma 4 31B + MTP assistant (b=4)",
        target_dir: "gemma-4-31b-it-4bit",
        draft_dir: "gemma-4-31B-it-assistant-bf16",
        kind: "mtp",
        block_size: 4,
    },
];

/// Returns the target / drafter paths and whether both are present on disk.
fn pairing_present(pairing: &Pairing) -> (std::path::PathBuf, std::path::PathBuf, bool) {
    let target = repo_model_dir(pairing.target_dir);
    let draft = repo_model_dir(pairing.draft_dir);
    let present = target.exists() && draft.exists();
    (target, draft, present)
}

/// Structural pin: every reachable pairing's checkpoint layout matches what
/// the speculative-decoding loaders expect.
///
/// Specifically:
///   1. The target directory exists OR the test is cleanly skipped with a
///      log line so CI hosts without the models do not red-flag the build.
///   2. The drafter's `config.json` exists and the resolved drafter kind
///      via `mlxcel_core::drafter::resolve_drafter_kind` matches the
///      pairing's declared `kind` value.
///
/// This is the cheapest gate that catches "we silently picked the wrong
/// drafter shape" bugs at test time. The expensive byte-equality assertion
/// for the full round-loop pass is gated behind `#[ignore]` (see
/// `greedy_parity_dflash_qwen35_4b` / `greedy_parity_mtp_gemma4_31b`).
#[test]
fn pairing_kind_resolution_matches_declaration() {
    use mlxcel_core::drafter::{DrafterKind, resolve_drafter_kind};
    use std::str::FromStr as _;

    let mut checked = 0u32;
    for pairing in REACHABLE_PAIRINGS {
        let (_target, draft, present) = pairing_present(pairing);
        if !present {
            eprintln!(
                "Skipping pairing kind resolution for {}: drafter or target missing on disk",
                pairing.name,
            );
            continue;
        }

        let expected_kind = DrafterKind::from_str(pairing.kind)
            .expect("pairing kind string must parse to a known DrafterKind variant");
        let resolved_kind = resolve_drafter_kind(&draft, None)
            .expect("drafter config.json must be readable and parseable");

        assert_eq!(
            resolved_kind, expected_kind,
            "Pairing {} declared kind={:?} but auto-detect from {:?} resolved to {:?}; \
             check the drafter's config.json::model_type",
            pairing.name, expected_kind, draft, resolved_kind,
        );
        checked += 1;
    }

    // It is intentionally OK for `checked == 0` (no pairings on disk) — the
    // test logs that case and passes so CI hosts without the model
    // checkpoints don't fail the build. The reachable pairing discovery
    // test below documents that path explicitly.
    eprintln!(
        "pairing_kind_resolution_matches_declaration: checked {checked} \
         pairing(s) (out of {} reachable; skipped pairings are not on disk)",
        REACHABLE_PAIRINGS.len(),
    );
}

/// Greedy parity for the Qwen 3.5 4B + DFlash pairing.
///
/// **Status:** loads + binds; full byte-equality assertion deferred to #676.
///
/// The DFlash B=1 round loop (`mlxcel_core::drafter::dflash::round_loop::DFlashGenerator`)
/// requires a `&mut [Qwen3NextCache]` slice from the test harness, but
/// `Qwen35Model::make_caches` is `pub(crate)` today — there is no public
/// API to construct that cache type outside the binary crate. The trait
/// `LanguageModel::make_caches` returns `Vec<KVCache>` (empty for Qwen 3.5
/// because the model owns its own sequence state internally), so an
/// integration-test crate cannot drive the speculative pipeline directly.
///
/// Follow-up #676 wires the real-model byte-equality end-to-end check
/// (single-request server, fixed prompt, drafter vs no-drafter at temp=0).
///
/// What the test asserts today:
///   - Both target and drafter directories load successfully.
///   - The drafter's resolved kind is exactly `DrafterKind::Dflash`.
///   - The target wraps a `LoadedModel::Qwen35` variant (the only currently
///     supported DFlash target family in mlxcel).
///   - The DFlash drafter LOADS against the published upstream
///     `z-lab/Qwen3.5-4B-DFlash` checkpoint — which omits
///     `embed_tokens.weight` — *and* BINDS to the Qwen 3.5 target,
///     resolving its `embed_tokens` lazily from the target (issue #675).
#[test]
#[ignore = "real-model heavy (loads Qwen3.5-4B target + drafter); runs in CI hardware lane only"]
fn greedy_parity_dflash_qwen35_4b() {
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, resolve_drafter_kind};

    let pairing = &REACHABLE_PAIRINGS[0];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {}: target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    eprintln!("[{}] Loading target from {:?}", pairing.name, target_path);
    let (loaded_target, _target_tokenizer) =
        load_model(&target_path).expect("target model must load");

    // Qwen 3.5 4B checkpoints from `mlx-community` are published as
    // `Qwen3_5ForConditionalGeneration` (VLM variant) even for the text-only
    // 4B checkpoint, so the variant we expect is `Qwen35VLM` /
    // `Qwen35MoeVLM` for the VLM-wrapped variants and `Qwen35` / `Qwen35Moe`
    // for the pure text-only variants (less common in `mlx-community`).
    let target_is_qwen35 = matches!(
        loaded_target,
        LoadedModel::Qwen35(_)
            | LoadedModel::Qwen35Moe(_)
            | LoadedModel::Qwen35VLM(_)
            | LoadedModel::Qwen35MoeVLM(_)
    );
    assert!(
        target_is_qwen35,
        "DFlash pairing requires a Qwen 3.5 target but load_model returned a different variant; \
         check the pairing target_dir matches a Qwen 3.5 4-bit checkpoint",
    );
    eprintln!(
        "[{}] Target loaded as Qwen 3.5 family (Qwen35Model / Qwen35VLModel)",
        pairing.name
    );

    eprintln!("[{}] Loading drafter from {:?}", pairing.name, draft_path);
    let resolved_kind =
        resolve_drafter_kind(&draft_path, None).expect("drafter config.json must be readable");
    assert_eq!(resolved_kind, DrafterKind::Dflash);

    // The drafter itself loads through `load_drafter` which constructs the
    // full DFlashDrafter (including weight loading + sanitize). This is the
    // load-bearing structural check: if the drafter cannot be constructed
    // against its own checkpoint, no amount of round-loop wiring downstream
    // will help.
    //
    // The upstream `z-lab/Qwen3.5-4B-DFlash` checkpoint does NOT ship
    // `embed_tokens.weight` — upstream Python sets `self.embed_tokens =
    // None` at construction and `bind`s to the target's `embed_tokens`
    // later
    // (`references/mlx-vlm/mlx_vlm/speculative/drafters/qwen3_dflash/dflash.py`
    // lines 88, 92-108). Issue #675 ported that lazy-bind shape: the Rust
    // loader builds the drafter with an `embed_tokens = None` tombstone
    // when the safetensors index omits the table, and resolves it from
    // the target during `Drafter::bind` via
    // `LanguageModel::embed_tokens_module`. So `load_drafter` MUST now
    // succeed on the published checkpoint — a `LoadFailed` here is a
    // regression of #675.
    let (mut drafter, drafter_kind) =
        mlxcel_core::drafter::load_drafter(&draft_path, Some(DrafterKind::Dflash)).expect(
            "DFlash drafter must load against the published z-lab/Qwen3.5-4B-DFlash \
             checkpoint (issue #675: embed_tokens.weight is absent and the loader \
             builds a lazy-bind tombstone instead of failing)",
        );
    assert_eq!(drafter_kind, DrafterKind::Dflash);
    eprintln!(
        "[{}] Drafter loaded successfully; block_size={}",
        pairing.name, pairing.block_size,
    );

    // Bind the drafter to the Qwen 3.5 target. For the published lazy-bind
    // checkpoint this is the load-bearing step from issue #675: `bind`
    // resolves the drafter's `embed_tokens` from the target's embedding
    // module (`LanguageModel::embed_tokens_module`, implemented by the
    // Qwen 3.5 family). A `BindFailed` here means either the target does
    // not expose `embed_tokens_module` or the lazy-bind tombstone was not
    // wired — both are #675 regressions.
    drafter
        .bind(&loaded_target)
        .expect("DFlash drafter must bind to the Qwen 3.5 target (issue #675 lazy-bind)");
    eprintln!(
        "[{}] Drafter bound to target; embed_tokens resolved via lazy-bind",
        pairing.name,
    );

    // FULL BYTE-EQUALITY TEST DEFERRED TO #676:
    // The full byte-equality assertion needs a public way to construct
    // `Qwen3NextCache` from outside the binary crate so this test can
    // drive `DFlashGenerator::run(...)`, or — per #676 — a single-request
    // server harness that submits a fixed prompt with and without
    // `--draft-kind dflash` at temp=0 and asserts byte-identical token
    // sequences. The load + bind pin above is the structural gate that
    // #675 unblocks; #676 layers the end-to-end check on top.
    eprintln!(
        "[{}] Load + bind check passed; full byte-equality assertion deferred to #676",
        pairing.name,
    );
}

/// Greedy parity for the Gemma 4 31B + MTP assistant pairing.
///
/// **Status:** scaffolded; full byte-equality assertion deferred to #666.
///
/// The MTP B=1 round loop (`mlxcel_core::speculative::mtp::MtpGenerator`)
/// requires a `&dyn MtpTarget` implementation on the target wrapper. The
/// underlying primitives (`Gemma4Wrapper::forward_with_speculative_sinks`,
/// `Gemma4Wrapper::rollback_speculative_cache`) are all public, but the
/// `MtpTarget` trait impl itself is intentionally deferred to follow-up
/// #666 per the explicit deferral note in PR #665.
///
/// What the test asserts today:
///   - Both target and drafter directories load successfully.
///   - The drafter's resolved kind is exactly `DrafterKind::Mtp` (which
///     auto-detects from `model_type: "gemma4_assistant"` in the drafter
///     config.json).
///   - The target wraps a `LoadedModel::Gemma4` variant.
///
/// Note on dtype: this pairing uses a 4-bit quantized target with a bf16
/// drafter. The 4-bit target keeps bf16 scales/biases as-is per
/// `docs/apple-silicon-precision.md`; the drafter is dtype-agnostic as
/// long as the target's `forward_with_speculative_sinks` preserves the
/// captured hidden + shared K/V slab dtypes (which it does — see the
/// "preserves the model's native bf16/f16 dtype — no f32 promotion" note
/// on `Gemma4SpeculativeSinks`).
#[test]
#[ignore = "real-model heavy (loads Gemma-4-31B target + drafter); runs in CI hardware lane only"]
fn greedy_parity_mtp_gemma4_31b() {
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, resolve_drafter_kind};

    let pairing = &REACHABLE_PAIRINGS[1];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {}: target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    eprintln!("[{}] Loading target from {:?}", pairing.name, target_path);
    let (loaded_target, _target_tokenizer) =
        load_model(&target_path).expect("target model must load");

    // Gemma 4 31B checkpoints are typically published as
    // `Gemma4ForConditionalGeneration` (VLM variant), so we accept either the
    // text-only `Gemma4` wrapper or the `Gemma4VLM` VLM wrapper. The MTP
    // path operates on the inner text model in both cases (the vision tower
    // is bypassed when no image tokens are present in the prompt).
    let target_is_gemma4 = matches!(
        loaded_target,
        LoadedModel::Gemma4(_) | LoadedModel::Gemma4VLM(_)
    );
    assert!(
        target_is_gemma4,
        "MTP pairing requires a Gemma 4 target but load_model returned a different variant; \
         check the pairing target_dir matches a Gemma 4 4-bit checkpoint",
    );
    eprintln!(
        "[{}] Target loaded as Gemma 4 family (Gemma4Wrapper / Gemma4VLModel)",
        pairing.name
    );

    eprintln!("[{}] Loading drafter from {:?}", pairing.name, draft_path);
    let resolved_kind =
        resolve_drafter_kind(&draft_path, None).expect("drafter config.json must be readable");
    assert_eq!(resolved_kind, DrafterKind::Mtp);

    let (_drafter, _kind) = mlxcel_core::drafter::load_drafter(&draft_path, Some(DrafterKind::Mtp))
        .expect("Gemma 4 MTP assistant drafter must load from checkpoint");
    eprintln!(
        "[{}] Drafter loaded; block_size={}",
        pairing.name, pairing.block_size
    );

    // FULL TEST DEFERRED TO #666:
    // The full byte-equality assertion needs an `MtpTarget` impl on
    // `Gemma4Wrapper`. The Gemma 4 wrapper has all the required hooks
    // (`forward_with_speculative_sinks`, `rollback_speculative_cache`); the
    // adapter struct that wires those hooks to the `MtpTarget` trait
    // signatures (`prefill_and_seed`, `verify_forward`, `verify_finalize`)
    // is the missing piece. See the module docstring for the deferral
    // rationale.
    eprintln!(
        "[{}] Structural check passed; full byte-equality assertion deferred to #666",
        pairing.name,
    );
}

/// Sanity that the test discovery against `models/` finds at least one of
/// the reachable pairings on hosts that have downloaded the checkpoints,
/// and cleanly skips with a log line on hosts that have not.
#[test]
fn reachable_pairing_discovery_reports_status() {
    let mut any_present = false;
    for pairing in REACHABLE_PAIRINGS {
        let (target_path, draft_path, present) = pairing_present(pairing);
        eprintln!(
            "  - {}: target={:?} draft={:?} present={}",
            pairing.name, target_path, draft_path, present,
        );
        any_present |= present;
    }
    if any_present {
        eprintln!("At least one reachable speculative pairing is present on disk.");
    } else {
        eprintln!(
            "No reachable speculative pairings on disk — skipping perf benchmarks. \
             Run `mlxcel download mlx-community/Qwen3.5-4B-DFlash` (and similar) to populate.",
        );
    }
}

/// Static check: the reachable pairing list cannot accidentally be empty
/// (a future cleanup that drops every reachable pairing should fail loudly
/// at compile time rather than silently making the whole file a no-op).
const _: () = {
    assert!(
        !REACHABLE_PAIRINGS.is_empty(),
        "REACHABLE_PAIRINGS must declare at least one pairing; \
         see docs/model_tests.md::Speculative drafters",
    );
};
