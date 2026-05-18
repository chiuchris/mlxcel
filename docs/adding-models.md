# Model Addition Guide

This guide is the practical checklist for adding new text models and VLMs to `mlxcel`.
It points to the concrete control surfaces that must stay consistent. If this repository later adds a maintainer workflow document, keep this checklist aligned with it.

## Goals

- Keep new model additions predictable.
- Reuse existing control-plane helpers instead of adding new one-off branches.
- Add tests alongside the integration points that are easiest to regress.

## Before You Start

1. Identify the upstream reference:
   - Text models: upstream `mlx-lm` model implementations.
   - VLMs: upstream `mlx-vlm` model implementations.
   - If local `references/` checkouts are not present, clone or inspect the upstream repositories separately.
2. Decide whether the architecture is:
   - A brand new model family
   - A format alias of an existing family
   - A VLM wrapper around an existing text model
3. Check whether an existing loader helper already matches the new model:
   - `src/model_metadata.rs`
   - `src/loading/mod.rs`
   - `src/loading/vlm.rs`
   - `src/models/mod.rs`
   - Start with the converged registration surface in `src/model_metadata.rs`.
     Standard text models should extend that registration table first, because
     `src/loading/config_backed.rs` now consumes the same source of truth.
4. Check whether the change also touches shared execution policy:
   - `src/execution/runtime.rs` for device/environment behavior
   - `src/execution/sampling.rs` for user-facing sampling defaults and greedy-vs-sampled assembly

## Text Model Checklist

1. Add the implementation file under `src/models/`.
2. Register the module and re-export in `src/models/mod.rs`.
3. Add a `ModelType` variant in `src/models/mod.rs`.
4. Extend `get_model_type()` in `src/models/detection.rs`.
   - Prefer shared helpers such as `detect_text_or_vlm()` and `detect_hunyuan_model_type()` when the new model fits an existing pattern.
5. Add the corresponding `LoadedModel` variant in `src/loaded_model.rs`.
   - Prefer extending the existing dispatch helpers instead of adding new repeated match tables:
     `delegate_language_model!` in `src/loaded_model.rs` and `VlmRuntimeRef`
     in `src/loaded_model_capabilities.rs`
6. Wire loading in `src/loading/mod.rs`.
   - Prefer existing helpers like `load_pair_from_dir()` and `load_owned_model_from_config!`.
   - Update `src/model_metadata.rs` so kind, adapter support, route selection,
     and standard config-backed registration stay centralized before touching
     the router.
   - If the model follows the standard text-model path, extend the shared
     registration surface in `src/model_metadata.rs` instead of adding a
     parallel entry list in `src/loading/config_backed.rs`.
7. If LoRA/adapters are supported, verify `load_model_from_weights()` in `src/loading/mod.rs`.
   - Non-standard adapter paths should extend `src/loading/special.rs` instead of growing `load_model_from_weights()` directly.

## VLM Checklist

1. Implement or reuse the vision encoder under `src/vision/encoders/`.
2. Implement or reuse the connector under `src/vision/connectors/`.
3. Implement or reuse the processor under `src/vision/processors/`.
4. Add the VLM `ModelType` detection in `src/models/detection.rs`.
   - If the base text family has both text-only and VLM variants, prefer `detect_text_or_vlm()`.
5. Add the loader entry in `src/loading/vlm.rs` or the matching family module under `src/loading/`.
   - Prefer shared helpers for config parsing, token defaults, and weight remapping.
   - Keep family-specific assembly grouped with its nearest peers:
     `src/loading/vlm_qwen.rs`, `src/loading/vlm_llava.rs`, `src/loading/vlm_gemma.rs`,
     `src/loading/vlm_pixtral.rs`, `src/loading/vlm_siglip.rs`, `src/loading/vlm_special.rs`.
   - Update `src/model_metadata.rs` so the router knows the family is
     multimodal and adapter loading policy remains explicit.
   - Register the directory entry point in `try_load_vlm_model_from_dir()` in `src/loading/mod.rs` so `load_model()` stays as a thin dispatcher.
6. Add or reuse the `LoadedModel` capability helpers in
   `src/loaded_model_capabilities.rs`.
   - Prefer extending `VlmRuntimeRef` or an existing multimodal helper over
     adding family-specific getters that only CLI/server use.
7. Reuse prompt helpers where possible:
   - Qwen-VL token insertion: `src/multimodal/qwen_vl.rs`
   - Generic image-token block expansion: `src/multimodal/vlm_prompt.rs`
   - Phi3V prompt tag handling: `src/multimodal/phi3v_prompt.rs`

## When to Create a New Family Module

Do not create a new file by default. Create one when the family has a distinct
control-plane identity.

Create a new loader family module when:

- config normalization is not a small variant of an existing family
- token defaults or weight-key remapping need dedicated tests
- the VLM wrapper uses a different prompt/runtime assembly path
- adding the logic inline would obscure an existing family boundary

Keep the model in an existing module when:

- it is primarily an alias or small config delta
- the same loader tests already express the policy
- the family is still recognizable after the change

If you are unsure, extend the existing family module first and split only when
the test file or router starts to lose a clear boundary.

## Where Regressions Usually Happen

- `src/models/mod.rs`
  - Missing module export or `ModelType` variant
- `src/models/detection.rs`
  - Missing aliases in `get_model_type()`
  - Text/VLM misclassification when `vision_config` is present
- `src/loading/mod.rs`
  - Divergence between `load_model()` and `load_model_from_weights()`
  - Adding a standard config-backed model as a one-off special case instead of the shared loader helpers
  - Adding a VLM directly into `load_model()` instead of `try_load_vlm_model_from_dir()`
- `src/model_metadata.rs`
  - Forgetting to update text/VLM kind, adapter support, route policy, or the
    shared standard-text registration entry before wiring loaders
- `src/loading/config_backed.rs`
  - Bypassing the shared registration surface and adding new one-off loader logic
  - Forgetting wrapper constructors for models such as `Llama4`, `Gemma3`, or `Ministral3`
- `src/loading/nonstandard.rs`
  - Leaving directory-only loader families in `src/loading/mod.rs` instead of the non-standard registry
- `src/loading/special.rs`
  - Adding adapter/owned-weight special handling inline instead of the special-weight registry
  - Forgetting Qwen3.5 text-config normalization or owned-weight sanitization before construction

Keep `src/loading/mod.rs` focused on route selection. If a new model family adds
substantial construction logic, prefer a dedicated sibling module and call it
from the router instead of growing `load_model()` or `load_model_from_weights()`
inline.

For very large model families, extract internal helper hotspots into a focused
sibling helper module when the code changes for different reasons than the main
decoder stack. Current examples:

- `src/models/gemma3n_helpers.rs`
- `src/models/llama4_helpers.rs`
- `src/loading/vlm.rs` and sibling family modules under `src/loading/`
  - Wrong default token IDs
  - Missing top-level quantization inheritance
  - Incorrect weight-key remapping between text and vision towers
- `src/loaded_model.rs` / `src/loaded_model_capabilities.rs`
  - Missing dispatch arm for a new variant
  - Missing capability wiring in `VlmRuntimeRef`
  - Updating the all-model dispatch macro but forgetting the multimodal
    capability switchboard

## Test Expectations

Add tests in the same slice as the model/control-plane change.

- For model detection helpers:
  - Add unit tests near `src/models/detection.rs`
- For sanitization helpers:
  - Add unit tests near `src/models/sanitize.rs`
- For loader normalization or token-default logic:
  - Add tests near `src/loading/tests.rs` or the relevant `src/loading/vlm*_tests.rs`
- For shared vision merge contracts:
  - Add tests near `src/vision/merge_tests.rs`
- For prompt/token expansion logic:
  - Add tests in dedicated helper test files such as `src/multimodal/qwen_vl_tests.rs`, `src/multimodal/vlm_prompt_tests.rs`, `src/multimodal/phi3v_prompt_tests.rs`
- For runtime validation:
  - Run `scripts/run_quality_gate.sh`
  - Add at least one local smoke test when a matching model exists
  - If the slice touched MLX-heavy ignored helper tests, run them explicitly
    with `--ignored --test-threads=1`

## Shared Function Policy

When touching shared functions used by multiple model families, update the local usage comments in the shared helper files, especially under:

- `src/lib/mlxcel-core/src/layers.rs`
- `src/lib/mlxcel-core/src/utils.rs`

Those comments act as the retest list for future changes.

## Cross-Cutting Execution Helpers

Keep entry-point policy in the shared execution layer when the behavior must be
identical across CLI, server, and future frontends.

- `src/execution/runtime.rs`
  - Environment-driven device selection (`MLXCEL_DEVICE`)
  - GPU wired-memory limit setup
- `src/execution/sampling.rs`
  - Centralized `SamplingConfig` assembly from resolved request defaults
  - Shared greedy vs non-greedy branching

If a new frontend or request type needs different defaults, resolve those
defaults at the edge and keep the final conversion in `src/execution/`.

Keep CLI-only prompt formatting and terminal output behavior in
`src/commands/generate.rs` instead of moving it into shared loading or server
modules.

For server-only boot behavior, keep startup policy in `src/server/startup.rs`
instead of growing `src/server/mod.rs`:

- API key / chat-template resolution precedence
- startup-time normalization of CLI-compatible flags
- warmup behavior
- Unix-socket vs TCP binding

Keep shared server types in the focused modules as well:

- `src/server/config.rs` for request/default configuration structs
- `src/server/state.rs` for `AppState` and metrics containers
- `src/server/model_provider.rs` for the public request/response channel API
- `src/server/model_worker.rs` for the long-lived worker thread, VLM request prep, and decode state

Keep server edge adapters out of the route files once more than one endpoint
needs the same behavior:

- `src/server/chat_request.rs` for OpenAI chat message flattening and prompt fallback assembly
- `src/server/request_options.rs` for request-default merging into `ServerGenerateOptions`
- `src/server/media.rs` for `data:` / `file://` image-source parsing
- `src/server/streaming.rs` for shared SSE channel and `[DONE]` emission helpers

## Dry-Run Workflow Validation

This section validates that the current architecture actually reduces ambiguity
when adding new model support.

### Dry Run A: Standard Text Model

Assume a new text model that follows the existing config-backed loading path.

Required surfaces today:

1. `src/models/<family>.rs`
2. `src/models/mod.rs`
3. `src/models/detection.rs`
4. `src/model_metadata.rs` through the converged registration surface plus
   `static_model_descriptor()` / `model_load_policy()`
5. `src/loading/config_backed.rs` only if shared config-backed loading behavior
   itself must change
6. `src/loaded_model.rs`
7. `src/loaded_model_capabilities.rs` only if the family changes multimodal
   capability exposure
8. tests near `src/models/detection_tests.rs`, `src/models/sanitize_tests.rs`, and `src/loading/tests.rs`

What should *not* happen:

- no new one-off construction branch inside `load_model()`
- no direct CLI or server changes unless user-visible behavior changes
- no family-specific getter added to `LoadedModel` if an existing capability is enough

Why this is better than the old path:

- route selection is centralized instead of duplicated across multiple loading matches
- adapter support is declared in one policy surface
- standard text constructor registration no longer lives in a separate parallel table
- the expected edit list is short enough to review before coding starts

### Dry Run B: New VLM Family

Assume a new VLM family needs its own token defaults and weight-key remapping.

Required surfaces today:

1. text model and/or VLM wrapper under `src/models/` or `src/vision/`
2. `src/models/mod.rs`
3. `src/models/detection.rs`
4. `src/loading/vlm_<family>.rs`
5. `src/loading/vlm.rs`
6. `src/model_metadata.rs` through `static_model_descriptor()` / `model_load_policy()`
7. `src/loaded_model.rs` through enum wiring
8. `src/loaded_model_capabilities.rs` through `VlmRuntimeRef`
9. `src/multimodal/` only if prompt/runtime preparation is truly new
10. tests near `src/loading/vlm_<family>_tests.rs` and any new multimodal helper test file

What should *not* happen:

- no concrete model-type checks added to CLI or server request paths
- no family-specific loading logic added directly to `src/loading/mod.rs`
- no duplicated prompt-rewrite logic across CLI and server

Why this is better than the old path:

- the family router lives in `src/loading/vlm.rs`
- family assembly stays beside peer VLM loaders
- multimodal frontends depend on capabilities, not family names

### Review Checklist for a Real Addition

Before opening a PR for a new model or VLM family, confirm:

1. loading policy was updated through `src/model_metadata.rs`
2. `LoadedModel` wiring stayed inside `src/loaded_model.rs` and
   `src/loaded_model_capabilities.rs` rather than creating a new one-off
   family getter
3. CLI and server still depend on shared helpers rather than the concrete model type
4. unit tests cover the new policy or normalization logic
5. at least one smoke test exists when a local model is available

## Commit Strategy

Prefer small checkpoints that isolate one control-plane surface:

- model detection
- loader normalization
- prompt preparation
- runtime initialization

This keeps regressions searchable and makes future model additions easier to compare against previous slices.
