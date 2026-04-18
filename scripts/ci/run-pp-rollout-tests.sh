#!/usr/bin/env bash
#
# Run the five rollout-checklist pipeline-parallel tests documented in
# `docs/PIPELINE_PARALLELISM.md`. These tests are `#[ignore]` by default
# because they exercise real model weights; this script runs them with
# `--ignored` so both CI and operators get a single, consistent entry point.
#
# Required environment:
#   (none — tests skip themselves cleanly when model weights are missing)
#
# Optional environment:
#   TEST_MODEL       Model directory name under `models/`. Default:
#                    `llama-3.2-1b-4bit`. The individual tests skip when the
#                    directory is absent, so CI runners without weights still
#                    complete the workflow step successfully.
#   CARGO_PROFILE    `debug` (default) or `release`. Real-model tests default
#                    to `debug` to keep CI wall-time reasonable; operators can
#                    override for a more representative measurement.
#
# Usage:
#   scripts/ci/run-pp-rollout-tests.sh
#
# Exit codes:
#   0   all tests passed (or skipped cleanly when the model is missing)
#   non-zero on the first failing `cargo test` invocation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ci/_common.sh
source "$SCRIPT_DIR/_common.sh"

export PP_CI_SCRIPT_TAG="rollout-tests"
ROOT_DIR="$(pp_ci_root)"
cd "$ROOT_DIR"

pp_ci_require cargo

CARGO_PROFILE="${CARGO_PROFILE:-debug}"
TEST_MODEL="${TEST_MODEL:-llama-3.2-1b-4bit}"
PROFILE_FLAGS=()
if [[ "$CARGO_PROFILE" == "release" ]]; then
  PROFILE_FLAGS+=("--release")
fi

pp_ci_log "repo root: $ROOT_DIR"
pp_ci_log "cargo profile: $CARGO_PROFILE"
pp_ci_log "target model: $TEST_MODEL"

if pp_ci_resolve_model_dir "$TEST_MODEL" >/dev/null 2>&1; then
  pp_ci_log "model weights resolved; tests will exercise real-model code paths"
else
  pp_ci_log "model weights absent; tests will self-skip but the workflow step still runs"
fi

# Each entry is "TEST_BINARY::TEST_NAME" — `cargo test` runs them with
# `--test <binary> <name>` so a single ignored test is targeted per call.
# The list mirrors the rollout checklist in docs/PIPELINE_PARALLELISM.md so
# it stays easy to audit and re-order from the documentation side.
TESTS=(
  "pipeline_cli_real_models::pipeline_cli_llama_real_model_parity"
  "pipeline_stage_executor_real_models::pipeline_stage_executor_llama_real_model_parity"
  "pipeline_stage_executor_real_models::pipeline_stage_worker_loop_llama_real_model_parity"
  "pipeline_server_real_models::pipeline_server_llama_multi_request_smoke"
  "pipeline_server_real_models::pipeline_server_llama_dense_baseline_smoke"
)

for entry in "${TESTS[@]}"; do
  test_binary="${entry%%::*}"
  test_name="${entry##*::}"
  pp_ci_log "running $test_binary :: $test_name"
  cargo test "${PROFILE_FLAGS[@]}" \
    --test "$test_binary" \
    "$test_name" \
    -- --ignored --nocapture --test-threads=1
done

pp_ci_log "rollout tests completed"
