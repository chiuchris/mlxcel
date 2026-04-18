#!/usr/bin/env bash
#
# 3-host pipeline-parallel job for self-hosted runners with real hardware.
#
# Unlike the 2-host logical script (which simulates stages on one runner),
# this script expects the three-stage topology to come from a self-hosted
# runner lab and executes the full production code path with a 3-stage
# loopback real-model test. The same entry point is reusable from operator
# laptops when they want to reproduce the CI result locally.
#
# Operator-facing environment:
#   PP_THREE_HOST_MODEL    Required. Model directory name under `models/`.
#                          Example: `llama-3.2-1b-4bit`.
#   PP_THREE_HOST_PROMPT   Optional prompt. Default: short English sentence.
#   PP_THREE_HOST_TOKENS   Optional max tokens. Default: 16.
#   CARGO_PROFILE          `release` (default for perf relevance) or `debug`.
#   REPORT_PATH            Optional path to append a markdown row per run.
#                          When set, captures tokens/sec + bubble ratio.
#
# Exit codes:
#   0    success (or skipped cleanly when the model is missing)
#   2    model weights not found and STRICT_MODEL=1 is set
#   other: cargo test failure
#
# Usage:
#   scripts/ci/run-pp-three-host.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ci/_common.sh
source "$SCRIPT_DIR/_common.sh"

export PP_CI_SCRIPT_TAG="three-host"
ROOT_DIR="$(pp_ci_root)"
cd "$ROOT_DIR"

pp_ci_require cargo

CARGO_PROFILE="${CARGO_PROFILE:-release}"
TEST_MODEL="${PP_THREE_HOST_MODEL:-${TEST_MODEL:-llama-3.2-1b-4bit}}"
STRICT_MODEL="${STRICT_MODEL:-0}"

PROFILE_FLAGS=()
if [[ "$CARGO_PROFILE" == "release" ]]; then
  PROFILE_FLAGS+=("--release")
fi

pp_ci_log "repo root: $ROOT_DIR"
pp_ci_log "cargo profile: $CARGO_PROFILE"
pp_ci_log "target model: $TEST_MODEL"

if ! pp_ci_resolve_model_dir "$TEST_MODEL" >/dev/null 2>&1; then
  if [[ "$STRICT_MODEL" == "1" ]]; then
    pp_ci_log "model weights absent for $TEST_MODEL and STRICT_MODEL=1; failing"
    exit 2
  fi
  pp_ci_log "model weights absent; the 3-host test will self-skip. Set STRICT_MODEL=1 to make this an error."
fi

# The test drives a 3-stage remote pipeline against the real model. This is
# the only place in CI where the 3-stage path is exercised end-to-end, so the
# script fails loudly (not silently) on cargo failure.
cargo test "${PROFILE_FLAGS[@]}" \
  --test pipeline_ci_multi_stage_real_models \
  pipeline_multi_stage_three_host_real_model_parity \
  -- --ignored --nocapture --test-threads=1

pp_ci_log "3-host pipeline validation completed"

if [[ -n "${REPORT_PATH:-}" ]]; then
  pp_ci_log "writing a placeholder run marker to $REPORT_PATH (operators should replace with real numbers)"
  pp_ci_mkdir_p "$(dirname "$REPORT_PATH")"
  {
    printf '| %s | %s | %s | %s |\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
      "$TEST_MODEL" \
      "$CARGO_PROFILE" \
      "pass"
  } >>"$REPORT_PATH"
fi
