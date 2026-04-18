#!/usr/bin/env bash
#
# Heterogeneous-memory pipeline-parallel scenario. Runs on every PR that
# touches PP paths so that regressions in either the partitioner or the
# per-stage admission control surface as a clear test failure instead of
# as a silent OOM on the memory-constrained stage in production.
#
# The test uses synthetic ModelProfile / DeviceSpec values, so it runs on
# any runner (GitHub-hosted Linux, self-hosted macOS) and requires no
# model weights.
#
# Usage:
#   scripts/ci/run-pp-heterogeneous-memory.sh
#
# Optional environment:
#   CARGO_PROFILE   `debug` (default) or `release`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ci/_common.sh
source "$SCRIPT_DIR/_common.sh"

export PP_CI_SCRIPT_TAG="heterogeneous-memory"
ROOT_DIR="$(pp_ci_root)"
cd "$ROOT_DIR"

pp_ci_require cargo

CARGO_PROFILE="${CARGO_PROFILE:-debug}"
PROFILE_FLAGS=()
if [[ "$CARGO_PROFILE" == "release" ]]; then
  PROFILE_FLAGS+=("--release")
fi

pp_ci_log "running heterogeneous-memory partition regression"

cargo test "${PROFILE_FLAGS[@]}" \
  --test pipeline_ci_multi_stage_real_models \
  pipeline_heterogeneous_memory_partition_is_stable \
  -- --nocapture

cargo test "${PROFILE_FLAGS[@]}" \
  --test pipeline_ci_multi_stage_real_models \
  pipeline_heterogeneous_memory_admission_rejects_oom \
  -- --nocapture

pp_ci_log "heterogeneous-memory scenario passed"
