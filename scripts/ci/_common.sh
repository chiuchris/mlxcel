# shellcheck shell=bash
# Shared helpers for the pipeline-parallel CI scripts.
#
# This file is intended to be `source`'d from other scripts under
# `scripts/ci/`. It provides:
#   - `pp_ci_root`               repository root regardless of invocation dir
#   - `pp_ci_require`            abort early if a required tool is missing
#   - `pp_ci_reserve_port`       reserve an ephemeral TCP port and print it
#   - `pp_ci_wait_for_health`    poll a `/health` URL until it returns 200
#   - `pp_ci_kill_pids`          terminate background PIDs cleanly on EXIT
#   - `pp_ci_mkdir_p`            mkdir -p idempotent wrapper for bash -u safety
#   - `pp_ci_log`                structured log with a script-tag prefix
#
# The helpers are deliberately dependency-free (no yq / jq / python) so they
# can run on a minimal Linux GitHub-hosted runner image and on a self-hosted
# macOS runner without extra setup.

set -euo pipefail

if [[ -n "${PP_CI_COMMON_LOADED:-}" ]]; then
  return 0
fi
PP_CI_COMMON_LOADED=1

pp_ci_root() {
  git -C "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" rev-parse --show-toplevel 2>/dev/null \
    || (cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
}

PP_CI_SCRIPT_TAG="${PP_CI_SCRIPT_TAG:-pp-ci}"

pp_ci_log() {
  local ts
  ts="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf '[%s %s] %s\n' "$ts" "$PP_CI_SCRIPT_TAG" "$*" >&2
}

pp_ci_require() {
  local missing=0
  for tool in "$@"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      pp_ci_log "missing required tool: $tool"
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    exit 127
  fi
}

pp_ci_mkdir_p() {
  for dir in "$@"; do
    mkdir -p "$dir"
  done
}

# Reserve an ephemeral port by binding to 127.0.0.1:0 and reading the assigned
# port number back out. The listening socket is closed before returning so a
# subsequent bind(2) in Rust can claim the same port. There is still a small
# race window, but it is the same one the production rollout script relies on
# and is sufficient for CI.
pp_ci_reserve_port() {
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
  else
    # Fallback for containers without Python (rare on CI): use a bash+nc loop
    # over the 19000-19999 user range.
    local port
    for port in $(seq 19000 19999); do
      if ! (echo > "/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1; then
        echo "$port"
        return 0
      fi
    done
    pp_ci_log "failed to reserve a port in the 19000-19999 range"
    return 1
  fi
}

pp_ci_wait_for_health() {
  local url="$1"
  local timeout_secs="${2:-90}"
  local end=$(( $(date +%s) + timeout_secs ))
  while [[ $(date +%s) -lt $end ]]; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  pp_ci_log "timed out waiting for $url after ${timeout_secs}s"
  return 1
}

# Maintains an array of PIDs the caller has launched in the background. When
# sourced scripts exit, `pp_ci_cleanup_pids` sends SIGTERM (escalating to
# SIGKILL) so nothing is left running past the workflow step.
PP_CI_PIDS=()

pp_ci_track_pid() {
  PP_CI_PIDS+=("$1")
}

pp_ci_kill_pids() {
  local pid
  for pid in "${PP_CI_PIDS[@]:-}"; do
    if [[ -z "$pid" ]]; then
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${PP_CI_PIDS[@]:-}"; do
    if [[ -z "$pid" ]]; then
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

# Optional helper: locate a model under ${MODEL_ROOT}, or under the repo's
# `models/` directory, or under the primary worktree's `models/`. Returns
# empty string (and non-zero status) when the model is absent, so callers
# can gracefully skip model-dependent steps on CI runners that do not ship
# model weights.
pp_ci_resolve_model_dir() {
  local name="$1"
  local root
  root="$(pp_ci_root)"
  local candidates=(
    "${MODEL_ROOT:-}/${name}"
    "${root}/models/${name}"
  )
  local primary_worktree
  primary_worktree="$(git -C "$root" worktree list --porcelain 2>/dev/null | sed -n 's/^worktree //p' | head -n1 || true)"
  if [[ -n "$primary_worktree" && "$primary_worktree" != "$root" ]]; then
    candidates+=("${primary_worktree}/models/${name}")
  fi
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}
