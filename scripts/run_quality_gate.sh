#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCLUDE_SMOKE=0
DRY_RUN=0
TEXT_MODEL="models/smollm-135m"
VLM_MODEL="models/llava-qwen-0.5b"
VLM_IMAGE="tests/fixtures/test_image.png"

usage() {
  cat <<'EOF'
Usage: scripts/run_quality_gate.sh [options]

Run the mlxcel quality gate baseline from the repository root.

Options:
  --include-smoke     Run CPU-only text + VLM smoke checks after the build/test baseline.
  --text-model PATH   Override the CPU-only text smoke model path.
  --vlm-model PATH    Override the CPU-only VLM smoke model path.
  --vlm-image PATH    Override the VLM smoke image path.
  --dry-run           Print commands without executing them.
  --help              Show this help.
EOF
}

run_cmd() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'

  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-smoke)
      INCLUDE_SMOKE=1
      shift
      ;;
    --text-model)
      TEXT_MODEL="$2"
      shift 2
      ;;
    --vlm-model)
      VLM_MODEL="$2"
      shift 2
      ;;
    --vlm-image)
      VLM_IMAGE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

run_cmd cargo fmt --all
run_cmd cargo test --lib --quiet
run_cmd cargo test --bin mlxcel --quiet
run_cmd cargo build --bins --quiet
run_cmd cargo clippy --all-targets -- -D warnings
run_cmd cargo test --manifest-path src/lib/mlxcel-core/Cargo.toml -- --test-threads=1
run_cmd cargo clippy --manifest-path src/lib/mlxcel-core/Cargo.toml --all-targets -- -D warnings

if [[ "$INCLUDE_SMOKE" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ ! -d "$TEXT_MODEL" ]]; then
      echo "Missing text smoke model directory: $TEXT_MODEL" >&2
      exit 1
    fi
    if [[ ! -d "$VLM_MODEL" ]]; then
      echo "Missing VLM smoke model directory: $VLM_MODEL" >&2
      exit 1
    fi
    if [[ ! -f "$VLM_IMAGE" ]]; then
      echo "Missing VLM smoke image: $VLM_IMAGE" >&2
      exit 1
    fi
  fi

  run_cmd env MLXCEL_BUILD_METAL=OFF cargo build --bin mlxcel --quiet
  run_cmd env MLXCEL_DEVICE=cpu target/debug/mlxcel generate -m "$TEXT_MODEL" -p "Hello" -n 1 --no-chat-template
  run_cmd env MLXCEL_DEVICE=cpu target/debug/mlxcel generate -m "$VLM_MODEL" --image "$VLM_IMAGE" -p "Describe the image." -n 1 --no-chat-template
fi
