#!/usr/bin/env bash
# Run the Rust + Python bridge-overhead microbenches side by side.
#
# Usage:
#   scripts/run_bridge_microbench.sh                 # both single-op and layer-shaped
#   scripts/run_bridge_microbench.sh --single-op     # only single-op
#   scripts/run_bridge_microbench.sh --layer-shaped  # only layer-shaped
#
# The layer-shaped run accepts optional positional shape arguments after
# the mode flag:
#   scripts/run_bridge_microbench.sh --layer-shaped HIDDEN N_LAYERS N_STEPS
# Defaults: HIDDEN=2816 N_LAYERS=30 N_STEPS=50 (Gemma 4 26B-a4b shape).
#
# See `docs/bridge-overhead-microbench.md` for interpretation guidance.

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="all"
case "${1:-}" in
  --single-op) MODE="single_op"; shift ;;
  --layer-shaped) MODE="layer_shaped"; shift ;;
  --help|-h)
    head -n 15 "$0" | sed 's/^# //; s/^#//'
    exit 0
    ;;
esac

HIDDEN="${1:-2816}"
N_LAYERS="${2:-30}"
N_STEPS="${3:-50}"

SINGLE_OP_BIN="./target/release/examples/bridge_overhead_microbench"
LAYER_BIN="./target/release/examples/layer_shaped_microbench"

need_build=0
[[ -x "$SINGLE_OP_BIN" ]] || need_build=1
[[ -x "$LAYER_BIN" ]] || need_build=1
if (( need_build )); then
  echo ">>> Building Rust examples..."
  cargo build --release \
    --example bridge_overhead_microbench \
    --example layer_shaped_microbench
fi

run_single_op() {
  echo "================================================================"
  echo "=== Single-op microbench: Rust (cxx bridge)"
  echo "================================================================"
  "$SINGLE_OP_BIN"
  echo
  echo "================================================================"
  echo "=== Single-op microbench: Python (nanobind)"
  echo "================================================================"
  python3 scripts/bridge_overhead_microbench_py.py
}

run_layer_shaped() {
  echo "================================================================"
  echo "=== Layer-shaped workload (HIDDEN=$HIDDEN N_LAYERS=$N_LAYERS N_STEPS=$N_STEPS):"
  echo "=== Rust"
  echo "================================================================"
  "$LAYER_BIN" "$HIDDEN" "$N_LAYERS" "$N_STEPS"
  echo
  echo "================================================================"
  echo "=== Layer-shaped workload: Python"
  echo "================================================================"
  python3 scripts/layer_shaped_microbench_py.py "$HIDDEN" "$N_LAYERS" "$N_STEPS"
}

case "$MODE" in
  single_op) run_single_op ;;
  layer_shaped) run_layer_shaped ;;
  all)
    run_single_op
    echo
    run_layer_shaped
    ;;
esac
