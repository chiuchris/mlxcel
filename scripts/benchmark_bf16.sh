#!/usr/bin/env bash
# bf16 Native Compute Validation Script
# Requires: CUDA GPU, built mlxcel binary
# Optional: nsys (for copy_v kernel profiling)
#
# Usage: ./scripts/benchmark_bf16.sh <model_path> [prompt]

set -euo pipefail

# Validate model path argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [prompt]"
    echo "  model_path: Path to the model directory (required)"
    echo "  prompt:     Prompt text (default: 'Explain quantum computing in simple terms.')"
    exit 1
fi

MODEL_PATH="$1"
PROMPT="${2:-Explain quantum computing in simple terms.}"
RUNS=3
BINARY="./target/release/mlxcel"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path '$MODEL_PATH' does not exist or is not a directory."
    exit 1
fi

echo "=== bf16 Native Compute Validation ==="
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo ""

# 1. Build verification
echo "--- Build Check ---"
cargo build --release --features cuda 2>&1 | tail -3
echo ""

if [ ! -x "$BINARY" ]; then
    echo "Error: Binary '$BINARY' not found or not executable after build."
    exit 1
fi

# 2. Baseline: fp32 conversion mode
echo "--- Baseline: fp32 conversion mode (MLX_BF16_NATIVE=0) ---"
for i in $(seq 1 $RUNS); do
    echo "Run $i/$RUNS:"
    MLX_BF16_NATIVE=0 $BINARY --model "$MODEL_PATH" --prompt "$PROMPT" --benchmark 2>&1 | grep -E "tok/s|memory|time" || echo "  (no matching output)"
done
echo ""

# 3. Native bf16 mode
echo "--- Native bf16 mode (MLX_BF16_NATIVE=1, default) ---"
for i in $(seq 1 $RUNS); do
    echo "Run $i/$RUNS:"
    MLX_BF16_NATIVE=1 $BINARY --model "$MODEL_PATH" --prompt "$PROMPT" --benchmark 2>&1 | grep -E "tok/s|memory|time" || echo "  (no matching output)"
done
echo ""

# 4. nsys profiling (if available)
if command -v nsys &> /dev/null; then
    echo "--- nsys Profiling: copy_v kernel count ---"

    echo "fp32 conversion mode:"
    MLX_BF16_NATIVE=0 nsys profile --stats=true -o /tmp/bf16_baseline -f true \
        $BINARY --model "$MODEL_PATH" --prompt "$PROMPT" 2>&1 \
        | { grep -c "copy_v" || echo "0"; } | xargs -I{} echo "  copy_v occurrences: {}"

    echo "Native bf16 mode:"
    MLX_BF16_NATIVE=1 nsys profile --stats=true -o /tmp/bf16_native -f true \
        $BINARY --model "$MODEL_PATH" --prompt "$PROMPT" 2>&1 \
        | { grep -c "copy_v" || echo "0"; } | xargs -I{} echo "  copy_v occurrences: {}"
else
    echo "nsys not found — skipping copy_v kernel profiling"
fi

echo ""
echo "=== Validation Complete ==="
