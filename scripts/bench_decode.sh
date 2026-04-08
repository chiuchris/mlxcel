#!/usr/bin/env bash
# Decode benchmark with warmup preheating for accurate measurements.
#
# Usage:
#   ./scripts/bench_decode.sh models/llama3-1b-4bit
#   ./scripts/bench_decode.sh all
#   ./scripts/bench_decode.sh all --vlm --image tests/fixtures/test_image.png
#   ./scripts/bench_decode.sh all --output benchmarks/custom_name.csv
#
# Default output: benchmarks/{backend}_{hardware}_{YYYY-MM-DD}.csv
#   e.g. benchmarks/metal_m1ultra_2026-03-31.csv
#   VLM: benchmarks/metal_m1ultra_vlm_2026-03-31.csv
#
# The script runs each model twice: a warmup pass (discarded) to preheat Metal
# shader compilation and memory-mapping, followed by the measured pass.
# Both passes use --profile for structured timing output.
#
# CSV columns (14):
#   model, model_path, prompt_tokens, generated_tokens,
#   prefill_ms, prefill_tok_s, decode_ms, decode_tok_s,
#   date, hardware, mlx_version, build_type, max_tokens, prompt
#
# Filename convention:
#   {backend}_{hardware}_{YYYY-MM-DD}.csv        (text)
#   {backend}_{hardware}_vlm_{YYYY-MM-DD}.csv    (VLM)
#   Optional suffix: {backend}_{hardware}_{YYYY-MM-DD}_{suffix}.csv

set -euo pipefail

MLXCEL="./target/release/mlxcel"
MODELS_DIR="./models"
BENCHMARKS_DIR="./benchmarks"
TEXT_PROMPT="Hello, how are you today?"
VLM_PROMPT="What is in this image?"
MAX_TOKENS=100
WARMUP_TOKENS=20
TIMEOUT=300
JIT_PREHEAT_TIMEOUT=600
VLM_IMAGE="tests/fixtures/test_image.png"
VLM_MODE=0
OUTPUT=""
SUFFIX=""
DATE=$(date '+%Y-%m-%d')
MLX_VERSION="0.31.1"
BUILD_TYPE="release"

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

# Full hardware string for CSV content (e.g. Apple_M1_Ultra_128GB)
detect_hardware_full() {
  local chip mem
  if [[ "$(uname)" == "Darwin" ]]; then
    chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
    mem=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0fGB", $1/1073741824}')
  else
    # Linux: detect NVIDIA GPU or fall back to CPU
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    local cuda_ver
    cuda_ver=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' || echo "")
    if [[ -n "$gpu_name" ]]; then
      # Avoid double "NVIDIA" prefix if gpu_name already starts with it
      if [[ "$gpu_name" == NVIDIA* ]]; then
        chip="${gpu_name}"
      else
        chip="NVIDIA_${gpu_name}"
      fi
      [[ -n "$cuda_ver" ]] && chip="${chip}_CUDA${cuda_ver}"
    else
      chip=$(cat /proc/cpuinfo 2>/dev/null | grep "model name" | head -1 | sed 's/.*: //' || echo "unknown")
    fi
    mem=$(free -b 2>/dev/null | awk '/^Mem:/{printf "%.0fGB", $2/1073741824}' || echo "")
  fi
  echo "${chip}_${mem}" | tr ' ' '_'
}

# Short hardware name for filenames (e.g. m1ultra, m5max, gb10)
detect_hardware_short() {
  local full
  full=$(detect_hardware_full)

  # Map known hardware to short names
  case "$full" in
    *M1_Ultra*)  echo "m1ultra" ;;
    *M1_Max*)    echo "m1max" ;;
    *M1_Pro*)    echo "m1pro" ;;
    *M2_Ultra*)  echo "m2ultra" ;;
    *M2_Max*)    echo "m2max" ;;
    *M3_Ultra*)  echo "m3ultra" ;;
    *M3_Max*)    echo "m3max" ;;
    *M4_Ultra*)  echo "m4ultra" ;;
    *M4_Max*)    echo "m4max" ;;
    *M5_Ultra*)  echo "m5ultra" ;;
    *M5_Max*)    echo "m5max" ;;
    *GB10*)      echo "gb10" ;;
    *)           echo "${full}" | tr '[:upper:]' '[:lower:]' | tr ',' '_' | cut -c1-20 ;;
  esac
}

# Detect backend from binary / platform
detect_backend() {
  if [[ "$(uname)" == "Linux" ]] && nvidia-smi &>/dev/null; then
    echo "cuda"
  elif "$MLXCEL" generate --help 2>&1 | grep -q "cuda"; then
    echo "cuda"
  else
    echo "metal"
  fi
}

HARDWARE_FULL=$(detect_hardware_full)
HARDWARE_SHORT=$(detect_hardware_short)
BACKEND=$(detect_backend)

# ---------------------------------------------------------------------------
usage() {
  cat <<'EOF'
Usage: bench_decode.sh <model_path|all> [options]

Options:
  --vlm               VLM mode (use image prompt)
  --image PATH        Image for VLM benchmark (default: tests/fixtures/test_image.png)
  --prompt TEXT        Override text prompt
  --max-tokens N      Max tokens to generate (default: 100)
  --warmup-tokens N   Tokens for warmup pass (default: 20)
  --timeout N         Timeout per run in seconds (default: 300)
  --jit-preheat-timeout N  Timeout for CUDA JIT warmup in seconds (default: 600)
  --output PATH       Write CSV to specific file (overrides auto-naming)
  --suffix TAG        Append suffix to auto-generated filename (e.g. --suffix baseline)
  --help              Show this help

Filename convention:
  {backend}_{hardware}_{YYYY-MM-DD}.csv           text benchmarks
  {backend}_{hardware}_vlm_{YYYY-MM-DD}.csv       VLM benchmarks
  {backend}_{hardware}_{YYYY-MM-DD}_{suffix}.csv  with --suffix
EOF
}

# ---------------------------------------------------------------------------
# Generate default output filename
# ---------------------------------------------------------------------------
default_output_path() {
  local name="${BACKEND}_${HARDWARE_SHORT}"
  if [[ "$VLM_MODE" -eq 1 ]]; then
    name="${name}_vlm"
  fi
  name="${name}_${DATE}"
  if [[ -n "$SUFFIX" ]]; then
    name="${name}_${SUFFIX}"
  fi
  echo "${BENCHMARKS_DIR}/${name}.csv"
}

# ---------------------------------------------------------------------------
# Parse --profile output into CSV fields
# ---------------------------------------------------------------------------
parse_profile() {
  local output="$1"
  local prompt_tok gen_tok prefill_ms prefill_tps decode_ms decode_tps

  prompt_tok=$(echo "$output" | sed -n 's/.*Prompt tokens:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
  gen_tok=$(echo "$output"    | sed -n 's/.*Generated tokens:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
  prefill_ms=$(echo "$output" | sed -n 's/.*Prefill:[[:space:]]*\([0-9.]*\) ms.*/\1/p' | head -1)
  prefill_tps=$(echo "$output" | sed -n 's/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)
  decode_ms=$(echo "$output"  | sed -n 's/.*Decode:[[:space:]]*\([0-9.]*\) ms.*/\1/p' | head -1)
  decode_tps=$(echo "$output" | sed -n 's/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)

  echo "${prompt_tok:-},${gen_tok:-},${prefill_ms:-},${prefill_tps:-},${decode_ms:-},${decode_tps:-}"
}

# ---------------------------------------------------------------------------
# Benchmark a single model
# ---------------------------------------------------------------------------
bench_one() {
  local model_path="$1"
  local model_name
  model_name=$(basename "$model_path")

  local prompt="$TEXT_PROMPT"
  local extra_args=()
  if [[ "$VLM_MODE" -eq 1 ]]; then
    prompt="$VLM_PROMPT"
    if [[ ! -f "$VLM_IMAGE" ]]; then
      echo "${model_name},${model_path},,,,,,,$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$prompt\",SKIP:vlm_image_not_found"
      return
    fi
    extra_args+=(--image "$VLM_IMAGE")
  fi

  # --- Warmup pass (preheat Metal shaders / CUDA JIT kernels & memory maps) ---
  # On CUDA, the first run of a model triggers JIT kernel compilation which can
  # take several minutes. Use JIT_PREHEAT_TIMEOUT (default 600s) for the warmup
  # pass so these models are not incorrectly marked as FAIL:warmup.
  local warmup_timeout="$TIMEOUT"
  if [[ "$BACKEND" == "cuda" ]]; then
    warmup_timeout="$JIT_PREHEAT_TIMEOUT"
  fi
  >&2 printf '>>> [warmup] %s ...\n' "$model_name"
  if ! timeout "$warmup_timeout" "$MLXCEL" generate \
      -m "$model_path" -p "$prompt" -n "$WARMUP_TOKENS" \
      ${extra_args[@]+"${extra_args[@]}"} --profile >/dev/null 2>&1; then
    >&2 echo "    warmup failed"
    echo "${model_name},${model_path},,,,,,,$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$prompt\",FAIL:warmup"
    return
  fi

  # --- Measured pass ---
  >&2 printf '>>> [bench]  %s ...\n' "$model_name"
  local raw
  if ! raw=$(timeout "$TIMEOUT" "$MLXCEL" generate \
      -m "$model_path" -p "$prompt" -n "$MAX_TOKENS" \
      ${extra_args[@]+"${extra_args[@]}"} --profile 2>&1); then
    >&2 echo "    benchmark failed"
    echo "${model_name},${model_path},,,,,,,$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$prompt\",FAIL:bench"
    return
  fi

  local fields
  fields=$(parse_profile "$raw")
  local decode_tps
  decode_tps=$(echo "$fields" | cut -d, -f6)

  if [[ -z "$decode_tps" ]]; then
    >&2 echo "    no decode output"
    echo "${model_name},${model_path},${fields},$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$prompt\",FAIL:no_output"
  else
    >&2 printf '    decode: %s tok/s\n' "$decode_tps"
    echo "${model_name},${model_path},${fields},$DATE,$HARDWARE_FULL,$MLX_VERSION,$BUILD_TYPE,$MAX_TOKENS,\"$prompt\""
  fi
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODEL_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vlm)            VLM_MODE=1; shift ;;
    --image)          VLM_IMAGE="$2"; VLM_MODE=1; shift 2 ;;
    --prompt)         TEXT_PROMPT="$2"; shift 2 ;;
    --max-tokens)     MAX_TOKENS="$2"; shift 2 ;;
    --warmup-tokens)  WARMUP_TOKENS="$2"; shift 2 ;;
    --timeout)        TIMEOUT="$2"; shift 2 ;;
    --jit-preheat-timeout) JIT_PREHEAT_TIMEOUT="$2"; shift 2 ;;
    --output)         OUTPUT="$2"; shift 2 ;;
    --suffix)         SUFFIX="$2"; shift 2 ;;
    --help)           usage; exit 0 ;;
    -*)               echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)                MODEL_ARG="$1"; shift ;;
  esac
done

if [[ -z "$MODEL_ARG" ]]; then
  echo "Error: model path or 'all' required" >&2
  usage >&2
  exit 1
fi

if [[ ! -x "$MLXCEL" ]]; then
  echo "Error: $MLXCEL not found. Run 'cargo build --release' first." >&2
  exit 1
fi

# Auto-generate output path if not specified
if [[ -z "$OUTPUT" ]]; then
  OUTPUT=$(default_output_path)
fi

# ---------------------------------------------------------------------------
# CSV header
# ---------------------------------------------------------------------------
CSV_HEADER="model,model_path,prompt_tokens,generated_tokens,prefill_ms,prefill_tok_s,decode_ms,decode_tok_s,date,hardware,mlx_version,build_type,max_tokens,prompt"

mkdir -p "$(dirname "$OUTPUT")"
echo "$CSV_HEADER" > "$OUTPUT"
>&2 echo "Output: $OUTPUT"
>&2 echo "Hardware: $HARDWARE_FULL ($HARDWARE_SHORT)"
>&2 echo "Backend: $BACKEND"
>&2 echo ""

emit() {
  echo "$1" | tee -a "$OUTPUT"
}

# ---------------------------------------------------------------------------
# Models known to crash the GPU (Metal timeout / address fault).
# Run these LAST to prevent GPU state corruption from affecting other models.
# ---------------------------------------------------------------------------
GPU_CRASH_MODELS="gemma3n-e4b-bf16"

is_gpu_crash_model() {
  local name
  name=$(basename "$1")
  for m in $GPU_CRASH_MODELS; do
    [[ "$name" == "$m" ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CUDA JIT preheat: on first run after build, CUDA JIT-compiles many kernels
# (binary ops, reduce, etc.) which can take 3-8 minutes per model. Run one
# small model first so shared kernels are cached before the benchmark loop.
# ---------------------------------------------------------------------------
if [[ "$BACKEND" == "cuda" && "$MODEL_ARG" == "all" ]]; then
  # Find a small model to preheat with
  preheat_model=""
  for candidate in smollm-135m-4bit ernie-4.5-0.3b-4bit qwen2.5-0.5b-bf16; do
    if [[ -d "$MODELS_DIR/$candidate" ]]; then
      preheat_model="$MODELS_DIR/$candidate"
      break
    fi
  done
  if [[ -n "$preheat_model" ]]; then
    >&2 echo "=== CUDA JIT preheat: $(basename "$preheat_model") ==="
    >&2 echo "    First run compiles CUDA JIT kernels (may take several minutes)..."
    if timeout "$JIT_PREHEAT_TIMEOUT" "$MLXCEL" generate \
        -m "$preheat_model" -p "Hello" -n 5 --profile >/dev/null 2>&1; then
      >&2 echo "    JIT preheat complete."
    else
      >&2 echo "    JIT preheat failed (non-fatal, continuing)."
    fi
    >&2 echo ""
  fi
fi

if [[ "$MODEL_ARG" == "all" ]]; then
  # First pass: run all models except known GPU-crash models
  for dir in "$MODELS_DIR"/*/; do
    [[ -d "$dir" ]] || continue
    is_gpu_crash_model "$dir" && continue
    result=$(bench_one "$dir")
    emit "$result"
  done
  # Second pass: run known GPU-crash models last
  for dir in "$MODELS_DIR"/*/; do
    [[ -d "$dir" ]] || continue
    is_gpu_crash_model "$dir" || continue
    result=$(bench_one "$dir")
    emit "$result"
  done
else
  if [[ ! -d "$MODEL_ARG" ]]; then
    echo "Error: model directory '$MODEL_ARG' not found" >&2
    exit 1
  fi
  result=$(bench_one "$MODEL_ARG")
  emit "$result"
fi

>&2 echo ""
>&2 echo "Results saved to: $OUTPUT"
