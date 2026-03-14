#!/bin/bash
# Quick benchmark of representative models with fixed scheduler
set -euo pipefail

MLXCEL="./target/release/mlxcel"
BENCH="./target/release/examples/batch_benchmark"
PROFILE="./target/release/examples/profile_batched_decode"
PORT=18080
OUTPUT="${1:-benchmark_batching_v2.log}"
MAX_TOKENS=100
RUNS=3
CONCURRENCY="1,2,4,8"

MODELS=(
  "smollm-135m"
  "qwen3-0.6b"
  "llama3-1b"
  "gemma2-2b"
  "qwen2-0.5b"
  "minicpm-2b"
  "phi2"
  "phi3-mini"
  "qwen2.5-7b"
  "llama3.1-8b-4bit"
  "deepseek-r1-7b"
  "command-r7b"
  "aya-8b"
)

echo "=== Continuous Batching Benchmark v2 (Fixed Scheduler) ===" | tee "$OUTPUT"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$OUTPUT"
echo "Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')" | tee -a "$OUTPUT"
echo "Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB", $1/1073741824}')" | tee -a "$OUTPUT"
echo "Config: max_tokens=$MAX_TOKENS, runs=$RUNS, concurrency=$CONCURRENCY, max_batch_size=8" | tee -a "$OUTPUT"
echo "========================================" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_DIR="./models/$MODEL_NAME"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "--- $MODEL_NAME: SKIP (not found) ---" | tee -a "$OUTPUT"
        continue
    fi

    echo "--- $MODEL_NAME ---" | tee -a "$OUTPUT"

    # Kill any existing server
    lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 1

    # Start server
    $MLXCEL serve -m "$MODEL_DIR" --max-batch-size 8 --port $PORT 2>/dev/null &
    SERVER_PID=$!

    READY=false
    for i in $(seq 1 60); do
        if curl -s --max-time 2 "http://localhost:$PORT/health" | grep -q "ok"; then
            READY=true
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then break; fi
        sleep 1
    done

    if [ "$READY" = false ]; then
        echo "  SKIP: Load timeout" | tee -a "$OUTPUT"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo "" | tee -a "$OUTPUT"
        continue
    fi

    BENCH_OUTPUT=$(timeout 300 $BENCH \
        --server "http://localhost:$PORT" \
        --concurrent "$CONCURRENCY" \
        --max-tokens $MAX_TOKENS \
        --runs $RUNS \
        --warmup 1 \
        --format table 2>&1) || true

    if echo "$BENCH_OUTPUT" | grep -q "Concurrency"; then
        RESULT=$(echo "$BENCH_OUTPUT" | sed -n '/^Concurrency/,/^$/p')
        echo "$RESULT" | tee -a "$OUTPUT"
    else
        echo "  FAIL" | tee -a "$OUTPUT"
    fi

    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 1
    echo "" | tee -a "$OUTPUT"
done

echo "=== Model-Level Profiling (forward vs forward_batched) ===" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

PROFILE_MODELS=("llama3.1-8b-4bit" "qwen3-0.6b" "qwen2.5-7b" "gemma2-2b" "phi3-mini")

for MODEL_NAME in "${PROFILE_MODELS[@]}"; do
    MODEL_DIR="./models/$MODEL_NAME"
    if [ ! -d "$MODEL_DIR" ]; then continue; fi

    echo "--- $MODEL_NAME (model-level) ---" | tee -a "$OUTPUT"
    PROF_OUT=$(timeout 300 $PROFILE \
        -m "$MODEL_DIR" \
        --batch-sizes 1,2,4,8 \
        --decode-steps 50 \
        --warmup 10 \
        --runs 3 2>&1) || true

    echo "$PROF_OUT" | grep -E "^(Batch|--|\d)" | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
done

echo "Done." | tee -a "$OUTPUT"
