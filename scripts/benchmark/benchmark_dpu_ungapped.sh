#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

# Default datasets (can be overridden by environment variables)
QUERY_FASTA="${QUERY_FASTA:-$ROOT_DIR/examples/QUERY_small.fasta}"
TARGET_FASTA="${TARGET_FASTA:-$ROOT_DIR/examples/DB_small.fasta}"

# Output directory
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

QUERY_DB="$RESULTS_DIR/query_db"
TARGET_DB="$RESULTS_DIR/target_db"

HOOK_SCRIPT="$SCRIPT_DIR/dram_energy_hook.sh"
MERGE_SCRIPT="$SCRIPT_DIR/merge_results.py"
ENERGY_LOG="energy_captures.txt"

ENABLE_ENERGY="${ENABLE_ENERGY:-false}"

# Ensure MMseqs2 is built
check_mmseqs() {
    if [ ! -x "$MMSEQS_BIN" ]; then
        echo "MMseqs2 binary not found at $MMSEQS_BIN. Please build it first."
        exit 1
    fi
}

# Prepare databases if they don't exist
prepare_dbs() {
    check_mmseqs
    
    if [ ! -f "${QUERY_DB}.dbtype" ]; then
        echo "Creating query database..."
        "$MMSEQS_BIN" createdb "$QUERY_FASTA" "$QUERY_DB" --mask 0 > /dev/null || echo "Failed to create query DB"
    fi

    if [ ! -f "${TARGET_DB}.dbtype" ]; then
        echo "Creating target database..."
        "$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0 > /dev/null || echo "Failed to create target DB"
    fi
}

PREFILTER_MODE="1"
PREFILTER_MODE_NAME="ungapped"

if [[ "$PREFILTER_MODE" == "3" ]]; then
    PREFILTER_MODE_NAME="ungapped_gapped"
fi

OUT_DIR="$RESULTS_DIR/$PREFILTER_MODE_NAME"
mkdir -p "$OUT_DIR"

prepare_dbs

rm -f "$ENERGY_LOG"

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="1000"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="10000"
# Minimum ungapped score threshold (default 15, override with MIN_UNGAPPED env var)
MIN_UNGAPPED="15"

DPU_COUNTS="2048,1024,512,256,128,64,2496"
DPU_COUNTS_FOR_LOOP=( 2048 1024 512 256 128 64 2496 )

CMD_DPU_STR="\"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB\" \"$TARGET_DB\" \"$OUT_DIR/${PREFILTER_MODE_NAME}_dpu_db-{dpus}\" \
--prefilter-mode \"$PREFILTER_MODE\" --comp-bias-corr 0 --dpu 1 -v 3 \
-e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --min-ungapped-score \"$MIN_UNGAPPED\" --dpu-num-dpus \"{dpus}\" \
2>&1 | tee \"$OUT_DIR/${PREFILTER_MODE_NAME}_dpu-{dpus}.log\""

BENCHMARK_RESULT="$OUT_DIR/bench_dpu_${PREFILTER_MODE_NAME}_params_dpus.json"
BENCHMARK_RAW="$OUT_DIR/bench_dpu_raw.json"

if [ -f "$BENCHMARK_RESULT" ]; then
    echo "[BENCHMARK] File $BENCHMARK_RESULT exists, will rename to $BENCHMARK_RESULT.old"
    mv "$BENCHMARK_RESULT" "$BENCHMARK_RESULT.old"
fi

HF_ARGS=(
    --warmup 0
    --runs 1
    --export-json "$BENCHMARK_RAW"
    --parameter-list dpus "$DPU_COUNTS"
    --show-output
    --command-name "Mode $PREFILTER_MODE prefilter on {dpus} DPUs"
)

if [ "$ENABLE_ENERGY" == "true" ]; then
    echo "[BENCHMARK] Energy Measurement ENABLED (DRAM)"
    HF_ARGS+=(--prepare "\"$HOOK_SCRIPT\" start")
    HF_ARGS+=(--cleanup "\"$HOOK_SCRIPT\" stop")
else
    echo "[BENCHMARK] Energy Measurement DISABLED (Time only)"
fi

hyperfine "${HF_ARGS[@]}" "$CMD_DPU_STR"

for dpu_count in "${DPU_COUNTS_FOR_LOOP[@]}"; do
    "$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$OUT_DIR/${PREFILTER_MODE_NAME}_dpu_db-${dpu_count}" "$OUT_DIR/${PREFILTER_MODE_NAME}_dpu-${dpu_count}.tsv"
done

if [ "$ENABLE_ENERGY" == "true" ]; then
    echo "[BENCHMARK] Merging DRAM Energy Data..."
    python3 "$MERGE_SCRIPT" "$BENCHMARK_RAW" "$ENERGY_LOG" > "$BENCHMARK_RESULT"
else
    cp "$BENCHMARK_RAW" "$BENCHMARK_RESULT"
fi

echo "[BENCHMARK] Succeeded benchmarking. Saved result to $BENCHMARK_RESULT"
