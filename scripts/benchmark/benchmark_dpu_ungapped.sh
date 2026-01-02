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

# Ensure MMseqs2 is built
check_mmseqs() {
    if [ ! -x "$MMSEQS_BIN" ]; then
        echo "MMseqs2 binary not found at $MMSEQS_BIN. Please build it first."
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

OUT_DIR="$RESULTS_DIR/ungapped"
mkdir -p "$OUT_DIR"

prepare_dbs

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="1000"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="10000"
# Minimum ungapped score threshold (default 15, override with MIN_UNGAPPED env var)
MIN_UNGAPPED="15"

DPU_COUNTS="1,2,4,16,64,256,512,1024,2048,2556"
DPU_COUNTS_FOR_LOOP=( 1 2 4 16 64 256 512 1024 2048 2556 )

CMD_DPU_STR="\"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB\" \"$TARGET_DB\" \"$OUT_DIR/ungapped_dpu_db-{dpus}\" \
--prefilter-mode 1 --comp-bias-corr 0 --dpu 1 -v 3 \
-e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --min-ungapped-score \"$MIN_UNGAPPED\" --dpu-num-dpus \"{dpus}\" \
2>&1 | tee \"$OUT_DIR/ungapped_dpu-{dpus}.log\""

BENCHMARK_RESULT="$OUT_DIR/bench_dpu_ungapped_params_dpus.json"

if [ -f "$BENCHMARK_RESULT" ]; then
    echo "[BENCHMARK] File $BENCHMARK_RESULT exists, will rename to $BENCHMARK_RESULT.old"
    mv "$BENCHMARK_RESULT" "$BENCHMARK_RESULT.old"
fi

hyperfine --warmup 0 \
            --runs 1 \
            --export-json "$BENCHMARK_RESULT" \
            --parameter-list dpus "$DPU_COUNTS" --show-output \
            --command-name "Ungapped prefilter on {dpus} DPUs" \
            "$CMD_DPU_STR"

for dpu_count in "${DPU_COUNTS_FOR_LOOP[@]}"; do
    "$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$OUT_DIR/ungapped_dpu_db-${dpu_count}" "$OUT_DIR/ungapped_dpu-${dpu_count}.tsv"
done

echo "[BENCHMARK] Succeeded benchmarking. Saved result to $BENCHMARK_RESULT"
