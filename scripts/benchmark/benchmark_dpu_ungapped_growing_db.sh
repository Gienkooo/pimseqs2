#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

# Default datasets (can be overridden by environment variables)
UNIREF_RANDOMIZED_TSV="${UNIREF_RANDOMIZED_TSV:-$ROOT_DIR/examples/uniref50_randomized.tsv}"
QUERY_FASTA="${QUERY_FASTA:-$ROOT_DIR/examples/QUERY_dpu.fasta}"
TARGET_FASTA="${TARGET_FASTA:-$ROOT_DIR/examples/DB_dpu.fasta}"

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
    QUERY_SIZE=$1
    TARGET_SIZE=$2

    rm -f "${QUERY_DB}"* "${TARGET_DB}"*

    echo "Creating query database..."
    tail -n +$((TARGET_SIZE + 1)) "$UNIREF_RANDOMIZED_TSV" | head -n "$QUERY_SIZE" | tr "\t" "\n" > "$QUERY_FASTA"
    "$MMSEQS_BIN" createdb "$QUERY_FASTA" "$QUERY_DB" --mask 0 > /dev/null || echo "Failed to create query DB"

    echo "Creating target database..."
    head -n "$TARGET_SIZE" "$UNIREF_RANDOMIZED_TSV" | tr "\t" "\n" > "$TARGET_FASTA"
    "$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0 > /dev/null || echo "Failed to create target DB"
}

OUT_DIR="$RESULTS_DIR/ungapped"
mkdir -p "$OUT_DIR"

check_mmseqs

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="1000"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="10000"
# Minimum ungapped score threshold (default 15, override with MIN_UNGAPPED env var)
MIN_UNGAPPED="15"

BASE_QUERY_DB_SIZE=1000
BASE_TARGET_DB_SIZE=10000
BASE_DPU_COUNT=256
MULTIPLIERS=( 1 2 4 8 )

for multiplier in "${MULTIPLIERS[@]}"; do
    TARGET_SIZE=$((multiplier*BASE_TARGET_DB_SIZE))
    QUERY_SIZE=$((multiplier*BASE_QUERY_DB_SIZE))

    prepare_dbs "$QUERY_SIZE" "$TARGET_SIZE"

    DPU_COUNT=$((multiplier*BASE_DPU_COUNT))
    DPU_DB="$OUT_DIR/ungapped_dpu_db_size_${QUERY_SIZE}_dpus_${DPU_COUNT}"

    BENCHMARK_RESULT="$OUT_DIR/bench_dpu_query_db_size_${QUERY_SIZE}_dpus_${DPU_COUNT}.json"

    CMD_DPU_STR="\"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB\" \"$TARGET_DB\" \"$DPU_DB\" \
    --prefilter-mode 1 --comp-bias-corr 0 --dpu 1 -v 3 \
    -e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --min-ungapped-score \"$MIN_UNGAPPED\" --dpu-num-dpus \"$DPU_COUNT\" \
    2>&1 | tee \"$OUT_DIR/ungapped_dpu_db_size_${QUERY_SIZE}_dpus_${DPU_COUNT}.log\""

    hyperfine --warmup 0 \
                --runs 1 \
                --export-json "$BENCHMARK_RESULT" \
                --show-output \
                --prepare "rm -f \"$DPU_DB\"*" \
                --command-name "Ungapped prefilter on $DPU_COUNT DPUs with DB of size $QUERY_SIZE" \
                "$CMD_DPU_STR"

    "$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$DPU_DB" "$OUT_DIR/ungapped_dpu_db_size_$QUERY_SIZE.tsv"

    echo "[BENCHMARK] Run succeeded. Saved result to $BENCHMARK_RESULT"
done
