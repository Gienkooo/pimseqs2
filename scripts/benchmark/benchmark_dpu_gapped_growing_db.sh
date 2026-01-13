#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
export MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

export UNIREF_RANDOMIZED_TSV="${UNIREF_RANDOMIZED_TSV:-$ROOT_DIR/examples/uniref50_randomized.tsv}"
export QUERY_FASTA="${QUERY_FASTA:-$ROOT_DIR/examples/QUERY_weak.fasta}"
TARGET_FASTA="${TARGET_FASTA:-$ROOT_DIR/examples/DB_weak.fasta}"

# Output directory
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

export QUERY_DB="$RESULTS_DIR/query_db"
TARGET_DB="$RESULTS_DIR/target_db"

OUT_DIR="$RESULTS_DIR/gapped"
mkdir -p "$OUT_DIR"

export DPU_DB="$OUT_DIR/gapped_dpu_db"

# Ensure MMseqs2 is built
check_mmseqs() {
    if [ ! -x "$MMSEQS_BIN" ]; then
        echo "MMseqs2 binary not found at $MMSEQS_BIN. Please build it first."
    fi
}

prepare_dbs() {
    QUERY_SIZE=$1
    TARGET_SIZE=$2

    rm -f "${QUERY_DB}"* "${DPU_DB}"*

    echo "Creating query database..."
    tail -n +$((TARGET_SIZE + 1)) "$UNIREF_RANDOMIZED_TSV" | head -n "$QUERY_SIZE" | tr "\t" "\n" > "$QUERY_FASTA"
    "$MMSEQS_BIN" createdb "$QUERY_FASTA" "$QUERY_DB" --mask 0 > /dev/null || echo "Failed to create query DB"
    echo "Created query database"
}

check_mmseqs

export -f prepare_dbs

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="1000"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="10000"

BASE_QUERY_DB_SIZE=5
BASE_TARGET_DB_SIZE=200000
BASE_DPU_COUNT=127
MULTIPLIERS="1,2,4,8,16"

BENCHMARK_RESULT="$OUT_DIR/bench_dpu_gapped_growing_db.json"

CMD_DPU_STR="\"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB\" \"$TARGET_DB\" \"$DPU_DB\" \
    --prefilter-mode 2 --comp-bias-corr 0 --dpu 1 -v 3 -e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --dpu-num-dpus \$(( {multiplier} * $BASE_DPU_COUNT )) 2>&1 | tee \"$OUT_DIR/gapped_dpu-$DPU_COUNT.log\""

REPORT_CMD="\"$MMSEQS_BIN\" createtsv \"$QUERY_DB\" \"$TARGET_DB\" \"$DPU_DB\" \"$OUT_DIR/gapped_dpu-$DPU_COUNT.tsv\""

echo "Creating target database..."
head -n "$BASE_TARGET_DB_SIZE" "$UNIREF_RANDOMIZED_TSV" | tr "\t" "\n" > "$TARGET_FASTA"
"$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0 > /dev/null || echo "Failed to create target DB"

echo "[BENCHMARK] Starting Hyperfine"

hyperfine --warmup 0 \
            --runs 2 \
            --export-json "$BENCHMARK_RESULT" \
            --show-output \
            --shell bash \
            --parameter-list multiplier $MULTIPLIERS \
            --prepare "prepare_dbs \$(( {multiplier} * $BASE_QUERY_DB_SIZE )) $BASE_TARGET_DB_SIZE" \
            --cleanup "$REPORT_CMD" \
            --command-name "Gapped prefilter on {multiplier} * $BASE_DPU_COUNT DPUs (db sizes: query {multiplier} * $BASE_QUERY_DB_SIZE, target $BASE_TARGET_DB_SIZE)" \
            "$CMD_DPU_STR"

echo "[BENCHMARK] Succeeded benchmarking. Saved result to $BENCHMARK_RESULT"
