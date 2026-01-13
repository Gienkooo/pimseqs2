#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
export MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

# Default datasets (can be overridden by environment variables)
export UNIREF_RANDOMIZED_TSV="${UNIREF_RANDOMIZED_TSV:-$ROOT_DIR/examples/uniref50_randomized.tsv}"
export QUERY_TSV="${QUERY_TSV:-$ROOT_DIR/examples/QUERY.tsv}"
export QUERY_FASTA="${QUERY_FASTA:-$ROOT_DIR/examples/QUERY_dpu.fasta}"
export TARGET_FASTA="${TARGET_FASTA:-$ROOT_DIR/examples/DB_dpu.fasta}"

# Output directory
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

export QUERY_DB="$RESULTS_DIR/query_db"
export TARGET_DB="$RESULTS_DIR/target_db"

# Ensure MMseqs2 is built
check_mmseqs() {
    if [ ! -x "$MMSEQS_BIN" ]; then
        echo "MMseqs2 binary not found at $MMSEQS_BIN. Please build it first."
    fi
}

export PREFILTER_MODE="3"
export PREFILTER_MODE_NAME="ungapped"

if [[ "$PREFILTER_MODE" == "3" ]]; then
    PREFILTER_MODE_NAME="ungapped_gapped"
fi

OUT_DIR="$RESULTS_DIR/$PREFILTER_MODE_NAME"
mkdir -p "$OUT_DIR"

export DPU_DB="$OUT_DIR/bench_dpu_${PREFILTER_MODE_NAME}_db"

# Prepare databases if they don't exist
prepare_dbs() {
    QUERY_SIZE=$1

    rm -f "${QUERY_DB}"* "${DPU_DB}"*

    echo "Creating query database..."
    head -n "$QUERY_SIZE" "$QUERY_TSV" | tr "\t" "\n" > "$QUERY_FASTA"
    "$MMSEQS_BIN" createdb "$QUERY_FASTA" "$QUERY_DB" --mask 0 > /dev/null || echo "Failed to create query DB"
    echo "Created query database"
}


check_mmseqs

export -f prepare_dbs

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="1000"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="10000"
# Minimum ungapped score threshold (default 15, override with MIN_UNGAPPED env var)
MIN_UNGAPPED="15"

BASE_QUERY_DB_SIZE=8
BASE_TARGET_DB_SIZE=200000
MULTIPLIERS="1,2,4,8"

BENCHMARK_RESULT="$OUT_DIR/bench_dpu_ungapped_growing_db.json"

CMD_DPU_STR="sudo \"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB\" \"$TARGET_DB\" \"$DPU_DB\" \
--prefilter-mode \"$PREFILTER_MODE\" --comp-bias-corr 0 --dpu 1 -v 3 \
-e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --min-ungapped-score \"$MIN_UNGAPPED\" \
--dpu-num-dpus 509 \
2>&1 | tee \"$OUT_DIR/bench_dpu_${PREFILTER_MODE_NAME}_growing_db_iter_{multiplier}.log\""

REPORT_CMD="\"$MMSEQS_BIN\" createtsv \"$QUERY_DB\" \"$TARGET_DB\" \"$DPU_DB\" \
\"$OUT_DIR/bench_dpu_${PREFILTER_MODE_NAME}_growing_db_iter_{multiplier}.tsv\""

echo "Creating target database..."
head -n "$BASE_TARGET_DB_SIZE" "$UNIREF_RANDOMIZED_TSV" | tr "\t" "\n" > "$TARGET_FASTA"
"$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0 > /dev/null || echo "Failed to create target DB"

hyperfine --warmup 0 \
            --runs 1 \
            --export-json "$BENCHMARK_RESULT" \
            --show-output \
            --shell bash \
            --parameter-list multiplier $MULTIPLIERS \
            --prepare "prepare_dbs \$(( {multiplier} * $BASE_QUERY_DB_SIZE ))" \
            --conclude "$REPORT_CMD" \
            --command-name "Weak scaling of $PREFILTER_MODE_NAME prefilter on 509 DPUs (db sizes: query {multiplier} * $BASE_QUERY_DB_SIZE, target $BASE_TARGET_DB_SIZE)" \
            "$CMD_DPU_STR"

echo "[BENCHMARK] Run succeeded. Saved result to $BENCHMARK_RESULT"
