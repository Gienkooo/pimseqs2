#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

# Default datasets (can be overridden by environment variables)
LARGE_QUERY_FASTA="${QUERY_FASTA:-$ROOT_DIR/examples/QUERY.fasta}"
LARGE_TARGET_FASTA="${TARGET_FASTA:-$ROOT_DIR/examples/DB.fasta}"
QUERY_FASTA="${QUERY_FASTA:-$ROOT_DIR/examples/QUERY_small.fasta}"
TARGET_FASTA="${TARGET_FASTA:-$ROOT_DIR/examples/DB_small.fasta}"

# Output directory
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

QUERY_DB="$RESULTS_DIR/query_db"
TARGET_DB="$RESULTS_DIR/target_db"
TARGET_DB_PADDED="$RESULTS_DIR/target_db_padded"

# Ensure MMseqs2 is built
check_mmseqs() {
    if [ ! -x "$MMSEQS_BIN" ]; then
        echo "MMseqs2 binary not found at $MMSEQS_BIN. Please build it first."
    fi
}

# Prepare databases if they don't exist
prepare_dbs() {
    check_mmseqs
    
    echo "Creating query database..."
    head -n "$1" "$LARGE_QUERY_FASTA" > "$QUERY_FASTA"
    "$MMSEQS_BIN" createdb "$QUERY_FASTA" "$QUERY_DB" --mask 0 > /dev/null || echo "Failed to create query DB"

    echo "Creating target database..."
    head -n "$1" "$LARGE_TARGET_FASTA" > "$TARGET_FASTA"
    "$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0 > /dev/null || echo "Failed to create target DB"

    echo "Creating target padded database..."
    "$MMSEQS_BIN" makepaddedseqdb "$TARGET_DB" "$TARGET_DB_PADDED" > /dev/null || echo "Failed to create padded target DB"
}

OUT_DIR="$RESULTS_DIR/ungapped"
mkdir -p "$OUT_DIR"

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="1000"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="10000"
# Minimum ungapped score threshold (default 15, override with MIN_UNGAPPED env var)
MIN_UNGAPPED="15"

DB_SIZES=( 1000 5000 10000 )

for db_size in "${DB_SIZES[@]}"; do

    prepare_dbs "$db_size"

    BENCHMARK_RESULT="$OUT_DIR/bench_gpu_db_size_of_$db_size.json"

    CMD_GPU_STR="\"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB\" \"$TARGET_DB_PADDED\" \"$OUT_DIR/ungapped_gpu_db-$db_size\" \
    --prefilter-mode 1 --comp-bias-corr 0 --gpu 1 -v 3 \
    -e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --min-ungapped-score \"$MIN_UNGAPPED\" \
    2>&1 | tee \"$OUT_DIR/ungapped_gpu-$db_size.log\""

    hyperfine --warmup 0 \
                --runs 1 \
                --export-json "$BENCHMARK_RESULT" \
                --show-output \
                --command-name "Ungapped prefilter GPU with DB of size $db_size" \
                "$CMD_GPU_STR"

    "$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$OUT_DIR/ungapped_gpu_db-$db_size" "$OUT_DIR/ungapped_gpu-$db_size.tsv"

    echo "[BENCHMARK] Run succeeded. Saved result to $BENCHMARK_RESULT"

done
