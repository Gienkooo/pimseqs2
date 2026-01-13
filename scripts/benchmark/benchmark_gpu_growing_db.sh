#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
export MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

# Default datasets (can be overridden by environment variables)
export UNIREF_RANDOMIZED_TSV="${UNIREF_RANDOMIZED_TSV:-$ROOT_DIR/examples/uniref50_randomized.tsv}"
export QUERY_FASTA="${QUERY_FASTA:-$ROOT_DIR/examples/QUERY_gpu.fasta}"
export TARGET_FASTA="${TARGET_FASTA:-$ROOT_DIR/examples/DB_gpu.fasta}"

# Output directory
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

export QUERY_DB="$RESULTS_DIR/query_db"
export TARGET_DB="$RESULTS_DIR/target_db"
export TARGET_DB_PADDED="$RESULTS_DIR/target_db_padded"

OUT_DIR="$RESULTS_DIR/ungapped_gapped"
mkdir -p "$OUT_DIR"

export GPU_DB="$OUT_DIR/ungapped_gapped_gpu_db"

HOOK_SCRIPT="$SCRIPT_DIR/energy_hook.sh"
MERGE_SCRIPT="$SCRIPT_DIR/merge_results.py"
ENERGY_LOG="energy_captures.txt"

# Ensure MMseqs2 is built
check_mmseqs() {
    if [ ! -x "$MMSEQS_BIN" ]; then
        echo "MMseqs2 binary not found at $MMSEQS_BIN. Please build it first."
        exit 1
    fi
}

# Prepare databases if they don't exist
prepare_dbs() {
    QUERY_SIZE=$1
    TARGET_SIZE=$2

    rm -f "${QUERY_DB}"* "${TARGET_DB}"* "${TARGET_DB_PADDED}"* "${GPU_DB}"*

    echo "Creating query database..."
    tail -n +$((TARGET_SIZE + 1)) "$UNIREF_RANDOMIZED_TSV" | head -n "$QUERY_SIZE" | tr "\t" "\n" > "$QUERY_FASTA"
    "$MMSEQS_BIN" createdb "$QUERY_FASTA" "$QUERY_DB" --mask 0 > /dev/null || echo "Failed to create query DB"

    echo "Creating target database..."
    head -n "$TARGET_SIZE" "$UNIREF_RANDOMIZED_TSV" | tr "\t" "\n" > "$TARGET_FASTA"
    "$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0 > /dev/null || echo "Failed to create target DB"

    echo "Creating target padded database..."
    "$MMSEQS_BIN" makepaddedseqdb "$TARGET_DB" "$TARGET_DB_PADDED" > /dev/null || echo "Failed to create padded target DB"
}

check_mmseqs

export -f prepare_dbs

rm -f "$ENERGY_LOG"

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="1000"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="10000"
# Minimum ungapped score threshold (default 15, override with MIN_UNGAPPED env var)
MIN_UNGAPPED="15"

BASE_QUERY_DB_SIZE=10000
BASE_TARGET_DB_SIZE=100000
MULTIPLIERS="1,2,4"

# Intermediate JSON (Times only)
BENCHMARK_RAW="$OUT_DIR/bench_gpu_raw.json"
# Final JSON (Times + Energy)
BENCHMARK_FINAL="$OUT_DIR/bench_gpu_growing_db.json"

CMD_GPU_STR="\"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB\" \"$TARGET_DB_PADDED\" \"$GPU_DB\" \
--prefilter-mode 3 --comp-bias-corr 0 --gpu 1 -v 3 \
-e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --min-ungapped-score \"$MIN_UNGAPPED\" \
2>&1 | tee \"$OUT_DIR/ungapped_gapped_gpu_growing_db_iter_{multiplier}.log\""

REPORT_CMD="\"$MMSEQS_BIN\" createtsv \"$QUERY_DB\" \"$TARGET_DB\" \"$GPU_DB\" \"$OUT_DIR/ungapped_gapped_gpu_growing_db_iter_{multiplier}.tsv\""

echo "[BENCHMARK] Starting Hyperfine with GPU Energy Monitoring..."

hyperfine --warmup 0 \
            --runs 2 \
            --export-json "$BENCHMARK_RAW" \
            --show-output \
            --shell bash \
            --parameter-list multiplier $MULTIPLIERS \
            --prepare "prepare_dbs \$(( {multiplier} * $BASE_QUERY_DB_SIZE )) \$(( {multiplier} * $BASE_TARGET_DB_SIZE )); \"$HOOK_SCRIPT\" start" \
            --cleanup "\"$HOOK_SCRIPT\" stop; $REPORT_CMD" \
            --command-name "Ungapped+gapped prefilter on GPU" \
            "$CMD_GPU_STR"

echo "[BENCHMARK] Merging Energy Data..."
python3 "$MERGE_SCRIPT" "$BENCHMARK_RAW" "$ENERGY_LOG" > "$BENCHMARK_FINAL"

echo "[BENCHMARK] Run succeeded."
echo "   Raw Data: $BENCHMARK_RAW"
echo "   Final Data (with Energy): $BENCHMARK_FINAL"
