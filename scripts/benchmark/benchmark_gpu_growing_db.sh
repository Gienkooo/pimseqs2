#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
export MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

# Default datasets (can be overridden by environment variables)
UNIREF_RANDOMIZED_TSV="${UNIREF_RANDOMIZED_TSV:-$ROOT_DIR/examples/uniref50_randomized.tsv}"
export QUERY_TSV="${QUERY_TSV:-$ROOT_DIR/examples/QUERY.tsv}"
export QUERY_FASTA="${QUERY_FASTA:-$ROOT_DIR/examples/QUERY_weak.fasta}"
TARGET_FASTA="${TARGET_FASTA:-$ROOT_DIR/examples/DB_weak.fasta}"

# Output directory
export RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

export QUERY_DB="$RESULTS_DIR/query_db"
TARGET_DB="$RESULTS_DIR/target_db"
TARGET_DB_PADDED="$RESULTS_DIR/target_db_padded"

OUT_DIR="$RESULTS_DIR/ungapped_gapped"
mkdir -p "$OUT_DIR"

GPU_DB="$OUT_DIR/ungapped_gapped_gpu_db-{multiplier}"

HOOK_SCRIPT="$SCRIPT_DIR/gpu_energy_hook.sh"
MERGE_SCRIPT="$SCRIPT_DIR/merge_results.py"
ENERGY_LOG="$RESULTS_DIR/energy_captures.txt"

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
    MULTIPLIER=$2

    echo "Creating query database..."
    head -n "$QUERY_SIZE" "$QUERY_TSV" | tr "\t" "\n" > "$QUERY_FASTA"
    ACTUAL_COUNT=$(grep -c "^>" "$QUERY_FASTA" || true)
    echo "  [INFO] Created query DB with $ACTUAL_COUNT sequences (Requested: $QUERY_SIZE)"
    "$MMSEQS_BIN" createdb "$QUERY_FASTA" "${QUERY_DB}-${MULTIPLIER}" --mask 0 > /dev/null || echo "Failed to create query DB"
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

BASE_QUERY_DB_SIZE=8
BASE_TARGET_DB_SIZE=200000
MULTIPLIERS="1,2,4,8,16,32"

# Intermediate JSON (Times only)
BENCHMARK_RAW="$OUT_DIR/bench_gpu_raw.json"
# Final JSON (Times + Energy)
BENCHMARK_FINAL="$OUT_DIR/bench_gpu_growing_db.json"

CMD_GPU_STR="\"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB-{multiplier}\" \"$TARGET_DB_PADDED\" \"$GPU_DB\" \
--prefilter-mode 3 --comp-bias-corr 0 --gpu 1 -v 3 \
-e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --min-ungapped-score \"$MIN_UNGAPPED\" \
2>&1 | tee \"$OUT_DIR/ungapped_gapped_gpu_growing_db_iter_{multiplier}.log\""

REPORT_CMD="\"$MMSEQS_BIN\" createtsv \"$QUERY_DB-{multiplier}\" \"$TARGET_DB\" \"$GPU_DB\" \"$OUT_DIR/ungapped_gapped_gpu_growing_db_iter_{multiplier}.tsv\""

echo "Creating target database..."
head -n "$BASE_TARGET_DB_SIZE" "$UNIREF_RANDOMIZED_TSV" | tr "\t" "\n" > "$TARGET_FASTA"
"$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0 > /dev/null || echo "Failed to create target DB"

echo "Creating target padded database..."
"$MMSEQS_BIN" makepaddedseqdb "$TARGET_DB" "$TARGET_DB_PADDED" > /dev/null || echo "Failed to create padded target DB"

echo "[BENCHMARK] Starting Hyperfine with GPU Energy Monitoring..."

hyperfine --warmup 0 \
            --runs 1 \
            --export-json "$BENCHMARK_RAW" \
            --show-output \
            --shell bash \
            --parameter-list multiplier $MULTIPLIERS \
            --prepare "prepare_dbs \$(( {multiplier} * $BASE_QUERY_DB_SIZE )) {multiplier}; \"$HOOK_SCRIPT\" start" \
            --cleanup "\"$HOOK_SCRIPT\" stop; $REPORT_CMD" \
            --command-name "Ungapped+gapped prefilter on GPU (db sizes: query {multiplier} * $BASE_QUERY_DB_SIZE, target $BASE_TARGET_DB_SIZE)" \
            "$CMD_GPU_STR"

echo "[BENCHMARK] Merging Energy Data..."
python3 "$MERGE_SCRIPT" "$BENCHMARK_RAW" "$ENERGY_LOG" > "$BENCHMARK_FINAL"

echo "[BENCHMARK] Run succeeded."
echo "   Raw Data: $BENCHMARK_RAW"
echo "   Final Data (with Energy): $BENCHMARK_FINAL"
