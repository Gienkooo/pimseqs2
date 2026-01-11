#!/bin/bash

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

HOOK_SCRIPT="$SCRIPT_DIR/cpu_energy_hook.sh"
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
    check_mmseqs

    if [ ! -f "${QUERY_DB}.dbtype" ]; then
        echo "Creating query database..."
        "$MMSEQS_BIN" createdb "$QUERY_FASTA" "$QUERY_DB" --mask 0
    fi

    if [ ! -f "${TARGET_DB}.dbtype" ]; then
        echo "Creating target database..."
        "$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0
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

THREADS_COUNT="8,16"
THREADS_FOR_LOOP=( 8 16 )

CMD_CPU_STR="\"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB\" \"$TARGET_DB\" \"$OUT_DIR/ungapped_cpu_db-{threads}\" --prefilter-mode 1 --comp-bias-corr 0 --threads {threads} -v 3 -e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --min-ungapped-score \"$MIN_UNGAPPED\" 2>&1 | tee \"$OUT_DIR/ungapped_cpu.log\""

BENCHMARK_RESULT="$ROOT_DIR/bench_cpu_ungapped_params_threads.json"
BENCHMARK_RAW="$ROOT_DIR/bench_cpu_raw.json"

if [ -f "$BENCHMARK_RESULT" ]; then
    echo "[BENCHMARK] File $BENCHMARK_RESULT exists, will rename to $BENCHMARK_RESULT.old"
    mv "$BENCHMARK_RESULT" "$BENCHMARK_RESULT.old"
fi

echo "[BENCHMARK] Result will be saved to $BENCHMARK_RESULT"

hyperfine --warmup 0 \
            --runs 2 \
            --export-json "$BENCHMARK_RAW" \
            --parameter-list threads "$THREADS_COUNT" --show-output \
            --prepare "rm -f \"$OUT_DIR/ungapped_cpu_db-{threads}\"*; \"$HOOK_SCRIPT\" start" \
            --cleanup "\"$HOOK_SCRIPT\" stop" \
            --command-name "Ungapped prefilter on CPU with {threads} threads" \
            "$CMD_CPU_STR"

for cpu_count in "${THREADS_FOR_LOOP[@]}"; do
    "$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$OUT_DIR/ungapped_cpu_db-${cpu_count}" "$OUT_DIR/ungapped_cpu-${cpu_count}.tsv"
done

echo "[BENCHMARK] Merging CPU Energy Data..."
python3 "$MERGE_SCRIPT" "$BENCHMARK_RAW" "$ENERGY_LOG" > "$BENCHMARK_RESULT"

echo "[BENCHMARK] Succeeded benchmarking. Saved result to $BENCHMARK_RESULT"
