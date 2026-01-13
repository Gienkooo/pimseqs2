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

OUT_DIR="$RESULTS_DIR/gapped"
mkdir -p "$OUT_DIR"

E_VALUE="${E_VALUE:-1000}"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="${MAX_SEQS:-10000}"

echo "Using E-value threshold: $E_VALUE, max-seqs: $MAX_SEQS"

prepare_dbs

QUERY_DB_SIZE=$(( $(wc -l < "$QUERY_FASTA") / 2 ))
TARGET_DB_SIZE=$(( $(wc -l < "$TARGET_FASTA") / 2 ))


CANDIDATE_DB="$OUT_DIR/candidate_db"
rm -f "${CANDIDATE_DB}"*

CPU_DB="$OUT_DIR/gapped_cpu_db-{threads}"

echo "Generating candidate pairs with CPU prefilter..."
"$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB" "$CANDIDATE_DB" \
    --threads "$(nproc)" --prefilter-mode 2 --dpu 0 -v 3 \
    > "$OUT_DIR/candidate_gen.log" 2>&1
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    cat "$OUT_DIR/candidate_gen.log"
    error "Candidate generation failed"
fi

THREADS_COUNT="16,12,8,6,4,2,1"

CMD_CPU_STR="\"$MMSEQS_BIN\" align \"$QUERY_DB\" \"$TARGET_DB\" \"$CANDIDATE_DB\" \"$CPU_DB\" --threads {threads} -v 3 -e \"$E_VALUE\" --comp-bias-corr 0 2>&1 | tee \"$OUT_DIR/gapped_cpu.log\""
    
BENCHMARK_RESULT="$ROOT_DIR/bench_cpu_gapped_params_threads.json"

if [ -f "$BENCHMARK_RESULT" ]; then
    echo "[BENCHMARK] File $BENCHMARK_RESULT exists, will rename to $BENCHMARK_RESULT.old"
    mv "$BENCHMARK_RESULT" "$BENCHMARK_RESULT.old"
fi

echo "[BENCHMARK] Result will be saved to $BENCHMARK_RESULT"

hyperfine --warmup 0 \
            --runs 2 \
            --export-json "$BENCHMARK_RESULT" \
            --parameter-list threads "$THREADS_COUNT" \
            --show-output \
            --command-name "Gapped align on CPU with {threads} threads (db sizes: query $QUERY_DB_SIZE, target $TARGET_DB_SIZE)" \
            "$CMD_CPU_STR"

echo "[BENCHMARK] Succeeded benchmarking. Saved result to $BENCHMARK_RESULT"
