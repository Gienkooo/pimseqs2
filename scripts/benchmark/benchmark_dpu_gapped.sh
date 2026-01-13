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
    fi
}

# Prepare databases if they don't exist
prepare_dbs() {
    check_mmseqs
    
    if [ ! -f "${QUERY_DB}.dbtype" ]; then
        echo "Creating query database from $QUERY_FASTA"
        "$MMSEQS_BIN" createdb "$QUERY_FASTA" "$QUERY_DB" --mask 0 > /dev/null || echo "Failed to create query DB"
    fi

    if [ ! -f "${TARGET_DB}.dbtype" ]; then
        echo "Creating target database from $TARGET_FASTA"
        "$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0 > /dev/null || echo "Failed to create target DB"
    fi
}

OUT_DIR="$RESULTS_DIR/gapped"
mkdir -p "$OUT_DIR"

prepare_dbs

QUERY_DB_SIZE=$(( $(wc -l < "$QUERY_FASTA") / 2 ))
TARGET_DB_SIZE=$(( $(wc -l < "$TARGET_FASTA") / 2 ))

CPU_RES="$OUT_DIR/gapped_cpu.tsv"
DPU_RES="$OUT_DIR/gapped_dpu.tsv"
CPU_DB="$OUT_DIR/gapped_cpu_db"
DPU_DB="$OUT_DIR/gapped_dpu_db"

# Clean up previous results
rm -f "${CPU_DB}"* "${DPU_DB}"* "${CPU_RES}" "${DPU_RES}"

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="${E_VALUE:-1000}"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="${MAX_SEQS:-10000}"

DPU_COUNTS="509,256,128,64,32"

CMD_DPU_STR="\"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB\" \"$TARGET_DB\" \"$OUT_DIR/gapped_dpu_db-{dpus}\" \
    --prefilter-mode 2 --comp-bias-corr 0 --dpu 1 -v 3 -e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --dpu-num-dpus \"{dpus}\" 2>&1 | tee \"$OUT_DIR/gapped_dpu-{dpus}.log\""

BENCHMARK_RESULT="$OUT_DIR/bench_dpu_gapped_params_dpus.json"

echo "3. Running Gapped Alignment on DPU"
hyperfine --warmup 0 \
            --runs 1 \
            --export-json "$BENCHMARK_RESULT" \
            --parameter-list dpus "$DPU_COUNTS" --show-output \
            --command-name "Gapped prefilter on {dpus} DPUs (db sizes: query $QUERY_DB_SIZE, target $TARGET_DB_SIZE)" \
            "$CMD_DPU_STR"

echo "[BENCHMARK] Succeeded benchmarking. Saved result to $BENCHMARK_RESULT"
