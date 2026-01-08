#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

UNIREF_RANDOMIZED_TSV="${UNIREF_RANDOMIZED_FASTA:-$ROOT_DIR/examples/uniref50_randomized.tsv}"
QUERY_FASTA="${QUERY_FASTA:-$ROOT_DIR/examples/QUERY_uniref50_weak.fasta}"
TARGET_FASTA="${TARGET_FASTA:-$ROOT_DIR/examples/DB_uniref50_weak.fasta}"

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

OUT_DIR="$RESULTS_DIR/gapped"
mkdir -p "$OUT_DIR"

CPU_DB="$OUT_DIR/gapped_cpu_db"

check_mmseqs

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="1000"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="10000"

BASE_QUERY_DB_SIZE=50
BASE_TARGET_DB_SIZE=500
BASE_DPU_COUNT=100
MULTIPLIERS=( 1 2 4 8 16 )

for multiplier in "${MULTIPLIERS[@]}"; do
    TARGET_SIZE=$((multiplier*BASE_TARGET_DB_SIZE))
    QUERY_SIZE=$((multiplier*BASE_QUERY_DB_SIZE))
    DPU_COUNT=$((multiplier*BASE_DPU_COUNT))
    DPU_DB="$OUT_DIR/gapped_dpu_db-$DPU_COUNT"

    echo "0. Creating query database..."
    tail -n +$((TARGET_SIZE + 1)) "$UNIREF_RANDOMIZED_TSV" | head -n "$QUERY_SIZE" | tr "\t" "\n" > "$QUERY_FASTA"
    "$MMSEQS_BIN" createdb "$QUERY_FASTA" "$QUERY_DB" --mask 0 > /dev/null || echo "Failed to create query DB"

    echo "0. Creating target database..."
    head -n "$TARGET_SIZE" "$UNIREF_RANDOMIZED_TSV" | tr "\t" "\n" > "$TARGET_FASTA"
    "$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0 > /dev/null || echo "Failed to create target DB"

    CANDIDATE_DB="$OUT_DIR/candidate_db"
    rm -f "${CANDIDATE_DB}"*

    echo "1. Generating candidate pairs with CPU prefilter..."
    "$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB" "$CANDIDATE_DB" \
        --threads "$(nproc)" --prefilter-mode 2 --dpu 0 -v 3 \
        > "$OUT_DIR/candidate_gen.log" 2>&1
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        cat "$OUT_DIR/candidate_gen.log"
        echo "Candidate generation failed"
    fi
    echo "1. Done"

    echo "2. Running Gapped Alignment on CPU with candidate pairs..."
    "$MMSEQS_BIN" align "$QUERY_DB" "$TARGET_DB" "$CANDIDATE_DB" "$CPU_DB" \
        --threads "$(nproc)" -v 3 -e "$E_VALUE" --comp-bias-corr 0 \
        > "$OUT_DIR/gapped_cpu.log" 2>&1
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        cat "$OUT_DIR/gapped_cpu.log"
        echo "CPU run failed"
    fi
    echo "2. Done"

    CMD_DPU_STR="\"$MMSEQS_BIN\" ungappedprefilter \"$QUERY_DB\" \"$TARGET_DB\" \"$DPU_DB\" \
        --prefilter-mode 2 --comp-bias-corr 0 --dpu 1 -v 3 -e \"$E_VALUE\" --max-seqs \"$MAX_SEQS\" --dpu-num-dpus \"$DPU_COUNT\" 2>&1 | tee \"$OUT_DIR/gapped_dpu-$DPU_COUNT.log\""

    BENCHMARK_RESULT="$OUT_DIR/bench_dpu_gapped_params_dpus-$DPU_COUNT.json"

    echo "3. Running Gapped Alignment on DPU"
    hyperfine --warmup 0 \
                --runs 1 \
                --export-json "$BENCHMARK_RESULT" \
                --show-output \
                --command-name "Gapped prefilter on $DPU_COUNT DPUs" \
                "$CMD_DPU_STR"

    "$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$DPU_DB" "$OUT_DIR/gapped_dpu-$DPU_COUNT.tsv"

    echo "[BENCHMARK] Succeeded benchmarking. Saved result to $BENCHMARK_RESULT"

    rm -f "${CPU_DB}"* "${TARGET_DB}"* "${QUERY_DB}"*
done

