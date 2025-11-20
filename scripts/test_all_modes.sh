#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

THREADS=$(nproc)
DPU_COUNT=${DPU_COUNT:-4}
TARGETS_PER_SHARD=16
QUERIES_PER_BATCH=8
VERBOSITY=3
FORCE=0
SPECIFIC_MODE=""

export MMSEQS_DPU_REQUEST="$DPU_COUNT"

usage() {
    echo "Usage: $0 [-f] [-m MODE] [--verbosity N]"
    echo "  -f           Force rebuild of database"
    echo "  -m MODE      0=Kmer, 1=Ungapped, 2=Gapped. (Default: Run All)"
    exit 1
}

while (($# > 0)); do
    case "$1" in
        -f) FORCE=1 ;;
        -m) shift; SPECIFIC_MODE="$1" ;;
        --verbosity) shift; VERBOSITY="$1" ;;
        *) usage ;;
    esac
    shift
done

QUERY_FASTA="$ROOT_DIR/examples/QUERY_small.fasta"
TARGET_FASTA="$ROOT_DIR/examples/DB_small.fasta"
TARGET_DB="$SCRIPT_DIR/tmp_compare/target_db"

echo "Configuring CMake (Ensuring DPU is enabled)..."
cmake -DENABLE_DPU=ON -DHAVE_DPU=1 -DCMAKE_BUILD_TYPE=Release -S "$ROOT_DIR" -B "$BUILD_DIR"

echo "Building MMseqs2..."
cmake --build "$BUILD_DIR" --target mmseqs -- -j "$THREADS"

echo "Building DPU Kernels..."
cmake --build "$BUILD_DIR" --target dpu_kernels

mkdir -p "$SCRIPT_DIR/tmp_compare"
if [[ $FORCE -eq 1 ]] || [[ ! -f "${TARGET_DB}.dbtype" ]]; then
    echo "Creating Database..."
    rm -rf "$TARGET_DB"*
    "$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" -v 0
fi

run_mode() {
    local MODE=$1
    local MODE_NAME=""
    case $MODE in
        0) MODE_NAME="K-MER" ;;
        1) MODE_NAME="UNGAPPED" ;;
        2) MODE_NAME="GAPPED" ;;
    esac

    echo "=========================================================="
    echo "  TESTING MODE $MODE: $MODE_NAME"
    echo "=========================================================="

    local TMP_DIR="$SCRIPT_DIR/tmp_mode${MODE}"
    mkdir -p "$TMP_DIR"
    
    local CPU_RES="$TMP_DIR/cpu.m8"
    local DPU_RES="$TMP_DIR/dpu.m8"
    local CPU_MSA="$TMP_DIR/cpu.a3m"
    local DPU_MSA="$TMP_DIR/dpu.a3m"

    # CPU RUN
    echo "Running CPU..."
    "$MMSEQS_BIN" easy-search "$QUERY_FASTA" "$TARGET_DB" "$CPU_RES" "$TMP_DIR/cpu_tmp" \
        --threads "$THREADS" --dpu 0 --prefilter-mode "$MODE" -v "$VERBOSITY" --remove-tmp-files 0 >/dev/null

    # Generate CPU MSA
    "$MMSEQS_BIN" result2msa "$TMP_DIR/cpu_tmp/latest/query" "$TARGET_DB" "$TMP_DIR/cpu_tmp/latest/result" \
        "$TMP_DIR/cpu_msa" --msa-format-mode 5 --threads "$THREADS" >/dev/null 2>&1
    "$MMSEQS_BIN" view "$TMP_DIR/cpu_msa" --id-list 0 > "$CPU_MSA"

    # DPU RUN 
    echo "Running DPU..."
    if ! env MMSEQS_DPU_FORCE_SIMULATOR=1 MMSEQS_DPU_PROFILE="backend=simulator" \
         MMSEQS_DPU_MAX_TARGETS_PER_SHARD="$TARGETS_PER_SHARD" \
         MMSEQS_DPU_MAX_QUERIES_PER_BATCH="$QUERIES_PER_BATCH" \
         "$MMSEQS_BIN" easy-search "$QUERY_FASTA" "$TARGET_DB" "$DPU_RES" "$TMP_DIR/dpu_tmp" \
            --threads "$THREADS" --dpu 1 --prefilter-mode "$MODE" -v "$VERBOSITY" --remove-tmp-files 0 2>&1 | tee "$TMP_DIR/dpu.log"; then
        echo "❌ DPU Run Failed! See $TMP_DIR/dpu.log"
        return 1
    fi

    # Generate DPU MSA
    "$MMSEQS_BIN" result2msa "$TMP_DIR/dpu_tmp/latest/query" "$TARGET_DB" "$TMP_DIR/dpu_tmp/latest/result" \
        "$TMP_DIR/dpu_msa" --msa-format-mode 5 --threads "$THREADS" >/dev/null 2>&1
    "$MMSEQS_BIN" view "$TMP_DIR/dpu_msa" --id-list 0 > "$DPU_MSA"

    # COMPARE
    sort "$CPU_RES" > "${CPU_RES}.s"
    sort "$DPU_RES" > "${DPU_RES}.s"

    echo "Results:"
    if diff -u "${CPU_RES}.s" "${DPU_RES}.s" > "$TMP_DIR/diff_m8.txt"; then
        echo "  ✅ Mode $MODE ($MODE_NAME): Alignment Matches"
    else
        echo "  ❌ Mode $MODE ($MODE_NAME): Alignment Mismatch"
        head -n 5 "$TMP_DIR/diff_m8.txt"
    fi

    if diff -u "$CPU_MSA" "$DPU_MSA" > "$TMP_DIR/diff_msa.txt"; then
        echo "  ✅ Mode $MODE ($MODE_NAME): MSA Matches"
    else
        echo "  ❌ Mode $MODE ($MODE_NAME): MSA Mismatch"
    fi
    echo ""
}

if [ -n "$SPECIFIC_MODE" ]; then
    run_mode "$SPECIFIC_MODE"
else
    cd "$ROOT_DIR"

    run_mode 0
    run_mode 1
    run_mode 2
fi