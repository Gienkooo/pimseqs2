#!/usr/bin/env bash

set -euo pipefail

ORIG_DIR=$(pwd)
cleanup() {
    cd "$ORIG_DIR"
}
trap cleanup EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

FORCE=0
VERBOSITY=${VERBOSITY:-3}
PERF_MODE=${PERF_MODE:-1}

usage() {
    cat <<EOF
Usage: ${0##*/} [options]

Options:
  -f, --force        Rebuild databases and rerun CPU/DPU pipelines even if outputs exist.
      --verbosity N  Set MMseqs verbosity level (default: ${VERBOSITY}).
  -h, --help         Show this message.
EOF
}

while (($# > 0)); do
    case "$1" in
        -f|--force)
            FORCE=1
            ;;
        --verbosity)
            shift || { echo "Missing value for --verbosity" >&2; exit 1; }
            VERBOSITY="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -m|--mode)
            shift || { echo "Missing value for --mode" >&2; exit 1; }
            PERF_MODE="$1"
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

#THREADS=${THREADS:-4}
THREADS=$(nproc)
JOBS=${JOBS:-$(nproc)}
DPU_COUNT=${DPU_COUNT:-4}
TARGETS_PER_SHARD=${TARGETS_PER_SHARD:-16}
QUERIES_PER_BATCH=${QUERIES_PER_BATCH:-8}
export MMSEQS_DPU_REQUEST="$DPU_COUNT"

# QUERY_FASTA="$ROOT_DIR/examples/bfd-first_non_consensus_sequences.fasta"
QUERY_FASTA="$ROOT_DIR/examples/QUERY_small.fasta"
TARGET_FASTA="$ROOT_DIR/examples/DB_small.fasta"

TMP_BASE="$SCRIPT_DIR/tmp_compare"
CPU_TMP="$TMP_BASE/cpu"
DPU_TMP="$TMP_BASE/dpu"
TARGET_DB="$TMP_BASE/target_search"
TARGET_MSA="$TMP_BASE/target_msa"
CPU_RESULT="$TMP_BASE/example_result_cpu.m8"
DPU_RESULT="$TMP_BASE/example_result_dpu.m8"
CPU_MSA="$TMP_BASE/example_msas_cpu.a3m"
DPU_MSA="$TMP_BASE/example_msas_dpu.a3m"

mkdir -p "$TMP_BASE"

configure_build() {
    if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
        cmake -DENABLE_DPU=ON -DHAVE_DPU=1 -DCMAKE_BUILD_TYPE=Release -S "$ROOT_DIR" -B "$BUILD_DIR"
    fi
}

build_mmseqs() {
    echo "Compiling MMseqs2 (single build for CPU and DPU runs)..."
    cmake --build "$BUILD_DIR" --target mmseqs -- -j "$JOBS"
}

prepare_databases() {
    if [[ $FORCE -eq 0 && -f "${TARGET_DB}.dbtype" && -f "${TARGET_MSA}.dbtype" ]]; then
        echo "Skipping target database preparation (use --force to regenerate)."
        return
    fi

    echo "Preparing target databases..."
    rm -rf "$TARGET_DB"* "$TARGET_MSA"*
    "$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB"
    "$MMSEQS_BIN" cpdb "$TARGET_DB" "$TARGET_MSA"
}

run_pipeline() {
    local label=$1
    local tmp_dir=$2
    local result_file=$3
    local dpu_flag=$4
    local log_file="$TMP_BASE/mmseqs_${label}.log"

    echo "Running easy-search ($label)..."
    rm -rf "$tmp_dir"
    mkdir -p "$tmp_dir"

    : > "$log_file"
    local easy_cmd=("$MMSEQS_BIN" easy-search
        "$QUERY_FASTA" "$TARGET_DB"
        "$result_file" "$tmp_dir"
        --threads "$THREADS" --remove-tmp-files 0 --dpu "$dpu_flag" --prefilter-mode "$PERF_MODE"
        -v "$VERBOSITY")

    if [ "$dpu_flag" -ne 0 ]; then
        if ! env MMSEQS_DPU_FORCE_SIMULATOR=1 MMSEQS_DPU_PROFILE=backend=simulator \
            MMSEQS_DPU_MAX_TARGETS_PER_SHARD="$TARGETS_PER_SHARD" \
            MMSEQS_DPU_MAX_QUERIES_PER_BATCH="$QUERIES_PER_BATCH" \
            "${easy_cmd[@]}" 2>&1 | tee "$log_file"; then
            echo "easy-search ($label) failed; see $log_file for details." >&2
            exit 1
        fi
    else
        if ! "${easy_cmd[@]}" 2>&1 | tee "$log_file"; then
            echo "easy-search ($label) failed; see $log_file for details." >&2
            exit 1
        fi
    fi

    local run_dir
    run_dir=$(readlink -f "$tmp_dir/latest")

    echo "Generating MSA output ($label)..."
    local result2msa_cmd=("$MMSEQS_BIN" result2msa \
        "$run_dir/query" "$TARGET_MSA" "$run_dir/result" \
        "$tmp_dir/example_msas" --msa-format-mode 5 --threads "$THREADS")

    if [ "$dpu_flag" -ne 0 ]; then
        if ! env MMSEQS_DPU_FORCE_SIMULATOR=1 MMSEQS_DPU_PROFILE=backend=simulator \
            MMSEQS_DPU_MAX_TARGETS_PER_SHARD="$TARGETS_PER_SHARD" \
            MMSEQS_DPU_MAX_QUERIES_PER_BATCH="$QUERIES_PER_BATCH" \
            "${result2msa_cmd[@]}" 2>>"$log_file"; then
            echo "result2msa ($label) failed; see $log_file for details." >&2
            exit 1
        fi
    else
        if ! "${result2msa_cmd[@]}" 2>>"$log_file"; then
            echo "result2msa ($label) failed; see $log_file for details." >&2
            exit 1
        fi
    fi
    local view_cmd=("$MMSEQS_BIN" view "$tmp_dir/example_msas" --id-list 0)

    if [ "$dpu_flag" -ne 0 ]; then
        if ! env MMSEQS_DPU_FORCE_SIMULATOR=1 MMSEQS_DPU_PROFILE=backend=simulator \
            MMSEQS_DPU_MAX_TARGETS_PER_SHARD="$TARGETS_PER_SHARD" \
            MMSEQS_DPU_MAX_QUERIES_PER_BATCH="$QUERIES_PER_BATCH" \
            "${view_cmd[@]}" > "$TMP_BASE/example_msas_${label}.a3m" 2>>"$log_file"; then
            echo "view ($label) failed; see $log_file for details." >&2
            exit 1
        fi
    else
        if ! "${view_cmd[@]}" > "$TMP_BASE/example_msas_${label}.a3m" 2>>"$log_file"; then
            echo "view ($label) failed; see $log_file for details." >&2
            exit 1
        fi
    fi
}

compare_outputs() {
    echo "Comparing alignment outputs..."
    local cpu_sorted="$TMP_BASE/example_result_cpu.sorted.m8"
    local dpu_sorted="$TMP_BASE/example_result_dpu.sorted.m8"

    sort "$CPU_RESULT" > "$cpu_sorted"
    sort "$DPU_RESULT" > "$dpu_sorted"

    if ! diff -u "$cpu_sorted" "$dpu_sorted" > "$TMP_BASE/result.diff"; then
        echo "Alignment outputs differ (see $TMP_BASE/result.diff)."
        return 1
    fi

    echo "Comparing MSA outputs..."
    if ! diff -u "$CPU_MSA" "$DPU_MSA" > "$TMP_BASE/msa.diff"; then
        echo "MSA outputs differ (see $TMP_BASE/msa.diff)."
        return 1
    fi

    echo "CPU and DPU outputs match."
}

cd "$ROOT_DIR"
configure_build
build_mmseqs
prepare_databases

run_pipeline "cpu" "$CPU_TMP" "$CPU_RESULT" 0
run_pipeline "dpu" "$DPU_TMP" "$DPU_RESULT" 1

compare_outputs