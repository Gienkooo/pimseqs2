#!/bin/bash
set -e

# =============================================================================
# Unified Build-and-Run Script for PIM-MMseqs2
# Usage: ./run.sh <mode> [options]
#
# Modes: kmer, ungapped, gapped, ungapped_gapped, all
# Options:
#   -n NUM_DPUS       Number of DPUs (default: env DPU_NUM_DPUS or 8)
#   -t TARGET_FASTA   Path to target FASTA (overrides default)
#   -q QUERY_FASTA    Path to query FASTA (overrides default)
#   -p                Enable profiling (sets DPU_PROFILE=1)
#   -h                Show this help
#
# Environment variables:
#   DPU_NUM_DPUS      Number of DPUs to use
#   DPU_PROFILE       Set to 1 to enable profiling output
#   E_VALUE           E-value threshold (default: 1000)
#   MAX_SEQS          Max sequences (default: 10000)
# =============================================================================

usage() {
    echo "Usage: $0 <mode> [-n NUM_DPUS] [-t TARGET_FASTA] [-q QUERY_FASTA] [-p] [-h]"
    echo ""
    echo "Modes:"
    echo "  kmer               Run k-mer prefilter validation"
    echo "  ungapped           Run ungapped prefilter validation"
    echo "  gapped             Run gapped prefilter validation"
    echo "  ungapped_gapped    Run combined ungapped+gapped validation"
    echo "  all                Run all modes sequentially"
    echo ""
    echo "Options:"
    echo "  -n NUM_DPUS        Number of DPUs to use (default: \$DPU_NUM_DPUS or 8)"
    echo "  -t TARGET_FASTA    Path to target FASTA file"
    echo "  -q QUERY_FASTA     Path to query FASTA file"
    echo "  -p                 Enable profiling output (DPU_PROFILE=1)"
    echo "  -h                 Show this help"
}

# --- Parse Mode ---
if [ $# -lt 1 ]; then
    usage
    exit 1
fi

MODE="$1"
shift

# --- Parse Options ---
DPU_COUNT=""
TARGET_FASTA_OVERRIDE=""
QUERY_FASTA_OVERRIDE=""

while getopts ":n:t:q:ph" opt; do
    case $opt in
        n) DPU_COUNT="$OPTARG" ;;
        t) TARGET_FASTA_OVERRIDE="$OPTARG" ;;
        q) QUERY_FASTA_OVERRIDE="$OPTARG" ;;
        p) export DPU_PROFILE=1 ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
    esac
done
shift $((OPTIND -1))

# --- Apply Overrides ---
if [ -n "$TARGET_FASTA_OVERRIDE" ]; then
    export TARGET_FASTA="$TARGET_FASTA_OVERRIDE"
fi
if [ -n "$QUERY_FASTA_OVERRIDE" ]; then
    export QUERY_FASTA="$QUERY_FASTA_OVERRIDE"
fi
if [ -n "$DPU_COUNT" ]; then
    if ! [[ "$DPU_COUNT" =~ ^[0-9]+$ ]] || [ "$DPU_COUNT" -le 0 ]; then
        echo "Invalid DPU count: $DPU_COUNT" >&2
        exit 1
    fi
    export DPU_NUM_DPUS="$DPU_COUNT"
fi

# --- Default DPU count ---
: "${DPU_NUM_DPUS:=8}"
export DPU_NUM_DPUS

# --- Default environment ---
: "${E_VALUE:=1000}"
: "${MAX_SEQS:=10000}"
export E_VALUE MAX_SEQS

# --- Build ---
echo "========================================"
echo "  Building project..."
echo "========================================"
mkdir -p build
cd build
make -j$(nproc)
cd ..
echo "Build complete."
echo ""

# --- Run Mode Functions ---
run_kmer() {
    echo "========================================"
    echo "  Running K-mer Prefilter Validation"
    echo "========================================"

    # K-mer specific config
    export MASK="${MASK:-1111111}"
    export KMER="${KMER:-7}"
    export EXACT_KMER_MATCHING="${EXACT_KMER_MATCHING:-1}"
    export SENSITIVITY="${SENSITIVITY:-3.0}"
    export CPU_TSV="scripts/validation/results/kmer/kmer_cpu.tsv"
    export DPU_TSV="scripts/validation/results/kmer/kmer_dpu.tsv"

    ./scripts/validation/validate_kmer_prefilter.sh -n "$DPU_NUM_DPUS"
}

run_ungapped() {
    echo "========================================"
    echo "  Running Ungapped Prefilter Validation"
    echo "========================================"
    time ./scripts/validation/validate_ungapped_prefilter.sh

    echo "Comparing results (CPU vs DPU)..."
    python3 scripts/validation/compare_results.py \
        scripts/validation/results/ungapped/ungapped_cpu.tsv \
        scripts/validation/results/ungapped/ungapped_dpu.tsv \
        UngappedPrefilter
}

run_gapped() {
    echo "========================================"
    echo "  Running Gapped Prefilter Validation"
    echo "========================================"
    time ./scripts/validation/validate_gapped_prefilter.sh

    echo "Comparing results (CPU vs DPU)..."
    python3 scripts/validation/compare_results.py \
        scripts/validation/results/gapped/gapped_cpu.tsv \
        scripts/validation/results/gapped/gapped_dpu.tsv \
        GappedPrefilter
}

run_ungapped_gapped() {
    echo "========================================"
    echo "  Running Ungapped+Gapped Validation"
    echo "========================================"
    time ./scripts/validation/validate_ungapped_gapped_prefilter.sh

    echo "Comparing results (DPU vs GPU)..."
    python3 scripts/validation/compare_results.py \
        scripts/validation/results/ungapped_gapped/ungapped_gapped_dpu.tsv \
        scripts/validation/results/ungapped_gapped/ungapped_gapped_gpu.tsv \
        UngappedGapped_DPU_vs_GPU
}

# --- Dispatch ---
case "$MODE" in
    kmer)
        run_kmer
        ;;
    ungapped)
        run_ungapped
        ;;
    gapped)
        run_gapped
        ;;
    ungapped_gapped)
        run_ungapped_gapped
        ;;
    all)
        run_kmer
        echo ""
        run_ungapped
        echo ""
        run_gapped
        echo ""
        run_ungapped_gapped
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "  Done."
echo "========================================"
