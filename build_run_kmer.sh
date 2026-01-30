#!/bin/bash
set -e

usage() {
    echo "Usage: $0 [-n NUM_DPUS] [-t TARGET_FASTA] [-q QUERY_FASTA]"
    echo "  -n NUM_DPUS     Number of DPUs to use (passed to validate script)"
    echo "  -t TARGET_FASTA Path to target FASTA (overrides default)"
    echo "  -q QUERY_FASTA  Path to query FASTA (overrides default)"
}

DPU_COUNT=""
TARGET_FASTA_OVERRIDE=""
QUERY_FASTA_OVERRIDE=""

while getopts ":n:t:q:h" opt; do
    case $opt in
        n) DPU_COUNT="$OPTARG" ;;
        t) TARGET_FASTA_OVERRIDE="$OPTARG" ;;
        q) QUERY_FASTA_OVERRIDE="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
    esac
done
shift $((OPTIND -1))

# Export Configuration (defaults can be overridden by args)
export QUERY_FASTA="${QUERY_FASTA_OVERRIDE:-examples/QUERY.fasta}"
export TARGET_FASTA="${TARGET_FASTA_OVERRIDE:-examples/DB.fasta}"
export CPU_TSV="scripts/validation/results/kmer/kmer_cpu.tsv"
export DPU_TSV="scripts/validation/results/kmer/kmer_dpu.tsv"

# Parameters
export MASK="1111111"
export KMER=7
export EXACT_KMER_MATCHING=1  
export SENSITIVITY=3.0

# Validate inputs
if [ ! -f "$QUERY_FASTA" ]; then
    echo "Query FASTA not found: $QUERY_FASTA" >&2
    exit 1
fi
if [ ! -f "$TARGET_FASTA" ]; then
    echo "Target FASTA not found: $TARGET_FASTA" >&2
    exit 1
fi
if [ -n "$DPU_COUNT" ]; then
    if ! [[ "$DPU_COUNT" =~ ^[0-9]+$ ]] || [ "$DPU_COUNT" -le 0 ]; then
        echo "Invalid DPU count: $DPU_COUNT" >&2
        exit 1
    fi
fi

echo "Building project..."
mkdir -p build
cd build
make -j$(nproc)
cd ..

echo "Running K-mer validation..."
if [ -n "$DPU_COUNT" ]; then
    ./scripts/validation/validate_kmer_prefilter.sh -n "$DPU_COUNT"
else
    ./scripts/validation/validate_kmer_prefilter.sh -n "8"
fi