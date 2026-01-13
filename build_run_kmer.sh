#!/bin/bash
set -e

export QUERY_FASTA="examples/QUERY.fasta"
export TARGET_FASTA="examples/DB.fasta"
export CPU_TSV="scripts/validation/results/kmer/kmer_cpu.tsv"
export DPU_TSV="scripts/validation/results/kmer/kmer_dpu.tsv"
export MASK="1111111"
export KMER=7

COMPARE_SCRIPT="scripts/validation/compare_results.py"
CHECKER_SCRIPT="scripts/validation/diagonal_checker.py"

echo "Building project..."
mkdir -p build
cd build
make -j$(nproc)
cd ..

echo "Running K-mer validation..."
./scripts/validation/validate_kmer_prefilter.sh

echo "Comparing results (CPU vs DPU)..."
python3 "$COMPARE_SCRIPT" \
    "$CPU_TSV" \
    "$DPU_TSV" \
    "KmerPrefilter"

echo "Verifying False Positives (Diagonal Check)..."
if [ -f "$CHECKER_SCRIPT" ]; then
    CPU_FP=$(python3 "$CHECKER_SCRIPT" --query "$QUERY_FASTA" --target "$TARGET_FASTA" --tsv "$CPU_TSV" --mask "$MASK")
    DPU_FP=$(python3 "$CHECKER_SCRIPT" --query "$QUERY_FASTA" --target "$TARGET_FASTA" --tsv "$DPU_TSV" --mask "$MASK")
    
    echo "A (CPU): ${CPU_FP}% false positives"
    echo "B (DPU): ${DPU_FP}% false positives"
else
    echo "WARNING: diagonal_checker.py not found at $CHECKER_SCRIPT. Skipping verification."
fi