#!/bin/bash
set -e

# Export Configuration
export QUERY_FASTA="examples/QUERY.fasta"
export TARGET_FASTA="examples/DB.fasta"
export CPU_TSV="scripts/validation/results/kmer/kmer_cpu.tsv"
export DPU_TSV="scripts/validation/results/kmer/kmer_dpu.tsv"

# Parameters
export MASK="1111111"
export KMER=7
export EXACT_KMER_MATCHING=1  
export SENSITIVITY=3.0

echo "Building project..."
mkdir -p build
cd build
make -j$(nproc)
cd ..

echo "Running K-mer validation..."
./scripts/validation/validate_kmer_prefilter.sh