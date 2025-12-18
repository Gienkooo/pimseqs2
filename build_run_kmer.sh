#!/bin/bash
set -e

echo "Building project..."
mkdir -p build
cd build
make -j$(nproc)
cd ..

echo "Running K-mer validation..."
./scripts/validation/validate_kmer_prefilter.sh

echo "Comparing results (CPU vs DPU)..."
python3 scripts/validation/compare_results.py \
	scripts/validation/results/kmer/kmer_cpu.tsv \
	scripts/validation/results/kmer/kmer_dpu.tsv \
	KmerPrefilter
