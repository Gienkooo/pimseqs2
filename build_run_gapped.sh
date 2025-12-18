#!/bin/bash
set -e

# Export environment variables so they're available to child scripts
export E_VALUE="${E_VALUE:-1000}"
export MAX_SEQS="${MAX_SEQS:-10000}"

echo "Building project..."
mkdir -p build
cd build
make -j$(nproc)
cd ..

echo "Running Gapped validation..."
time ./scripts/validation/validate_gapped_prefilter.sh

echo "Comparing results (CPU vs DPU)..."
python3 scripts/validation/compare_results.py \
	scripts/validation/results/gapped/gapped_cpu.tsv \
	scripts/validation/results/gapped/gapped_dpu.tsv \
	GappedPrefilter
