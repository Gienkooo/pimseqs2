#!/bin/bash
set -e

export E_VALUE="${E_VALUE:-1000}"
export MAX_SEQS="${MAX_SEQS:-10000}"

echo "Building project..."
mkdir -p build
cd build
make -j$(nproc)
cd ..

echo "Running DPU vs GPU Validation..."

export DPU_NUM_DPUS=12
time ./scripts/validation/validate_ungapped_gapped_prefilter.sh

echo "Comparing results (DPU vs GPU)..."

python3 scripts/validation/compare_results.py \
    scripts/validation/results/ungapped_gapped/ungapped_gapped_dpu.tsv \
    scripts/validation/results/ungapped_gapped/ungapped_gapped_gpu.tsv \
    UngappedGapped_DPU_vs_GPU