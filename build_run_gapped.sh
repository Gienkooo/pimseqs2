#!/bin/bash
set -e

echo "Building project..."
cd build
make -j$(nproc)
cd ..

echo "Running Gapped validation..."
./scripts/validation/validate_gapped_prefilter.sh

echo "Comparing results..."
python3 scripts/validation/compare_results.py scripts/validation/results/gapped/gapped_cpu.tsv scripts/validation/results/gapped/gapped_dpu.tsv GappedPrefilter
