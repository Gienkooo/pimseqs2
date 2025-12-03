#!/bin/bash
set -e

echo "Building project..."
cd build
make -j$(nproc)
cd ..

echo "Running Ungapped validation..."
./scripts/validation/validate_ungapped_prefilter.sh

echo "Comparing results..."
python3 scripts/validation/compare_results.py scripts/validation/results/ungapped/ungapped_cpu.tsv scripts/validation/results/ungapped/ungapped_dpu.tsv UngappedPrefilter
