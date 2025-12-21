#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "=== Running K-mer Prefilter Validation ==="
"$SCRIPT_DIR/validate_kmer_prefilter.sh"
echo ""

echo "=== Running Ungapped Prefilter Validation ==="
"$SCRIPT_DIR/validate_ungapped_prefilter.sh"
echo ""

echo "=== Running Gapped Prefilter Validation ==="
"$SCRIPT_DIR/validate_gapped_prefilter.sh"
echo ""
