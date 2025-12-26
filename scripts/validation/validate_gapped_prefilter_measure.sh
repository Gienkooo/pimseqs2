#!/bin/bash

# Validate Gapped Prefilter (CPU vs DPU)
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

prepare_dbs

OUT_DIR="$RESULTS_DIR/gapped"
mkdir -p "$OUT_DIR"

CPU_RES="$OUT_DIR/gapped_cpu.tsv"
DPU_RES="$OUT_DIR/gapped_dpu.tsv"
CPU_DB="$OUT_DIR/gapped_cpu_db"
DPU_DB="$OUT_DIR/gapped_dpu_db"

# Optimization: Inject configuration header to avoid full rebuilds
mkdir -p build
INCLUDE_DIR="$(pwd)/build"
CONFIG_HEADER="$INCLUDE_DIR/measure_config.h"
# Initialize with default value to ensure file exists for CMake
echo "#define MEASURE_R 0" > "$CONFIG_HEADER"

# Cleanup trap
cleanup() {
    rm -f "$CONFIG_HEADER"
}
trap cleanup EXIT

echo "Configuring CMake..."
cd build
cmake .. -DCMAKE_C_FLAGS="-DBENCHMARKING=1 -I$INCLUDE_DIR" -DCMAKE_CXX_FLAGS="-DBENCHMARKING=1 -I$INCLUDE_DIR"
cd ..

for R_VAL in 0 4 8 16 32 48 64 96 128; do
    echo "Building project with R=${R_VAL}..."
    
    # Update config header
    echo "#define MEASURE_R ${R_VAL}" > "$CONFIG_HEADER"
    
    cd build
    make -j"$(nproc)"
    cd ..


# Clean up previous results
    rm -f "${CPU_DB}"* "${DPU_DB}"* "${CPU_RES}" "${DPU_RES}"

    # E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
    E_VALUE="${E_VALUE:-1000}"

    # Run DPU
    log "Running Gapped Prefilter on DPU..."
    # Use ungappedprefilter with --prefilter-mode 2 (PREF_MODE_EXHAUSTIVE) to trigger gapped kernel
    echo "$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB" "$DPU_DB" \
        --dpu 1 --prefilter-mode 2 -v 3 -e "$E_VALUE" --comp-bias-corr 0 2>&1 | tee "$OUT_DIR/gapped_dpu.log"
    "$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB" "$DPU_DB" \
        --dpu 1 --prefilter-mode 2 -v 3 -e "$E_VALUE" --comp-bias-corr 0 2>&1 | tee "$OUT_DIR/gapped_dpu.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        error "DPU run failed"
    fi
done
