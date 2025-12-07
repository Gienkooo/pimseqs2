#!/bin/bash

# Validate Gapped Prefilter (CPU vs DPU)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

prepare_dbs

OUT_DIR="$RESULTS_DIR/gapped"
mkdir -p "$OUT_DIR"

CPU_RES="$OUT_DIR/gapped_cpu.tsv"
DPU_RES="$OUT_DIR/gapped_dpu.tsv"
CPU_DB="$OUT_DIR/gapped_cpu_db"
DPU_DB="$OUT_DIR/gapped_dpu_db"

# Clean up previous results
rm -f "${CPU_DB}"* "${DPU_DB}"* "${CPU_RES}" "${DPU_RES}"

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="${E_VALUE:-1000}"
log "Using E-value threshold: $E_VALUE"

# Run CPU
log "Running Gapped (Exhaustive) Search on CPU..."
# Using 'search' with exhaustive mode to mimic gapped prefilter behavior (SW alignment)
"$MMSEQS_BIN" search "$QUERY_DB" "$TARGET_DB" "$CPU_DB" "$TMP_DIR" \
    --exhaustive-search 1 --threads $(nproc) -v 3 -e "$E_VALUE" --comp-bias-corr 0 \
    > "$OUT_DIR/gapped_cpu.log" 2>&1
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    cat "$OUT_DIR/gapped_cpu.log"
    error "CPU run failed"
fi

# Run DPU
log "Running Gapped Prefilter on DPU..."
# Use ungappedprefilter with --prefilter-mode 2 (PREF_MODE_EXHAUSTIVE) to trigger gapped kernel
"$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB" "$DPU_DB" \
    --dpu 1 --prefilter-mode 2 -v 3 -e "$E_VALUE" 2>&1 | tee "$OUT_DIR/gapped_dpu.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "DPU run failed"
fi

# Convert results to TSV for comparison
log "Converting results..."
# CPU: search produces Alignment DB, use convertalis
"$MMSEQS_BIN" convertalis "$QUERY_DB" "$TARGET_DB" "$CPU_DB" "$CPU_RES" \
    --format-output query,target,raw > /dev/null 2>&1

# DPU: prefilter produces Prefilter DB, use createtsv
"$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$DPU_DB" "$DPU_RES" > /dev/null 2>&1

log "Results saved to:"
log "  CPU: $CPU_RES"
log "  DPU: $DPU_RES"
