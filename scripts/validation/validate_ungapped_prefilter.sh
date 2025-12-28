#!/bin/bash

# Validate Ungapped Prefilter (CPU vs DPU)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

prepare_dbs

OUT_DIR="$RESULTS_DIR/ungapped"
mkdir -p "$OUT_DIR"

CPU_RES="$OUT_DIR/ungapped_cpu.tsv"
DPU_RES="$OUT_DIR/ungapped_dpu.tsv"
CPU_DB="$OUT_DIR/ungapped_cpu_db"
DPU_DB="$OUT_DIR/ungapped_dpu_db"

# E-value threshold (default high for validation to check all scores, override with E_VALUE env var)
E_VALUE="${E_VALUE:-1000}"
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="${MAX_SEQS:-10000}"
# Minimum ungapped score threshold (default 15, override with MIN_UNGAPPED env var)
MIN_UNGAPPED="${MIN_UNGAPPED:-15}"
log "Using E-value threshold: $E_VALUE, max-seqs: $MAX_SEQS, min-ungapped-score: $MIN_UNGAPPED"

# Run CPU
log "Running Ungapped Prefilter on CPU..."
"$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB" "$CPU_DB" --prefilter-mode 1 --comp-bias-corr 0 --threads $(nproc) -v 3 -e "$E_VALUE" --max-seqs "$MAX_SEQS" --min-ungapped-score "$MIN_UNGAPPED" 2>&1 | tee "$OUT_DIR/ungapped_cpu.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "CPU run failed"
fi

# Run DPU
log "Running Ungapped Prefilter on DPU..."
dpu-profiling dpu-sections -- "$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB" "$DPU_DB" --prefilter-mode 1 --comp-bias-corr 0 --dpu 1 -v 3 -e "$E_VALUE" --max-seqs "$MAX_SEQS" --min-ungapped-score "$MIN_UNGAPPED" 2>&1 | tee "$OUT_DIR/ungapped_dpu.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "DPU run failed"
fi

# Convert results to TSV for comparison
log "Converting results..."
"$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$CPU_DB" "$CPU_RES" > /dev/null 2>&1
"$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$DPU_DB" "$DPU_RES" > /dev/null 2>&1

log "Results saved to:"
log "  CPU: $CPU_RES"
log "  DPU: $DPU_RES"
