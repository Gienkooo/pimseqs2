#!/bin/bash

# Validate K-mer Prefilter (CPU vs DPU)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

prepare_dbs

OUT_DIR="$RESULTS_DIR/kmer"
mkdir -p "$OUT_DIR"

CPU_RES="$OUT_DIR/kmer_cpu.tsv"
DPU_RES="$OUT_DIR/kmer_dpu.tsv"
CPU_DB="$OUT_DIR/kmer_cpu_db"
DPU_DB="$OUT_DIR/kmer_dpu_db"

# Max results per query (default high to not truncate results during validation)
MAX_SEQS="${MAX_SEQS:-10000}"
# Minimum diagonal score threshold (k-mer hit count when --diag-score 0)
MIN_DIAG_SCORE="${MIN_DIAG_SCORE:-2}"
log "Using max-seqs: $MAX_SEQS, min-diag-score: $MIN_DIAG_SCORE"

# Run CPU with --diag-score 0 to disable ungapped alignment scoring
# This makes CPU use raw k-mer hit counts like DPU does
log "Running K-mer Prefilter on CPU..."
"$MMSEQS_BIN" prefilter "$QUERY_DB" "$TARGET_DB" "$CPU_DB" --threads $(nproc) -v 3 --max-seqs "$MAX_SEQS" --diag-score 0 --min-ungapped-score "$MIN_DIAG_SCORE" 2>&1 | tee "$OUT_DIR/kmer_cpu.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "CPU run failed"
fi

# Run DPU with same threshold
log "Running K-mer Prefilter on DPU..."
"$MMSEQS_BIN" prefilter "$QUERY_DB" "$TARGET_DB" "$DPU_DB" --dpu 1 -v 3 --max-seqs "$MAX_SEQS" --min-ungapped-score "$MIN_DIAG_SCORE" 2>&1 | tee "$OUT_DIR/kmer_dpu.log"
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
