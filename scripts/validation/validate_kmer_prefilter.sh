#!/bin/bash

# Validate K-mer Prefilter (CPU vs DPU)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

prepare_dbs

OUT_DIR="$RESULTS_DIR/kmer"
mkdir -p "$OUT_DIR"

MASK="${MASK:-1111111}"
KMER="${KMER:-7}"

# If CPU_TSV is set (from parent), use it. Otherwise default to OUT_DIR location.
CPU_RES="${CPU_TSV:-$OUT_DIR/kmer_cpu.tsv}"
DPU_RES="${DPU_TSV:-$OUT_DIR/kmer_dpu.tsv}"
CPU_DB="$OUT_DIR/kmer_cpu_db"
DPU_DB="$OUT_DIR/kmer_dpu_db"

# --diag-score 0 
# -k 5 
# --spaced-kmer-pattern "10111111" 
# --exact-kmer-matching 1 
# --mask 0

log "Configuration:"
log "  Mask: $MASK"
log "  K-mer: $KMER"
log "  Query DB: $QUERY_DB"
log "  Target DB: $TARGET_DB"

# 1. Run CPU
log "Running K-mer Prefilter on CPU..."
"$MMSEQS_BIN" prefilter "$QUERY_DB" "$TARGET_DB" "$CPU_DB" --exact-kmer-matching 1 -k $KMER --spaced-kmer-mode 0 --spaced-kmer-pattern "$MASK" --diag-score 0 --threads $(nproc) -v 3 > "$OUT_DIR/kmer_cpu.log" 2>&1
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "CPU run failed. Check $OUT_DIR/kmer_cpu.log"
fi

log "Running Gapped align on CPU..."
"$MMSEQS_BIN" align "$QUERY_DB" "$TARGET_DB" "$CPU_DB" "$CPU_DB.aln"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "CPU run failed"
fi

# 2. Run DPU
log "Running K-mer Prefilter on DPU..."
"$MMSEQS_BIN" prefilter "$QUERY_DB" "$TARGET_DB" "$DPU_DB" --exact-kmer-matching 1 -k $KMER --spaced-kmer-mode 0 --spaced-kmer-pattern "$MASK" --dpu 1 -v 3 > "$OUT_DIR/kmer_dpu.log" 2>&1
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "DPU run failed. Check $OUT_DIR/kmer_dpu.log"
fi

# 3. Convert results to TSV
log "Converting results to TSV..."
"$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$CPU_DB" "$CPU_RES" > /dev/null 2>&1
"$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$DPU_DB" "$DPU_RES" > /dev/null 2>&1

log "Results saved to:"
log "  CPU: $CPU_RES"
log "  DPU: $DPU_RES"
