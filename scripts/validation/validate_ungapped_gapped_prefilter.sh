#!/bin/bash

# Validate Gapped Prefilter (DPU vs GPU)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

prepare_dbs

OUT_DIR="$RESULTS_DIR/ungapped_gapped"
mkdir -p "$OUT_DIR"

DPU_RES="$OUT_DIR/ungapped_gapped_dpu.tsv"
GPU_RES="$OUT_DIR/ungapped_gapped_gpu.tsv"
DPU_DB="$OUT_DIR/ungapped_gapped_dpu_db"
GPU_DB="$OUT_DIR/ungapped_gapped_gpu_db"

# Clean up previous results
rm -f "${DPU_DB}"* "${GPU_DB}"* "${DPU_RES}" "${GPU_RES}"

# E-value threshold (default high for validation to check all scores)
E_VALUE="${E_VALUE:-1000}"
# Max results per query (default high to not truncate results)
MAX_SEQS="${MAX_SEQS:-10000}"
# Number of DPUs (default 0 = auto, override with DPU_NUM_DPUS env var)
DPU_NUM_DPUS="${DPU_NUM_DPUS:-0}"

log "Comparing DPU vs GPU Gapped Prefilter"
log "Using E-value threshold: $E_VALUE, max-seqs: $MAX_SEQS, num-dpus: $DPU_NUM_DPUS"

# -----------------------------------------------------------------------------
# PREPARE GPU DATABASE (Required!)
# -----------------------------------------------------------------------------
TARGET_DB_PAD="${TARGET_DB}_pad"
if [ ! -f "${TARGET_DB_PAD}.dbtype" ]; then
    log "Creating padded database for GPU..."
    "$MMSEQS_BIN" makepaddedseqdb "$TARGET_DB" "$TARGET_DB_PAD" > /dev/null
fi

# -----------------------------------------------------------------------------
# 2. Run GPU (Ungapped + Gapped)
# -----------------------------------------------------------------------------
log "2. Running Ungapped-Gapped Prefilter on GPU..."
"$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB_PAD" "$GPU_DB" \
    --gpu 1 \
    --prefilter-mode 3 \
    -v 3 \
    -e "$E_VALUE" \
    --min-ungapped-score 0 \
    --max-seqs "$MAX_SEQS" \
    --comp-bias-corr 0 \
    2>&1 | tee "$OUT_DIR/ungapped_gapped_gpu.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    cat "$OUT_DIR/ungapped_gapped_gpu.log"
    error "GPU run failed"
fi


# -----------------------------------------------------------------------------
# 1. Run DPU (Ungapped + Gapped)
# -----------------------------------------------------------------------------
log "1. Running Ungapped-Gapped Prefilter on DPU..."
"$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB" "$DPU_DB" \
    --dpu 1 \
    --dpu-num-dpus "$DPU_NUM_DPUS" \
    --prefilter-mode 3 \
    -v 3 \
    -e "$E_VALUE" \
    --min-ungapped-score 0 \
    --max-seqs "$MAX_SEQS" \
    --comp-bias-corr 0 \
    2>&1 | tee "$OUT_DIR/ungapped_gapped_dpu.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "DPU run failed"
fi

# -----------------------------------------------------------------------------
# 3. Convert & Compare
# -----------------------------------------------------------------------------
log "Converting results..."

# Convert DPU results
"$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$DPU_DB" "$DPU_RES" > /dev/null 2>&1

# Convert GPU results
"$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB_PAD" "$GPU_DB" "$GPU_RES" > /dev/null 2>&1

log "Results saved to:"
log "  DPU: $DPU_RES"
log "  GPU: $GPU_RES"

log "Comparing results (DPU vs GPU)..."
# Simple line count check
DPU_COUNT=$(wc -l < "$DPU_RES")
GPU_COUNT=$(wc -l < "$GPU_RES")

echo "-----------------------------------"
echo "Total Hits:"
echo "  DPU: $DPU_COUNT"
echo "  GPU: $GPU_COUNT"
echo "-----------------------------------"