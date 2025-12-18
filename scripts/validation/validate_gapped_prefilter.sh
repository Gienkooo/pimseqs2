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
# Max results per query (default high to not truncate results during validation)
MAX_SEQS="${MAX_SEQS:-10000}"
log "Using E-value threshold: $E_VALUE, max-seqs: $MAX_SEQS"

# To isolate and test ONLY the gapped alignment stage, we follow this workflow:
# 1. Generate a prefilter database of candidate pairs on the CPU.
# 2. Run gapped alignment on CPU using those candidates as input (ground truth).
# 3. Run gapped prefilter on DPU using the exact same candidates as input.
# This provides a true apples-to-apples comparison of the SW implementations.

CANDIDATE_DB="$OUT_DIR/candidate_db"
rm -f "${CANDIDATE_DB}"*

log "1. Generating candidate pairs with CPU prefilter..."
"$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB" "$CANDIDATE_DB" \
    --threads $(nproc) --prefilter-mode 2 --dpu 0 -v 3 \
    > "$OUT_DIR/candidate_gen.log" 2>&1
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    cat "$OUT_DIR/candidate_gen.log"
    error "Candidate generation failed"
fi

# Run CPU
log "2. Running Gapped Alignment on CPU with candidate pairs..."
"$MMSEQS_BIN" align "$QUERY_DB" "$TARGET_DB" "$CANDIDATE_DB" "$CPU_DB" \
    --threads $(nproc) -v 3 -e "$E_VALUE" --comp-bias-corr 0 \
    > "$OUT_DIR/gapped_cpu.log" 2>&1
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    cat "$OUT_DIR/gapped_cpu.log"
    error "CPU run failed"
fi

# Run DPU
log "3. Running Gapped Prefilter on DPU with candidate pairs..."
# We use the CANDIDATE_DB as the third argument to the DPU runner.
# The host pipeline will read this and feed only those pairs to the DPU.
"$MMSEQS_BIN" ungappedprefilter "$QUERY_DB" "$TARGET_DB" "$DPU_DB" \
    --dpu 1 --prefilter-mode 2 -v 3 -e "$E_VALUE" --max-seqs "$MAX_SEQS" --comp-bias-corr 0 2>&1 | tee "$OUT_DIR/gapped_dpu.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "DPU run failed"
fi

# Convert results to TSV for comparison
log "Converting results..."
# CPU: align produces Alignment DB, use createtsv
"$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$CPU_DB" "$CPU_RES" > /dev/null 2>&1

# DPU: prefilter produces Prefilter DB, use createtsv
"$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$DPU_DB" "$DPU_RES" > /dev/null 2>&1

log "Results saved to:"
log "  CPU: $CPU_RES"
log "  DPU: $DPU_RES"
