#!/bin/bash

# Validate K-mer Prefilter (CPU vs DPU)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

prepare_dbs

OUT_DIR="$RESULTS_DIR/kmer"
mkdir -p "$OUT_DIR"

# 1. Configuration & Defaults
MASK="${MASK:-1111111}"
KMER="${KMER:-7}"
EXACT_KMER_MATCHING="${EXACT_KMER_MATCHING:-1}"
SENSITIVITY="${SENSITIVITY:-3.0}"

# overriding number of DPUs
DPU_COUNT=""
NO_DIAG_CHECK=0

usage() {
    echo "Usage: $0 [-n NUM_DPUS] [--no-diag-check]"
    echo "  -n NUM_DPUS       Number of DPUs to use"
    echo "  --no-diag-check   Disable expensive geometric diagonal verification"
}

# Parse args manually to handle long options
while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            DPU_COUNT="$2"
            shift 2
            ;;
        --no-diag-check)
            NO_DIAG_CHECK=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

# Validate and select final DPU count (default uses get_dpu_count)
if [ -n "$DPU_COUNT" ]; then
    if ! [[ "$DPU_COUNT" =~ ^[0-9]+$ ]] || [ "$DPU_COUNT" -le 0 ]; then
        error "Invalid DPU count: $DPU_COUNT"
    fi
fi
FINAL_DPU_COUNT="${DPU_COUNT:-$(get_dpu_count)}"

# If CPU_TSV is set (from parent), use it. Otherwise default to OUT_DIR location.
CPU_RES="${CPU_TSV:-$OUT_DIR/kmer_cpu.tsv}"
DPU_RES="${DPU_TSV:-$OUT_DIR/kmer_dpu.tsv}"
CPU_DB="$OUT_DIR/kmer_cpu_db"
DPU_DB="$OUT_DIR/kmer_dpu_db"
DPU_LOG="$OUT_DIR/kmer_dpu.log"
CPU_DIAG="$OUT_DIR/kmer_cpu_diag.log"
DPU_DIAG="$OUT_DIR/kmer_dpu_diag.log"

log "Configuration:"
log "  Mask: $MASK"
log "  K-mer: $KMER"
log "  Exact Matching: $EXACT_KMER_MATCHING"
log "  Sensitivity: $SENSITIVITY"
log "  Query DB: $QUERY_DB"
log "  Target DB: $TARGET_DB"
log "  DPU Count: ${FINAL_DPU_COUNT:-(auto)}"
log "  Diagonal Check: $( [ $NO_DIAG_CHECK -eq 1 ] && echo "DISABLED" || echo "ENABLED" )"

# 2. Run CPU
log "Running K-mer Prefilter on CPU..."
"$MMSEQS_BIN" prefilter "$QUERY_DB" "$TARGET_DB" "$CPU_DB" \
    --exact-kmer-matching "$EXACT_KMER_MATCHING" \
    -s "$SENSITIVITY" \
    -k "$KMER" \
    --mask 0 \
    --mask-lower-case 0 \
    --mask-n-repeat 0 \
    --max-seqs "${MAX_SEQS:-10000}" \
    --split 1 \
    --spaced-kmer-mode 0 \
    --min-ungapped-score 0 \
    --spaced-kmer-pattern "$MASK" \
    --diag-score 0 \
    --threads 1 \
    -v 3 > "$OUT_DIR/kmer_cpu.log" 2>&1

if [ $? -ne 0 ]; then
    log "WARNING: CPU run failed (likely OOM). DPU-only validation. Check $OUT_DIR/kmer_cpu.log"
fi

# 3. Run DPU
log "Running K-mer Prefilter on DPU..."
DPU_PROFILE=1 "$MMSEQS_BIN" prefilter "$QUERY_DB" "$TARGET_DB" "$DPU_DB" \
    --exact-kmer-matching "$EXACT_KMER_MATCHING" \
    -s "$SENSITIVITY" \
    -k "$KMER" \
    --mask 0 \
    --mask-lower-case 0 \
    --mask-n-repeat 0 \
    --spaced-kmer-mode 0 \
    --min-ungapped-score 0  \
    --spaced-kmer-pattern "$MASK" \
    --split 0 \
    --dpu 1 \
    --dpu-num-dpus "$FINAL_DPU_COUNT" \
    --threads 1 \
    -v 3 > "$DPU_LOG" 2>&1

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "DPU run failed. Check $DPU_LOG"
fi

{
    # 4. Convert results to TSV
    log "Converting results to TSV..."
    "$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$CPU_DB" "$CPU_RES" > /dev/null 2>&1
    "$MMSEQS_BIN" createtsv "$QUERY_DB" "$TARGET_DB" "$DPU_DB" "$DPU_RES" > /dev/null 2>&1

    log "Log saved to:"
    log "  CPU: $OUT_DIR/kmer_cpu.log"
    log "  DPU: $DPU_LOG"

    # 5. Run Verification Tool
    VERIFY_TOOL="$SCRIPT_DIR/verify_kmer_impl.py"
    log "Running Comprehensive Verification..."

    CMD="python3 \"$VERIFY_TOOL\" \"$CPU_RES\" \"$DPU_RES\" --kmer \"$KMER\" --mask \"$MASK\""

    # Add FASTA args if diagonal check is enabled
    if [ $NO_DIAG_CHECK -eq 0 ]; then
        CMD="$CMD --query \"$QUERY_FASTA\" --target \"$TARGET_FASTA\""
    else
        CMD="$CMD --no-diag-check"
    fi

    log "Executing: $CMD"
    eval $CMD

    log "Verification Complete."
} 2>&1 | tee -a "$DPU_LOG"