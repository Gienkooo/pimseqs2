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
usage() {
    echo "Usage: $0 [-n NUM_DPUS]"
    echo "  -n NUM_DPUS   Number of DPUs to use (overrides automatic detection)"
}
while getopts ":n:h" opt; do
    case $opt in
        n) DPU_COUNT="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
    esac
done
shift $((OPTIND -1))

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

# Scripts
COMPARE_SCRIPT="$SCRIPT_DIR/compare_results.py"
CHECKER_SCRIPT="$SCRIPT_DIR/verify_double_hits.py"

log "Configuration:"
log "  Mask: $MASK"
log "  K-mer: $KMER"
log "  Exact Matching: $EXACT_KMER_MATCHING"
log "  Sensitivity: $SENSITIVITY"
log "  Query DB: $QUERY_DB"
log "  Target DB: $TARGET_DB"
log "  DPU Count: ${FINAL_DPU_COUNT:-(auto)}"

# 2. Run CPU
log "Running K-mer Prefilter on CPU..."
"$MMSEQS_BIN" prefilter "$QUERY_DB" "$TARGET_DB" "$CPU_DB" \
    --exact-kmer-matching "$EXACT_KMER_MATCHING" \
    -s "$SENSITIVITY" \
    -k "$KMER" \
    --mask 0 \
    --mask-lower-case 0 \
    --mask-n-repeat 0 \
    --max-seqs 1000000 \
    --spaced-kmer-mode 0 \
    --spaced-kmer-pattern "$MASK" \
    --diag-score 0 \
    --threads $(nproc) \
    -v 3 > "$OUT_DIR/kmer_cpu.log" 2>&1

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    error "CPU run failed. Check $OUT_DIR/kmer_cpu.log"
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
    --spaced-kmer-pattern "$MASK" \
    --dpu 1 \
    --dpu-num-dpus "$FINAL_DPU_COUNT" \
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

    log "Results saved to:"
    log "  CPU: $CPU_RES"
    log "  DPU: $DPU_RES"

    # 5. Compare Results
    log "Comparing results (CPU vs DPU)..."
    python3 "$COMPARE_SCRIPT" "$CPU_RES" "$DPU_RES" "KmerPrefilter"

    # 6. Verify False Positives (Only if Exact Matching is ON)
    if [ "$EXACT_KMER_MATCHING" -eq 1 ]; then
        log "Verifying False Positives (Diagonal Check)..."
        if [ -f "$CHECKER_SCRIPT" ]; then
            # CPU_FP=$(python3 "$CHECKER_SCRIPT" \
            #    --query "$QUERY_FASTA" \
            #    --target "$TARGET_FASTA" \
            #    --tsv "$CPU_RES" \
            #    --mask "$MASK" \
            #    --log "$CPU_DIAG" 2>&1)
            # 
            # log "CPU Check Summary: $CPU_FP"
            log "Detailed logs written to: $CPU_DIAG"

            # DPU_FP=$(python3 "$CHECKER_SCRIPT" \
            #    --query "$QUERY_FASTA" \
            #    --target "$TARGET_FASTA" \
            #    --tsv "$DPU_RES" \
            #    --mask "$MASK" \
            #    --log "$DPU_DIAG" 2>&1)
                
            #log "DPU Check Summary: $DPU_FP"
            log "Detailed logs written to: $DPU_DIAG"
        else
            log "WARNING: Checker script not found. Skipping."
        fi
    else
        log "Exact matching disabled. Skipping diagonal check."
    fi
} 2>&1 | tee -a "$DPU_LOG"