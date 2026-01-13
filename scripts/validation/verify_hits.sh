#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -f "$SCRIPT_DIR/common.sh" ]; then
    source "$SCRIPT_DIR/common.sh"
else
    echo "Error: common.sh not found in $SCRIPT_DIR."
    echo "Please ensure it exists to provide default database paths."
    exit 1
fi

QUERY_FILE="$QUERY_FASTA"
TARGET_FILE="$TARGET_FASTA"
TSV_FILE="$SCRIPT_DIR/results/kmer/kmer_dpu.tsv"

KMER_SIZE=7
MASK="1111111" 
VISUALIZE="false"
CHECK_COUNTS="false"  

PYTHON_SCRIPT="$SCRIPT_DIR/diagonal_checker.py"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --query) QUERY_FILE="$2"; shift ;;
        --target) TARGET_FILE="$2"; shift ;;
        --tsv) TSV_FILE="$2"; shift ;;
        --mask) MASK="$2"; shift ;;
        --viz) VISUALIZE="true" ;;
        --check-counts) CHECK_COUNTS="true" ;; 
        -h|--help) 
            echo "Usage: $0 [--query FILE] [--target FILE] [--tsv FILE] [--mask BINARY_STR] [--viz] [--check-counts]"
            echo "Defaults:"
            echo "  Query:  $QUERY_FASTA"
            echo "  Target: $TARGET_FASTA"
            exit 0 
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Calculate K-mer size from mask
KMER_SIZE=$(echo -n "$MASK" | tr -cd '1' | wc -c)

echo "Configuration:"
echo "  Query:        $QUERY_FILE"
echo "  Target:       $TARGET_FILE"
echo "  TSV:          $TSV_FILE"
echo "  Mask:         $MASK"
echo "  Check Counts: $CHECK_COUNTS"
echo "  Viz:          $VISUALIZE"
echo "----------------------------------------"

for f in "$QUERY_FILE" "$TARGET_FILE" "$TSV_FILE" "$PYTHON_SCRIPT"; do
    if [ ! -r "$f" ]; then
        echo "Error: Cannot read $f."
        echo "Please verify file paths and permissions."
        exit 1
    fi
done

CMD_FLAGS="--query \"$QUERY_FILE\" --target \"$TARGET_FILE\" --tsv \"$TSV_FILE\" --kmer $KMER_SIZE --mask \"$MASK\""

if [ "$VISUALIZE" = "true" ]; then
    CMD_FLAGS="$CMD_FLAGS --visualize"
fi

if [ "$CHECK_COUNTS" = "true" ]; then
    CMD_FLAGS="$CMD_FLAGS --check-counts"
fi

eval python3 "$PYTHON_SCRIPT" $CMD_FLAGS

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Processing failed with error code $EXIT_CODE"
    exit $EXIT_CODE
else
    echo "Processing completed successfully."
fi