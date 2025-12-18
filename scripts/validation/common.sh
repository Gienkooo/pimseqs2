#!/bin/bash

# Common configuration and functions for validation scripts

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
MMSEQS_BIN="$BUILD_DIR/src/mmseqs"

# Default datasets (can be overridden by environment variables)
QUERY_FASTA="${QUERY_FASTA:-$ROOT_DIR/examples/QUERY_small.fasta}"
TARGET_FASTA="${TARGET_FASTA:-$ROOT_DIR/examples/DB_small.fasta}"

# Output directory
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$RESULTS_DIR"

# Temporary directory
TMP_DIR="${TMP_DIR:-$RESULTS_DIR/tmp}"
mkdir -p "$TMP_DIR"

# Database paths
QUERY_DB="$RESULTS_DIR/query_db"
TARGET_DB="$RESULTS_DIR/target_db"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Ensure MMseqs2 is built
check_mmseqs() {
    if [ ! -x "$MMSEQS_BIN" ]; then
        error "MMseqs2 binary not found at $MMSEQS_BIN. Please build it first."
    fi
}

# Prepare databases if they don't exist
prepare_dbs() {
    check_mmseqs
    
    if [ ! -f "${QUERY_DB}.dbtype" ]; then
        log "Creating query database..."
        "$MMSEQS_BIN" createdb "$QUERY_FASTA" "$QUERY_DB" --mask 0 > /dev/null || error "Failed to create query DB"
    fi

    if [ ! -f "${TARGET_DB}.dbtype" ]; then
        log "Creating target database..."
        "$MMSEQS_BIN" createdb "$TARGET_FASTA" "$TARGET_DB" --mask 0 > /dev/null || error "Failed to create target DB"
    fi
}

# Run comparison using Python script
compare_results() {
    local cpu_res="$1"
    local dpu_res="$2"
    local label="$3"
    
    log "Comparing results for $label..."
    python3 "$SCRIPT_DIR/compare_results.py" "$cpu_res" "$dpu_res" "$label"
}
