#!/bin/bash
# usage: ./dram_energy_hook.sh [start|stop]

MODE=$1
LOG_FILE="/tmp/rapl_dram_log.txt"
PID_FILE="/tmp/rapl_poller.pid"
ENERGY_OUT="energy_captures.txt"

# Helper to get all DRAM energy files
get_energy_files() {
    grep -l "dram" /sys/class/powercap/intel-rapl:*/intel-rapl:*:*/name 2>/dev/null | sed 's/name$/energy_uj/' | sort
}

# Helper to get the corresponding MAX ranges
get_max_files() {
    grep -l "dram" /sys/class/powercap/intel-rapl:*/intel-rapl:*:*/name 2>/dev/null | sed 's/name$/max_energy_range_uj/' | sort
}

if [ "$MODE" == "start" ]; then
    rm -f "$LOG_FILE"

    # Fix: Store file lists in arrays to handle spaces/newlines correctly
    mapfile -t MAX_FILES < <(get_max_files)
    mapfile -t ENERGY_FILES < <(get_energy_files)

    # Check if we actually found any DRAM domains
    if [ ${#MAX_FILES[@]} -eq 0 ]; then
        echo "Error: No DRAM RAPL domains found." >&2
        exit 1
    fi

    # 1. Initialize Log File with Max Ranges (Header Line)
    # Using "${MAX_FILES[@]}" expands to separate arguments
    paste -d " " "${MAX_FILES[@]}" > "$LOG_FILE"

    # 2. Capture immediate start value
    paste -d " " "${ENERGY_FILES[@]}" >> "$LOG_FILE"

    # 3. Start Background Poller
    (
        while true; do
            sleep 60
            paste -d " " "${ENERGY_FILES[@]}" >> "$LOG_FILE"
        done
    ) &
    
    # 4. Save PID
    echo $! > "$PID_FILE"

elif [ "$MODE" == "stop" ]; then
    # 1. Kill the Poller
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        # Check if process is running before killing
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null
        fi
        rm "$PID_FILE"
    fi

    # 2. Capture the FINAL value immediately
    mapfile -t ENERGY_FILES < <(get_energy_files)
    if [ ${#ENERGY_FILES[@]} -gt 0 ]; then
        paste -d " " "${ENERGY_FILES[@]}" >> "$LOG_FILE"
    fi

    # 3. Process the Log File with AWK
    awk '
    # Line 1: MAX Ranges
    NR == 1 {
        for (i=1; i<=NF; i++) max[i] = $i;
        next;
    }

    # Line 2: Start Value
    NR == 2 {
        for (i=1; i<=NF; i++) prev[i] = $i;
        next;
    }

    # Line 3+: Subsequent samples
    {
        for (i=1; i<=NF; i++) {
            curr = $i;
            delta = curr - prev[i];
            
            # Handle Wraparound
            if (delta < 0) {
                delta = delta + max[i];
            }
            
            total_uj += delta;
            prev[i] = curr;
        }
    }

    # Output Total Joules
    END {
        if (total_uj == "") total_uj = 0;
        printf "%.6f\n", total_uj / 1000000;
    }
    ' "$LOG_FILE" >> "$ENERGY_OUT"

    # Clean up
    rm -f "$LOG_FILE"
fi