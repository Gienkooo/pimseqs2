#!/bin/bash
# usage: ./cpu_energy_hook.sh [start|stop]

MODE=$1
START_FILE="/tmp/rapl_start_uj"
ENERGY_OUT="energy_captures.txt"

# Helper to sum up energy from all sockets (package-0, package-1, etc.)
get_total_energy_uj() {
    cat /sys/class/powercap/intel-rapl:*/energy_uj 2>/dev/null | awk '{s+=$1} END {print s}'
}

if [ "$MODE" == "start" ]; then
    # Snapshot current energy counter (Microjoules)
    get_total_energy_uj > "$START_FILE"

elif [ "$MODE" == "stop" ]; then
    # 1. Read End Value
    END_VAL=$(get_total_energy_uj)
    
    # 2. Read Start Value
    if [ -f "$START_FILE" ]; then
        START_VAL=$(cat "$START_FILE")
    else
        START_VAL=0
    fi

    # 3. Calculate Delta and Convert to Joules
    # Microjoules / 1,000,000 = Joules
    # Use awk for handling large integers and float division
    ENERGY_JOULES=$(awk -v end="$END_VAL" -v start="$START_VAL" 'BEGIN {print (end - start) / 1000000}')

    # 4. Save to file
    echo "$ENERGY_JOULES" >> "$ENERGY_OUT"
    
    # Cleanup
    rm -f "$START_FILE"
fi