#!/bin/bash
# usage: ./dram_energy_hook.sh [start|stop]

MODE=$1
START_FILE="/tmp/rapl_dram_start_uj"
ENERGY_OUT="energy_captures.txt"

# Function to sum energy from all DRAM domains on the system
# It looks for RAPL domains named "dram" (usually under socket/package)
get_dram_energy_uj() {
    # Find all 'name' files containing "dram" inside powercap
    # Then read the corresponding 'energy_uj' file in the same directory
    grep -l "dram" /sys/class/powercap/intel-rapl:*/intel-rapl:*:*/name 2>/dev/null | \
    sed 's/name$/energy_uj/' | \
    xargs cat 2>/dev/null | \
    awk '{s+=$1} END {print s}'
}

if [ "$MODE" == "start" ]; then
    # Snapshot current DRAM energy
    get_dram_energy_uj > "$START_FILE"

elif [ "$MODE" == "stop" ]; then
    # 1. Read End Value
    END_VAL=$(get_dram_energy_uj)
    
    # 2. Read Start Value
    if [ -f "$START_FILE" ]; then
        START_VAL=$(cat "$START_FILE")
    else
        START_VAL=0
    fi

    # 3. Calculate Delta (Microjoules -> Joules)
    # If END < START (counter overflow), RAPL usually handles it, 
    # but for simplicity we assume monotonic increase here.
    ENERGY_JOULES=$(awk -v end="$END_VAL" -v start="$START_VAL" 'BEGIN {print (end - start) / 1000000}')

    # 4. Handle case where DRAM domain might not exist (empty string)
    if [ -z "$ENERGY_JOULES" ]; then ENERGY_JOULES=0; fi

    echo "$ENERGY_JOULES" >> "$ENERGY_OUT"
    rm -f "$START_FILE"
fi
