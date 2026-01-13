#!/bin/bash

# usage: ./gpu_energy_hook.sh [start|stop]

MODE=$1
RESULTS_DIR="${RESULTS_DIR:-.}"
PID_FILE="$RESULTS_DIR/gpu_monitor.pid"
RAW_POWER_LOG="$RESULTS_DIR/gpu_power.log"
ENERGY_OUT="$RESULTS_DIR/energy_captures.txt"

if [ "$MODE" == "start" ]; then
    nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader -lms 10 > "$RAW_POWER_LOG" 2>/dev/null &
    echo $! > "$PID_FILE"
    
    # Brief pause to ensure logger catches the start
    sleep 0.1

elif [ "$MODE" == "stop" ]; then
    if [ -f "$PID_FILE" ]; then
        kill "$(cat \"$PID_FILE\")" 2>/dev/null
        rm "$PID_FILE"
    fi

    # Calculate Energy (Joules) = Sum(Watts) * 0.01s
    # Handle empty files or arithmetic errors gracefully
    TOTAL_JOULES=$(awk '{s+=$1} END {if (NR > 0) print s * 0.01; else print 0}' "$RAW_POWER_LOG")
    
    echo "$TOTAL_JOULES" >> "$ENERGY_OUT"
    rm -f "$RAW_POWER_LOG"
fi