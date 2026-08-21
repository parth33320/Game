#!/bin/bash

TARGET="checkpoints/best_ppo_agent_dist.pt"
TABLE_FILE="checkpoints/best_history_table.md"
LAST_MOD=0

while true; do
    if [ -f "$TARGET" ]; then
        CURRENT_MOD=$(stat -c %Y "$TARGET")
        
        # If the actual physical file size/time changes, it is a genuine record!
        if [ "$CURRENT_MOD" -ne "$LAST_MOD" ]; then
            sleep 1
            TIMESTAMP=$(date +"%Y-%m-%d %I:%M:%S %p")
            
            # Pull the maximum number from the log file to get the live metrics
            DIST=$(grep -oP 'Max Distance: \K[0-9.]+' master_execution.log | sort -g | tail -n 1)
            
            # Failsafe if the text log buffer is running slow
            if [ -z "$DIST" ]; then DIST="Breakthrough"; fi
            
            echo "| $TIMESTAMP | $DIST pixels |" >> "$TABLE_FILE"
            LAST_MOD=$CURRENT_MOD
        fi
    fi
    sleep 10
done
