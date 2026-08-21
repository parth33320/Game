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
            METRICS=$(python3 - <<'PY'
import torch
checkpoint = torch.load("checkpoints/best_ppo_agent_dist.pt", map_location="cpu", weights_only=False)
print("| {distance:.1f} pixels | Stage {stage} | Screen {screen} + X {fine} | Boss HP {boss} | HP {health} | Lives {lives} |".format(
    distance=float(checkpoint.get("max_x_pos", 0.0)),
    stage=int(checkpoint.get("stage", 0)),
    screen=int(checkpoint.get("coarse_screen", 0)),
    fine=int(checkpoint.get("fine_x", 0)),
    boss=int(checkpoint.get("boss_hp", 16)),
    health=int(checkpoint.get("health", 16)),
    lives=int(checkpoint.get("lives", 3)),
))
PY
)
            echo "| $TIMESTAMP $METRICS" >> "$TABLE_FILE"
            read -r STAGE DISTANCE_PIXELS < <(python3 - <<'PY'
import torch
checkpoint = torch.load("checkpoints/best_ppo_agent_dist.pt", map_location="cpu", weights_only=False)
print(int(checkpoint.get("stage", 0)), int(float(checkpoint.get("max_x_pos", 0.0))))
PY
)
            MAP=$(printf '%*s' "$((DISTANCE_PIXELS / 16))" '' | tr ' ' '#')
            echo "" >> "$TABLE_FILE"
            echo "Global map: Stage $STAGE [$MAP Simon position]" >> "$TABLE_FILE"
            LAST_MOD=$CURRENT_MOD
        fi
    fi
    sleep 10
done
