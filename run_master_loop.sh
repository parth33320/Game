#!/bin/bash

# Ensure ROM is registered before doing anything else
echo "👾 [Master Loop] Registering Castlevania ROM..."
python3 -m retro.import .

# ---------------------------------------------------------------------
# PHASE 1: Run Imitation Pre-training (Runs once to seed the network)
# ---------------------------------------------------------------------
if [ ! -f "checkpoints/imitation_baseline.pt" ]; then
    echo "🧠 [Phase 1] Starting Behavioral Cloning from CastlevaniaTAS.bk2..."
    python3 agent/pretrain_imitation.py
else
    echo "✅ [Phase 1] Found existing imitation baseline weights. Skipping pre-train."
fi

# ---------------------------------------------------------------------
# PHASE 2: Infinite Training, Anti-Stagnation, and Verification Loop
# ---------------------------------------------------------------------
RUN_COUNT=1

while true; do
    echo "----------------------------------------------------------------------"
    echo "🚀 Launching Autonomous Training Loop - Run #$RUN_COUNT"
    echo "----------------------------------------------------------------------"
    
    # Start the training agent on CPU in the background
    # It reads config/active_training_params.json and runs until completion or stagnation
    python3 scripts/train_agent.py
    TRAIN_EXIT_CODE=$?
    
    # 0 = Game Completed Fairly (Target Condition)
    # 2 = Reward Hacking / Stagnation Detected (Trigger Auto-Retrain)
    if [ $TRAIN_EXIT_CODE -eq 2 ]; then
        echo "⚠️ [ALERT] Stagnation or Reward Hacking detected by the engine wrapper."
        echo "♻️ Rolling back weights to the last safe checkpoint and restarting run..."
        # Copy our safe base weights over the corrupted/hacked model weights
        cp checkpoints/imitation_baseline.pt checkpoints/model_weights_latest.pt
        RUN_COUNT=$((RUN_COUNT+1))
        sleep 5
        continue
    fi
    
    # If the script exits cleanly (0), it means the AI successfully completed the game!
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "🎉 [SUCCESS] AI Player completed Castlevania NES autonomously!"
        echo "🎬 Generating high-speed <10 minute proof video asset..."
        
        # Runs your frame-skipping wrapper script to render the 4x fast-forward demo
        python3 scripts/generate_walkthrough_video.py
        
        echo "✅ Proof Video 'castlevania_walkthrough_run_${RUN_COUNT}.mp4' successfully saved."
        echo "🔄 Automatically restarting the game loop to prove infinite playback loop..."
        
        RUN_COUNT=$((RUN_COUNT+1))
        sleep 10
    else
        echo "❌ Unexpected error code ($TRAIN_EXIT_CODE). Restarting environment core..."
        sleep 10
    fi
done