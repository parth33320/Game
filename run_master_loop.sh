#!/bin/bash

echo "👾 [Master Loop] Registering Castlevania ROM..."
python3 -m retro.import .

RUN_COUNT=1

while true; do
    echo "----------------------------------------------------------------------"
    echo "🚀 Launching Autonomous Training Loop - Run #$RUN_COUNT"
    echo "----------------------------------------------------------------------"
    
    python3 scripts/train_agent.py
    TRAIN_EXIT_CODE=$?
    
    # Code 2 = Stagnation / Reward Hacking detected -> Reset weights and auto-retrain
    if [ $TRAIN_EXIT_CODE -eq 2 ]; then
        echo "⚠️ [ALERT] Stagnation detected. Adapting hyperparameters and selecting the best progress checkpoint..."
        python3 scripts/auto_tuner.py
        RUN_COUNT=$((RUN_COUNT+1))
        sleep 5
        continue
    fi
    
    # Code 0 = True Game Completion! Beat Dracula, make video, and STOP FOREVER.
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "🎉 [SUCCESS] AI Player completed Castlevania NES autonomously!"
        echo "🎬 Generating high-speed <10 minute proof video asset..."
        
        python3 scripts/generate_walkthrough_video.py
        
        echo "✅ Proof Video 'castlevania_walkthrough_run_${RUN_COUNT}.mp4' successfully saved."
        echo "🏁 [FINISH] Goal reached. Shutting down master orchestration engine permanently to save your balance."
        break
    else
        echo "❌ AI did not complete the game yet (Exit Code: $TRAIN_EXIT_CODE). Continuing training..."
        RUN_COUNT=$((RUN_COUNT+1))
        sleep 5
    fi
done
