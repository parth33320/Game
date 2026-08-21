#!/bin/bash
# run_codespace_training.sh
# GitHub Codespaces Training Orchestrator Script
# Handles background execution and prevents terminal idle timeouts during RL training & walkthrough generation.

set -e

echo "=========================================================="
echo "Starting Castlevania AI Codespaces Training Orchestrator"
echo "Timestamp: $(date)"
echo "=========================================================="

# 1. Start Watchdog Process Supervisor in Background to prevent idle timeout and handle auto-recovery
echo "[1/3] Launching Watchdog process supervisor in background..."
python3 scripts/watchdog.py > watchdog.log 2>&1 &
WATCHDOG_PID=$!
echo "Watchdog PID: $WATCHDOG_PID"

# 2. Monitor Watchdog until training pipeline completes
echo "[2/3] Training pipeline running in background (logging to watchdog.log)..."
wait $WATCHDOG_PID || true

echo "Training pipeline completed or watchdog exited."

# 3. Generate final 720p HD Castlevania walkthrough verification video
echo "[3/3] Generating final 720p HD 'castlevania_walkthrough.mp4' verification video..."
python3 scripts/generate_walkthrough_video.py --output castlevania_walkthrough.mp4 --steps 300

echo "=========================================================="
echo "Codespaces Training & Video Generation Finished Successfully!"
echo "Generated output: castlevania_walkthrough.mp4"
echo "Timestamp: $(date)"
echo "=========================================================="
