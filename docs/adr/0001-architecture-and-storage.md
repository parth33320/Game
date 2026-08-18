# 0001 Architecture and Storage Decisions for Emulator AI & RL Pipeline

## Context
We require a local emulator AI gaming pipeline and RL platformer training system. The system needs to support real-time RAM state reading without vision models, speed control toggles (8x/1x), Ollama-powered local LLM decision generation, Restream unified chat command override priority, PyTorch & Stable-Baselines3 PPO training pipelines, and headless RTMP streaming via FFmpeg.

## Decisions

1. **Memory Scraping Over Vision Models**: We extract structured game state directly from RAM memory maps to achieve deterministic, low-latency state representation without requiring heavy vision processing or OCR overhead.
2. **Control Hierarchy & Override Priority**: Restream chat inputs take strict priority over AI/RL policies. When chat commands exist in the queue, viewer actions execute immediately. AI/RL routines resume only when chat is idle.
3. **PPO Algorithm for Platformer Control**: We standardize on Proximal Policy Optimization (PPO) using PyTorch for the custom platformer pipeline and Stable-Baselines3 for the NES Castlevania pipeline due to PPO's stability in discrete action space environments.
4. **FFmpeg Headless RTMP Streaming**: We use direct Python FFmpeg subprocess piping to stream virtual AV feeds to Restream ingest servers (`rtmp://live.restream.io/live`), eliminating desktop GUI or OBS dependencies.
5. **Append-Only JSONL Audit Logging**: System inputs, LLM decisions, RAM state snapshots, and chat overrides are logged to an append-only JSONL file for auditability and post-gameplay evaluation.

## Consequences
- Clean separation of concerns between emulation, AI decision making, chat ingestion, RL training, and media encoding.
- Deterministic testability across all modules with mock data.
- Low-overhead execution optimized for headless environments like GitHub Codespaces.
