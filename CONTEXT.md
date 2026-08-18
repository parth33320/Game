# Context: Local Emulator AI & RL Gaming Pipeline

This document defines the Ubiquitous Language and domain terminology for the emulator AI gameplay, RL agent training pipelines, Restream chat integration, and headless streaming.

## Language

**Emulator State Reader**:
Extracted RAM data and gameplay flags read directly from emulator memory without vision models.
_Avoid_: Screen scraper, OCR, vision reader

**RAM State**:
The parsed snapshot of memory containing text box status, active menu options, player coordinates, and opponent battle stats.
_Avoid_: Frame buffer, pixel dump

**Speed Controller**:
The emulator mechanism that toggles fast-forward mode (8x speed) during overworld exploration/grinding and returns to normal speed (1x) during decision prompts.
_Avoid_: Fast forwarder, frame skipper

**Local LLM Decision Engine**:
An Ollama-powered client calling local models (Llama 3 or Qwen 2.5) using comedic persona prompt templates to select discrete button inputs.
_Avoid_: Cloud AI, GPT API

**Restream Chat Aggregator**:
A unified chat ingestion listener connecting to Restream Chat API / webhooks (as well as Twitch and YouTube APIs) to parse chat commands across multiple streaming platforms.
_Avoid_: IRC bot, Twitch reader

**Chat Override Priority**:
The control hierarchy where active viewer chat inputs strictly override and preempt local LLM decisions or automated RL routines.
_Avoid_: Chat fallback, manual mode

**Platformer Environment**:
A Gymnasium-compatible environment wrapping a retro emulator (Gym-Retro or NES-py) for retro platformers (e.g. Castlevania / NES).
_Avoid_: Game simulator, AI box

**PPO Agent Policy**:
A Proximal Policy Optimization network (implemented via PyTorch and Stable-Baselines3) that processes RAM or pixel observations and outputs discrete button actions (Jump, Move Left/Right, Attack/Whip, Crouch).
_Avoid_: Q-learning model, heuristic bot

**Reward Shaping Engine**:
Calculates scalar rewards for horizontal progression, collecting items/score, and surviving, while heavily penalizing life loss, damage, or falling into pits.
_Avoid_: Score counter, loss function

**Headless Streamer**:
A background process running Python and FFmpeg to encode virtual audio/video outputs directly to Restream RTMP ingest servers without OBS or GUI overhead.
_Avoid_: OBS recorder, screen capturer

**Audit Logger**:
An append-only JSONL log recorder tracking decision history, state transitions, and system inputs for accountability.
_Avoid_: File dumper, print logger
