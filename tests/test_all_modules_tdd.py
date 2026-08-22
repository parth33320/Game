import pytest
import os
import json
import tempfile
import numpy as np
import torch

from audit.audit_logger import AuditLogger
from ai.llm_player import LLMPlayer
from agent.rewards import PlatformerRewardCalculator
from env.rewards import RetroRewardEngine
from emulator.ram_scraper import RAMScraper
from emulator.castlevania_autonomous_engine import CastlevaniaAutonomousEngine
from scripts.analyze_walkthroughs import analyze
from scripts.stream_gameplay import FFmpegRestreamStreamer
from agent.imitation import parse_walkthrough

class DummyEnv:
    def __init__(self):
        self.retro_env = None
        self.unwrapped = self
    def get_ram(self):
        ram = [0] * 0x0200
        ram[0x0018] = 0x05  # Active gameplay state
        ram[0x0045] = 16    # Player Health
        return ram
    def step(self, action):
        return np.zeros(18, dtype=np.float32), 0.0, False, False, {}
    def reset(self):
        return np.zeros(18, dtype=np.float32), {}
    def close(self):
        pass

def test_audit_logger_edge_cases_and_corruption_handling(tmp_path):
    log_filepath = str(tmp_path / "corrupt_audit.jsonl")

    # Write some valid lines and a corrupted line
    with open(log_filepath, "w") as f:
        f.write(json.dumps({"timestamp": 100, "event_type": "init", "details": {"data": "test1"}}) + "\n")
        f.write("CORRUPTED_NON_JSON_LINE_HELL_12345\n")
        f.write(json.dumps({"timestamp": 102, "event_type": "step", "details": {"data": "test2"}}) + "\n")

    logger = AuditLogger(log_filepath=log_filepath)
    # Log a new entry
    logger.log_event("finish", {"data": "test3"})

    entries = logger.read_all_events()
    assert len(entries) == 3
    assert entries[0]["event_type"] == "init"
    assert entries[1]["event_type"] == "step"
    assert entries[2]["event_type"] == "finish"

def test_llm_player_fallback_and_persona_formatting():
    player = LLMPlayer(model_name="mock_model", persona="comedic hero")

    ram_state = {"hp": 16, "boss_hp": 16, "score": 1000, "stage": 1, "x_pos": 250, "y_pos": 100}
    res = player.select_action(ram_state)
    action = res["action"]
    dialogue = res["dialogue"]

    assert action in ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
    assert len(dialogue) > 0

def test_platformer_reward_calculator():
    calculator = PlatformerRewardCalculator()
    calculator.reset({"x_pos": 100, "score": 0, "lives": 3, "health": 16})

    # Move forward
    reward = calculator.calculate_reward({"x_pos": 120, "score": 0, "lives": 3, "health": 16})
    assert reward > 0.0

def test_retro_reward_engine():
    engine = RetroRewardEngine()
    engine.reset({"x_pos": 100, "score": 0, "lives": 3, "health": 16})

    reward = engine.calculate_reward({"x_pos": 120, "score": 0, "lives": 3, "health": 16})
    assert reward > 0.0

def test_castlevania_autonomous_engine_state_transitions(tmp_path):
    engine = CastlevaniaAutonomousEngine(
        env_creation_func=lambda: DummyEnv(),
        save_dir=str(tmp_path / "ckpt")
    )
    assert engine.current_state == "PLAYING"

    # Run a step
    obs, reward, term, trunc, info = engine.process_autonomous_step(
        rl_agent=None,
        rl_agent_policy_func=lambda obs_tensor: 1
    )
    assert engine.current_state == "PLAYING"
    engine.env.close()

def test_analyze_walkthroughs_invalid_file():
    res = analyze("non_existent_walkthrough_file.bk2")
    assert res["path"] == "non_existent_walkthrough_file.bk2"
    assert "error" in res

def test_ffmpeg_restream_streamer_command_generation():
    streamer = FFmpegRestreamStreamer(stream_key="live_test", rtmp_url="rtmp://localhost/live")
    cmd = streamer.build_ffmpeg_command(output_target="rtmp://localhost/live/live_test")
    assert "ffmpeg" in cmd[0]
    assert "rtmp://localhost/live/live_test" in cmd[-1]

def test_imitation_parse_walkthrough_invalid_format():
    with pytest.raises(ValueError, match="Unsupported walkthrough format"):
        parse_walkthrough("invalid_file.txt")
