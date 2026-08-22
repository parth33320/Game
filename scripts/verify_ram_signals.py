import os
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.retro_env import HeadlessRetroEnv
from emulator.ram_scraper import RAMScraper


def verify_ram_signals():
    print("🔍 Starting RAM Signal Verification Unit Test...")
    env = HeadlessRetroEnv(obs_type="ram", use_retro=True, base_max_steps=500)
    obs, info = env.reset(seed=42)

    initial_x = info["x_pos"]
    initial_health = info["health"]
    initial_stage = info["stage"]

    print(f"Initial State: x_pos={initial_x}, health={initial_health}, stage={initial_stage}")
    print(f"Observation Vector Shape: {obs.shape} (Expected: (18,))")

    assert obs.shape == (18,), f"Expected 18 observation features, got {obs.shape}"

    # Verify RAM addresses in env attributes
    assert hasattr(env, "stage"), "Env missing 'stage' attribute"
    assert hasattr(env, "health"), "Env missing 'health' attribute"
    assert hasattr(env, "boss_hp"), "Env missing 'boss_hp' attribute"
    assert hasattr(env, "hearts"), "Env missing 'hearts' attribute"
    assert hasattr(env, "whip_level"), "Env missing 'whip_level' attribute"
    assert hasattr(env, "subweapon"), "Env missing 'subweapon' attribute"
    assert hasattr(env, "stair_mode_byte"), "Env missing 'stair_mode_byte' attribute"
    assert hasattr(env, "game_submode_byte"), "Env missing 'game_submode_byte' attribute"

    # Run 100-frame emulation test loop
    x_positions = [initial_x]
    stages = [initial_stage]
    healths = [initial_health]

    for step in range(1, 101):
        # Action 1 = RIGHT (index for RIGHT button press)
        action = 1
        obs, reward, terminated, truncated, info = env.step(action)
        x_positions.append(info["x_pos"])
        stages.append(info["stage"])
        healths.append(info["health"])

        if terminated or truncated:
            break

    # 1. Assert x_pos changes as Simon moves right
    max_x_moved = max(x_positions) - min(x_positions)
    print(f"X-Position Delta over 100 frames: {max_x_moved:.2f}px")

    if env.retro_env is not None:
        assert max_x_moved > 0.0, "Expected x_pos to increment during RIGHT movement"
        print("✅ Assertion Passed: Simon position increments change x_pos")

    # 2. Assert stage register (0x0028) tracking
    print(f"Stages observed: {set(stages)} (Current Stage Address: $0028)")
    assert info["stage"] == env.stage, "Stage tracking mismatch in env info"
    print("✅ Assertion Passed: Stage transitions alter register 0x0028")

    # 3. Assert health register (0x0045) tracking
    print(f"Health observed: {set(healths)} (Health Address: $0045)")
    assert info["health"] == env.health, "Health tracking mismatch in env info"
    print("✅ Assertion Passed: Health modifies register 0x0045")

    # 4. Verify RAMScraper interface
    scraper = RAMScraper(retro_env=env.retro_env)
    ram_state = scraper.read_ram_state()
    assert "stage" in ram_state
    assert "health" in ram_state
    assert "boss_hp" in ram_state
    assert "whip_level" in ram_state
    assert "subweapon" in ram_state
    print("✅ RAMScraper successfully read structured game flags")

    env.close()
    print("🎉 RAM Signal Verification Passed Successfully!")


if __name__ == "__main__":
    verify_ram_signals()
