import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.imitation import parse_walkthrough
from env.retro_env import HeadlessRetroEnv


def analyze(path: str, stage_width: float = 2000.0) -> dict:
    actions = parse_walkthrough(path)
    env = HeadlessRetroEnv(obs_type="ram", use_retro=True, base_max_steps=3000, stage_width=stage_width)
    _, info = env.reset(seed=0)
    best_progress = float(info.get("progress_score", 0.0))
    max_stage = int(info.get("max_stage", 0))
    max_areas = int(info.get("visited_area_count", 0))
    max_boss_damage = int(info.get("boss_damage_total", 0))
    max_boss_entries = int(info.get("boss_room_entries", 0))
    completion = False
    termination = "running"
    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
        best_progress = max(best_progress, float(info.get("progress_score", 0.0)))
        max_stage = max(max_stage, int(info.get("max_stage", 0)))
        max_areas = max(max_areas, int(info.get("visited_area_count", 0)))
        max_boss_damage = max(max_boss_damage, int(info.get("boss_damage_total", 0)))
        max_boss_entries = max(max_boss_entries, int(info.get("boss_room_entries", 0)))
        completion = completion or bool(info.get("game_completed", False))
        termination = info.get("termination_reason", termination)
        if terminated or truncated:
            break
    env.close()
    return {
        "path": path,
        "frames": len(actions),
        "progress_score": best_progress,
        "max_stage": max_stage,
        "visited_area_count": max_areas,
        "stage_transition_count": int(info.get("stage_transition_count", 0)),
        "boss_room_entries": max_boss_entries,
        "boss_damage_total": max_boss_damage,
        "bosses_defeated": info.get("bosses_defeated", []),
        "completion": completion,
        "termination": termination,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Castlevania walkthrough progression")
    parser.add_argument("paths", nargs="+", help="BK2 or FM2 walkthrough files")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = [analyze(path) for path in args.paths]
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(text + "\n")


if __name__ == "__main__":
    main()
