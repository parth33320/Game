import argparse
import json
import os
import sys
from typing import Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.imitation import parse_walkthrough
from env.retro_env import HeadlessRetroEnv


def analyze(path: str, stage_width: float = 2000.0, savestate_dir: str = "checkpoints/savestates") -> dict:
    os.makedirs(savestate_dir, exist_ok=True)
    try:
        actions = parse_walkthrough(path)
    except Exception as e:
        return {"path": path, "error": str(e)}

    # Ensure base_max_steps accommodates the full length of the recorded action sequence
    max_steps = max(len(actions) + 500, 3000)
    env = HeadlessRetroEnv(obs_type="ram", use_retro=True, base_max_steps=max_steps, stage_width=stage_width)
    _, info = env.reset(seed=0)

    best_progress = float(info.get("progress_score", 0.0))
    max_stage = int(info.get("max_stage", 0))
    max_areas = int(info.get("visited_area_count", 0))
    max_boss_damage = int(info.get("boss_damage_total", 0))
    max_boss_entries = int(info.get("boss_room_entries", 0))
    completion = False
    termination = "running"

    saved_stages = set()
    initial_stage = int(info.get("stage", 0))

    # Capture initial stage 0 savestate if not yet present
    stage_0_path = os.path.join(savestate_dir, f"stage_{initial_stage}.state")
    if not os.path.exists(stage_0_path):
        state_bytes = env.capture_savestate()
        if state_bytes:
            with open(stage_0_path, "wb") as f:
                f.write(state_bytes)
            saved_stages.add(initial_stage)

    for frame_idx, action in enumerate(actions):
        _, _, terminated, truncated, info = env.step(action)
        curr_stage = int(info.get("stage", 0))

        # Generate clean stage_0.state through stage_18.state upon entering new substage
        if curr_stage not in saved_stages:
            state_path = os.path.join(savestate_dir, f"stage_{curr_stage}.state")
            state_bytes = env.capture_savestate()
            if state_bytes:
                with open(state_path, "wb") as f:
                    f.write(state_bytes)
                saved_stages.add(curr_stage)

        best_progress = max(best_progress, float(info.get("progress_score", 0.0)))
        max_stage = max(max_stage, curr_stage)
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
        "saved_stages": sorted(list(saved_stages))
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Castlevania walkthrough progression")
    parser.add_argument("paths", nargs="+", help="BK2 or FM2 walkthrough files")
    parser.add_argument("--output", default="walkthrough_progress.json")
    parser.add_argument("--savestate-dir", default="checkpoints/savestates")
    args = parser.parse_args()

    report = []
    for path in args.paths:
        if os.path.isfile(path):
            res = analyze(path, savestate_dir=args.savestate_dir)
            report.append(res)
            print(f"Analyzed {path}: max_stage={res.get('max_stage')}, progress={res.get('progress_score')}, completion={res.get('completion')}")

    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(text + "\n")
        print(f"Saved analysis report to {args.output}")


if __name__ == "__main__":
    main()
