import os
import sys
import glob
import json
import time
import shutil
import torch
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audit.audit_logger import AuditLogger

TRAINING_AUDIT_FILE = "training_audit.json"
ACTIVE_PARAMS_FILE = "config/active_training_params.json"
CHECKPOINT_DIR = "checkpoints"
RESUME_TARGET_FILE = "checkpoints/resume_target.pt"


def load_training_audit(audit_file: str = TRAINING_AUDIT_FILE) -> Dict[str, Any]:
    if not os.path.exists(audit_file):
        print(f"Warning: Training audit file '{audit_file}' not found.")
        return {}
    try:
        with open(audit_file, "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error reading '{audit_file}': {e}")
        return {}


def load_active_params(params_file: str = ACTIVE_PARAMS_FILE) -> Dict[str, Any]:
    default_params = {
        "initial_lr": 1.5e-4,
        "ent_coef": 0.05,
        "gamma": 0.99,
        "time_penalty": -0.02,
        "score_weight": 0.01,
        "progress_multiplier": 1.0,
        "penalize_zero_velocity": False,
        "zero_velocity_penalty": -0.05,
        "lr_warmup_steps": 0,
        "reinit_policy_entropy": False
    }
    if os.path.exists(params_file):
        try:
            with open(params_file, "r") as f:
                saved = json.load(f)
                default_params.update(saved)
        except Exception as e:
            print(f"Warning: Could not read '{params_file}': {e}")
    return default_params


def adjust_hyperparameters(failure_reason: str, current_params: Dict[str, Any]) -> Dict[str, Any]:
    new_params = dict(current_params)
    # Reset single-cycle flags
    new_params["reinit_policy_entropy"] = False

    if failure_reason == "STAGNATION_PLATEAU":
        curr_ent = float(new_params.get("ent_coef", 0.05))
        max_ent = max(0.0, float(new_params.get("max_ent_coef", curr_ent * 1.5)))
        curr_lr = float(new_params.get("initial_lr", 1.5e-4))
        curr_prog = float(new_params.get("progress_multiplier", 1.0))

        new_params["ent_coef"] = round(min(curr_ent * 1.5, max_ent), 6)
        new_params["initial_lr"] = max(3e-5, round(curr_lr * 0.75, 8))
        new_params["progress_multiplier"] = round(min(curr_prog * 1.25, 3.0), 4)
        print(f"[AutoTuner] Adapted for STAGNATION_PLATEAU: ent_coef={new_params['ent_coef']}, lr={new_params['initial_lr']}, progress_mult={new_params['progress_multiplier']}")

    elif failure_reason == "REWARD_HACKING":
        new_params["time_penalty"] = -0.05
        new_params["penalize_zero_velocity"] = True
        new_params["zero_velocity_penalty"] = -0.05
        new_params["score_weight"] = 0.0  # Reset non-displacement rewards to zero
        print(f"[AutoTuner] Adapted for REWARD_HACKING: time_penalty=-0.05, zero_vel_penalty=-0.05, score_weight=0.0")

    elif failure_reason == "COLLAPSED_EXPLORATION":
        curr_ent = float(new_params.get("ent_coef", 0.05))
        curr_warmup = int(new_params.get("lr_warmup_steps", 0))
        curr_lr = float(new_params.get("initial_lr", 1.5e-4))

        new_params["reinit_policy_entropy"] = True
        new_params["ent_coef"] = max(0.05, round(curr_ent * 2.0, 6))
        new_params["initial_lr"] = round(curr_lr * 1.25, 8)
        new_params["lr_warmup_steps"] = curr_warmup + 100
        print(f"[AutoTuner] Adapted for COLLAPSED_EXPLORATION: reinit_policy_entropy=True, ent_coef={new_params['ent_coef']}, lr_warmup_steps={new_params['lr_warmup_steps']}")

    else:
        print(f"[AutoTuner] Unrecognized or general failure reason '{failure_reason}'. No hyperparameter adjustments applied.")

    return new_params


def save_active_params(params: Dict[str, Any], params_file: str = ACTIVE_PARAMS_FILE):
    os.makedirs(os.path.dirname(os.path.abspath(params_file)), exist_ok=True)
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    print(f"[AutoTuner] Saved active hyperparameters to '{params_file}'.")


def resolve_best_distance_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR, resume_target: str = RESUME_TARGET_FILE) -> Optional[str]:
    """
    Scans checkpoints/ directory for all saved best models (e.g. best_ppo_agent_dist_*.pt).
    Parses metadata headers to select model checkpoint with highest stage (primary) and max_x_pos (secondary).
    Copies this champion file to checkpoints/resume_target.pt.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not os.path.exists(checkpoint_dir):
        return None

    all_ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    # Filter out resume_target.pt to avoid self-selection
    candidate_ckpts = [c for c in all_ckpts if os.path.basename(c) != os.path.basename(resume_target)]

    if not candidate_ckpts:
        print(f"[AutoTuner] No candidate checkpoints found in '{checkpoint_dir}'.")
        return None

    best_ckpt_path = None
    highest_score = (-1, -1.0, -1.0)
    best_stage = 0
    highest_max_x = 0.0

    for ckpt_path in candidate_ckpts:
        fname = os.path.basename(ckpt_path)
        x_from_file = -1.0

        if "dist_" in fname:
            try:
                parts = fname.replace(".pt", "").split("dist_")
                if len(parts) > 1 and parts[1].isdigit():
                    x_from_file = float(parts[1])
            except Exception:
                pass

        x_from_payload = -1.0
        stage_from_payload = 0
        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                stage_from_payload = int(payload.get("max_stage", payload.get("stage", 0)))
                if "max_x_pos" in payload:
                    x_from_payload = float(payload["max_x_pos"])
        except Exception:
            pass

        effective_max_x = max(x_from_file, x_from_payload)
        if effective_max_x < 0.0:
            effective_max_x = 0.0
        effective_stage = max(0, stage_from_payload)
        mtime = os.path.getmtime(ckpt_path)

        score = (effective_stage, effective_max_x, mtime)
        if score > highest_score or best_ckpt_path is None:
            highest_score = score
            best_stage = effective_stage
            highest_max_x = effective_max_x
            best_ckpt_path = ckpt_path

    if best_ckpt_path:
        shutil.copyfile(best_ckpt_path, resume_target)
        print(f"[AutoTuner] Selected champion checkpoint '{best_ckpt_path}' (Stage: {best_stage}, Max X: {highest_max_x:.1f}) -> copied to '{resume_target}'.")
        return resume_target

    return None


def run_auto_tuner():
    print("\n=======================================================")
    print("EXECUTING AUTO-TUNER & DISTANCE CHECKPOINT RESOLVER")
    print("=======================================================")

    audit_data = load_training_audit()
    failure_reason = audit_data.get("failure_reason", "UNKNOWN")
    print(f"[AutoTuner] Loaded failure reason from training_audit.json: {failure_reason}")

    current_params = load_active_params()
    updated_params = adjust_hyperparameters(failure_reason, current_params)
    save_active_params(updated_params)

    resume_ckpt = resolve_best_distance_checkpoint()

    audit_logger = AuditLogger("training_audit.jsonl")
    audit_logger.log_event("auto_tuner_adaptation", {
        "failure_reason": failure_reason,
        "adapted_params": updated_params,
        "resume_target_checkpoint": resume_ckpt,
        "timestamp": time.time()
    })

    print("=======================================================\n")


if __name__ == "__main__":
    run_auto_tuner()
