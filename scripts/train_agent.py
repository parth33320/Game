import sys
import os
import glob
import time
import json
import copy
import statistics
import queue as queue_module
import multiprocessing as mp
import numpy as np
import torch
import torch.optim as optim

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.model import ActorCriticPPO
from audit.audit_logger import AuditLogger
from env.retro_env import HeadlessRetroEnv

ACTIVE_PARAMS_FILE = "config/active_training_params.json"
RESUME_TARGET_FILE = "checkpoints/resume_target.pt"
TRAINING_AUDIT_FILE = "training_audit.json"

def curriculum_phase_for_stage_and_distance(stage: int, distance: float, active_params: dict) -> int:
    if not active_params.get("curriculum_enabled", True):
        return 0
    thresholds = active_params.get("curriculum_thresholds", [{"stage": 0, "x_pos": 500}, {"stage": 0, "x_pos": 1000}, {"stage": 1, "x_pos": 500}])
    phase = 0
    for item in thresholds:
        if isinstance(item, dict):
            req_stage = int(item.get("stage", 0))
            req_x = float(item.get("x_pos", 0.0))
            if stage > req_stage or (stage == req_stage and distance >= req_x):
                phase += 1
        elif isinstance(item, (int, float)):
            if distance >= float(item):
                phase += 1
    return phase

def curriculum_phase_for_distance(distance: float, active_params: dict, stage: int = 0) -> int:
    return curriculum_phase_for_stage_and_distance(stage, distance, active_params)

def load_active_params(params_file: str = ACTIVE_PARAMS_FILE) -> dict:
    default_params = {
        "initial_lr": 1.5e-4,
        "lr_floor": 1e-6,
        "ent_coef": 0.05,
        "gamma": 0.99,
        "stagnation_frame_budget": 120,
        "stagnation_min_episodes": 1000,
        "stagnation_patience_episodes": 2000,
        "penalize_zero_velocity": True,
        "zero_velocity_penalty": -10.0,
        "total_episodes": 100000,
        "checkpoint_interval": 5
    }
    if os.path.exists(params_file):
        try:
            with open(params_file, "r") as f:
                saved_params = json.load(f)
                default_params.update(saved_params)
                print(f"🛡️ Enforcing Hyperparameters from config: {default_params}")
        except Exception as e:
            print(f"Warning: Failed to load {params_file} ({e}). Using defaults.")
    return default_params

def find_target_checkpoint(checkpoint_dir: str, default_target: str = "checkpoints/ppo_agent_ep5000.pt") -> str:
    candidates = glob.glob(os.path.join(checkpoint_dir, "*.pt")) if os.path.exists(checkpoint_dir) else []
    if os.path.exists(RESUME_TARGET_FILE) and RESUME_TARGET_FILE not in candidates:
        candidates.append(RESUME_TARGET_FILE)
    scored_candidates = []
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            distance = float(payload.get("max_x_pos", -1.0)) if isinstance(payload, dict) else -1.0
        except Exception:
            distance = -1.0
        is_best_checkpoint = os.path.basename(candidate) == "best_ppo_agent_dist.pt"
        is_latest_checkpoint = os.path.basename(candidate) == "model_weights_latest.pt"
        scored_candidates.append((distance, is_best_checkpoint, is_latest_checkpoint, os.path.getmtime(candidate), candidate))
    if scored_candidates:
        return max(scored_candidates)[4]
    if os.path.exists(default_target):
        return default_target
    if os.path.exists(checkpoint_dir):
        ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if ckpts:
            ckpts.sort(key=os.path.getmtime, reverse=True)
            return ckpts[0]
    return default_target

def trigger_early_stopping(failure_reason: str, episode: int, total_reward: float, max_x_pos: float, entropy_val: float, active_params: dict, audit_logger: AuditLogger, exit_code: int = 42, metrics=None):
    print(f"\n=======================================================")
    print(f"EARLY STOPPING TRIGGERED: Failure Reason = '{failure_reason}'")
    print(f"Episode: {episode} | Reward: {total_reward:.2f} | Max X: {max_x_pos:.1f} | Entropy: {entropy_val:.4f}")
    if metrics:
        print(f"Evidence metrics: {json.dumps(metrics, sort_keys=True)}")
    print(f"=======================================================\n")
    audit_payload = {
        "status": "EARLY_STOPPING",
        "failure_reason": failure_reason,
        "episode": episode,
        "total_reward": total_reward,
        "max_x_pos": max_x_pos,
        "entropy": entropy_val,
        "metrics": metrics or {},
        "active_params": active_params,
        "timestamp": time.time()
    }
    with open(TRAINING_AUDIT_FILE, "w") as f:
        json.dump(audit_payload, f, indent=2)
    audit_logger.log_event("early_stopping_triggered", audit_payload)
    sys.exit(exit_code)

def save_training_checkpoint(path: str, episode: int, max_x_pos: float, model: ActorCriticPPO, optimizer: optim.Optimizer, info=None):
    metadata = info or {}
    torch.save({
        "episode": episode,
        "max_x_pos": max_x_pos,
        "stage": int(metadata.get("stage", 0)),
        "coarse_screen": int(metadata.get("coarse_screen", 0)),
        "fine_x": int(metadata.get("fine_x", 0)),
        "boss_hp": int(metadata.get("boss_hp", 16)),
        "in_boss_room": bool(metadata.get("in_boss_room", False)),
        "health": int(metadata.get("health", 16)),
        "lives": int(metadata.get("lives", 3)),
        "max_stage": int(metadata.get("max_stage", metadata.get("stage", 0))),
        "progress_score": float(metadata.get("progress_score", max_x_pos)),
        "area_id": metadata.get("area_id", "unknown"),
        "visited_area_count": int(metadata.get("visited_area_count", 0)),
        "stage_transition_count": int(metadata.get("stage_transition_count", 0)),
        "boss_room_entries": int(metadata.get("boss_room_entries", 0)),
        "bosses_defeated": list(metadata.get("bosses_defeated", [])),
        "boss_damage_total": int(metadata.get("boss_damage_total", 0)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)

def collect_rollout_worker(args):
    worker_id, episode, model_state, active_params = args[:4]
    deterministic = bool(args[4]) if len(args) > 4 else False
    torch.set_num_threads(1)
    base_steps = int(active_params.get("curriculum_steps", 400)) + int(active_params.get("curriculum_phase", 0)) * int(active_params.get("curriculum_phase_step_bonus", 300))
    env = HeadlessRetroEnv(
        obs_type="ram",
        use_retro=True,
        base_max_steps=base_steps,
        reward_params=active_params,
        stage_width=float(active_params.get("stage_width", 2000.0)),
    )
    curriculum_phase = int(active_params.get("curriculum_phase", 0))
    state_dir = active_params.get("savestate_dir", "checkpoints/savestates")
    state_path = os.path.join(state_dir, f"phase_{curriculum_phase}.state")
    state_loaded = env.load_savestate(state_path)
    model = ActorCriticPPO(input_dim=15, num_actions=len(env.ACTION_NAMES), is_mlp=True)
    model.load_state_dict(model_state)
    model.eval()

    observations = []
    actions = []
    rewards = []
    log_probs = []
    entropies = []
    obs, info = env.reset(seed=episode * 100 + worker_id)
    info["curriculum_phase"] = curriculum_phase
    info["savestate_loaded"] = state_loaded
    done = False
    max_x_pos = 0.0
    termination_reason = "unknown"
    milestone_states = {}
    thresholds = active_params.get("curriculum_thresholds", [])

    with torch.no_grad():
        while not done:
            observations.append(obs.copy())
            state_tensor = torch.as_tensor(obs, dtype=torch.float32)
            if deterministic:
                logits, _ = model(state_tensor)
                distribution = torch.distributions.Categorical(logits=logits)
                action = int(torch.argmax(logits, dim=-1).item())
                log_prob = distribution.log_prob(torch.tensor(action))
                entropy = distribution.entropy()
            else:
                action, log_prob, _, entropy = model.get_action(state_tensor)
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            log_probs.append(float(log_prob.item()))
            entropies.append(float(entropy.item()))
            done = terminated or truncated
            max_x_pos = max(max_x_pos, float(info.get("max_x_pos", 0.0)))
            curr_stage = int(info.get("stage", 0))
            for index, threshold in enumerate(thresholds, start=1):
                met = False
                if isinstance(threshold, dict):
                    req_stage = int(threshold.get("stage", 0))
                    req_x = float(threshold.get("x_pos", 0.0))
                    met = curr_stage > req_stage or (curr_stage == req_stage and max_x_pos >= req_x)
                elif isinstance(threshold, (int, float)):
                    met = max_x_pos >= float(threshold)
                if met and index not in milestone_states:
                    state = env.capture_savestate()
                    if state is not None:
                        milestone_states[index] = state
            termination_reason = info.get("termination_reason", termination_reason)

    env.close()
    return observations, actions, rewards, log_probs, max_x_pos, termination_reason, info, statistics.mean(entropies), milestone_states

def _rollout_process_entry(args, result_queue):
    try:
        result_queue.put(("ok", args[0], collect_rollout_worker(args)))
    except Exception as error:
        result_queue.put(("error", args[0], type(error).__name__, str(error)))

class RolloutBatchExecutor:
    def __init__(self, context, worker_count):
        self.context = context
        self.worker_count = worker_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def map(self, function, batch):
        if function is not collect_rollout_worker:
            raise ValueError("RolloutBatchExecutor only supports collect_rollout_worker")
        result_queue = self.context.Queue()
        processes = [self.context.Process(target=_rollout_process_entry, args=(args, result_queue)) for args in batch]
        try:
            for process in processes:
                process.start()

            results = {}
            for _ in processes:
                try:
                    message = result_queue.get(timeout=300)
                except queue_module.Empty as error:
                    raise RuntimeError("Worker exited or timed out before returning its rollout") from error
                if message[0] == "error":
                    raise RuntimeError(f"Worker {message[1]} failed: {message[2]}: {message[3]}")
                results[message[1]] = message[2]

            for process in processes:
                process.join()
            return [results[args[0]] for args in batch]
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            result_queue.close()
            result_queue.join_thread()

def train_parallel_ppo_agent(checkpoint_dir: str, params_file: str, active_params: dict, worker_count: int):
    os.makedirs(checkpoint_dir, exist_ok=True)
    initial_lr = float(active_params.get("initial_lr", 1.5e-4))
    lr_floor = float(active_params.get("lr_floor", 3e-5))
    ent_coef = min(max(0.0, float(active_params.get("ent_coef", 0.05))),
                   max(0.0, float(active_params.get("max_ent_coef", 0.08))))
    max_ent_coef = max(0.0, float(active_params.get("max_ent_coef", ent_coef)))
    exploration_ramp_batches = max(0, int(active_params.get("exploration_ramp_batches", 50)))
    exploration_entropy_step = max(0.0, float(active_params.get("exploration_entropy_step", 0.01)))
    current_ent_coef = ent_coef
    gamma = float(active_params.get("gamma", 0.99))
    total_episodes = int(active_params.get("total_episodes", 100000))
    checkpoint_interval = int(active_params.get("checkpoint_interval", 25))
    ppo_clip = float(active_params.get("ppo_clip", 0.2))
    checkpoint = find_target_checkpoint(checkpoint_dir)
    recovery_window = int(active_params.get("best_policy_recovery_batches", 100))

    model = ActorCriticPPO(input_dim=15, num_actions=9, is_mlp=True)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    start_episode = 1
    best_max_x_pos = 0.0
    if os.path.exists(checkpoint):
        model.load_checkpoint_weights(checkpoint, optimizer=optimizer)
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            start_episode = int(raw.get("episode", 0)) + 1
            best_max_x_pos = float(raw.get("max_x_pos", 0.0))

    best_policy_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_policy_optimizer_state = copy.deepcopy(optimizer.state_dict())
    batches_without_improvement = 0

    context = mp.get_context("spawn")
    print(f"Starting {worker_count} parallel Simons from episode {start_episode} (best distance: {best_max_x_pos:.1f})")
    histories = []
    audit_logger = AuditLogger("training_audit.jsonl")

    eval_interval = int(active_params.get("evaluation_interval_batches", 50))
    with RolloutBatchExecutor(context, worker_count) as pool:
        for batch_start in range(start_episode, start_episode + total_episodes, worker_count):
            current_stage = int(best_info.get("stage", 0)) if isinstance(best_info, dict) else 0
            active_params["curriculum_phase"] = curriculum_phase_for_stage_and_distance(current_stage, best_max_x_pos, active_params)
            batch = [(index, batch_start + index, {key: value.cpu() for key, value in model.state_dict().items()}, active_params)
                     for index in range(worker_count)]
            try:
                results = pool.map(collect_rollout_worker, batch)
            except Exception as error:
                trigger_early_stopping(
                    "EMULATOR_FAILURE",
                    batch_start,
                    0.0,
                    0.0,
                    0.0,
                    active_params,
                    audit_logger,
                    exit_code=2,
                    metrics={"error_type": type(error).__name__, "error": str(error), "worker_count": worker_count},
                )
            all_observations, all_actions, all_rewards = [], [], []
            batch_max = 0.0
            old_log_probs = []
            best_info = {}
            best_milestone_states = {}
            entropy_values = []
            termination_counts = {}
            hacking_workers = 0
            reward_totals = []
            for observations, actions, rewards, rollout_log_probs, max_x_pos, termination_reason, info, rollout_entropy, milestone_states in results:
                all_observations.extend(observations)
                all_actions.extend(actions)
                all_rewards.extend(rewards)
                old_log_probs.extend(rollout_log_probs)
                if max_x_pos > batch_max:
                    batch_max = max_x_pos
                    best_info = info
                    best_milestone_states = milestone_states
                histories.append(max_x_pos)
                entropy_values.append(rollout_entropy)
                termination_counts[termination_reason] = termination_counts.get(termination_reason, 0) + 1
                hacking_workers += int(bool(info.get("reward_hacking_detected", False)))
                reward_totals.append(sum(rewards))

            obs_tensor = torch.as_tensor(np.asarray(all_observations), dtype=torch.float32)
            action_tensor = torch.as_tensor(all_actions, dtype=torch.long)
            reward_tensor = torch.as_tensor(all_rewards, dtype=torch.float32)
            returns = []
            offset = 0
            for observations, _, rewards, _, _, _, _, _, _ in results:
                discounted = 0.0
                local_returns = []
                for reward in reversed(rewards):
                    discounted = reward + gamma * discounted
                    local_returns.insert(0, discounted)
                returns.extend(local_returns)
                offset += len(observations)
            returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
            logits, values = model(obs_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(action_tensor)
            old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32)
            advantages = returns_tensor - values.squeeze(-1).detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratios = torch.exp(log_probs - old_log_probs_tensor)
                clipped_ratios = torch.clamp(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip)
                policy_loss = -torch.minimum(ratios * advantages, clipped_ratios * advantages).mean()
                loss = (policy_loss
                    + 0.5 * (returns_tensor - values.squeeze(-1)).pow(2).mean()
                    - current_ent_coef * distribution.entropy().mean())
                if not torch.isfinite(loss):
                    trigger_early_stopping(
                        "INVALID_UPDATE",
                        episode=batch_start,
                        total_reward=float(reward_tensor.sum()),
                        max_x_pos=batch_max,
                        entropy_val=statistics.mean(entropy_values),
                        active_params=active_params,
                        audit_logger=audit_logger,
                        exit_code=2,
                        metrics={"loss": float(loss.detach()), "batch_samples": len(all_rewards)},
                    )
            optimizer.zero_grad()
            loss.backward()
            non_finite_gradients = any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            if non_finite_gradients:
                trigger_early_stopping(
                    "INVALID_UPDATE",
                    batch_start,
                    float(reward_tensor.sum()),
                    batch_max,
                    statistics.mean(entropy_values),
                    active_params,
                    audit_logger,
                    exit_code=2,
                    metrics={"loss": float(loss.detach()), "non_finite_gradients": True},
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(param_group["lr"], lr_floor)

            episode = min(batch_start + worker_count - 1, start_episode + total_episodes - 1)
            print(f"Episodes {batch_start}-{episode} | batch max distance: {batch_max:.1f}")

            batch_metrics = {
                "worker_count": worker_count,
                "hacking_workers": hacking_workers,
                "termination_counts": termination_counts,
                "mean_reward": statistics.mean(reward_totals),
                "mean_entropy": statistics.mean(entropy_values),
                "batch_max_distance": batch_max,
                "best_distance": best_max_x_pos,
            }
            hacking_threshold = max(2, (worker_count + 1) // 2)
            if hacking_workers >= hacking_threshold and batch_max < 10.0:
                trigger_early_stopping(
                    "REWARD_HACKING",
                    episode,
                    float(reward_tensor.sum()),
                    batch_max,
                    batch_metrics["mean_entropy"],
                    active_params,
                    audit_logger,
                    exit_code=2,
                    metrics=batch_metrics,
                )
            min_entropy = float(active_params.get("min_entropy", 0.25))
            if len(histories) >= 100 and batch_metrics["mean_entropy"] < min_entropy:
                trigger_early_stopping(
                    "COLLAPSED_EXPLORATION",
                    episode,
                    float(reward_tensor.sum()),
                    batch_max,
                    batch_metrics["mean_entropy"],
                    active_params,
                    audit_logger,
                    exit_code=2,
                    metrics=batch_metrics | {"min_entropy": min_entropy},
                )
            regression_window = int(active_params.get("regression_window_batches", 100))
            regression_ratio = float(active_params.get("regression_ratio", 0.70))
            if len(histories) >= regression_window and best_max_x_pos > 0:
                recent = histories[-regression_window:]
                if statistics.mean(recent) < best_max_x_pos * regression_ratio and max(recent) < best_max_x_pos * 0.85:
                    trigger_early_stopping(
                        "CATASTROPHIC_REGRESSION",
                        episode,
                        float(reward_tensor.sum()),
                        batch_max,
                        batch_metrics["mean_entropy"],
                        active_params,
                        audit_logger,
                        exit_code=2,
                        metrics=batch_metrics | {
                            "regression_window": regression_window,
                            "recent_mean_distance": statistics.mean(recent),
                            "regression_ratio": regression_ratio,
                        },
                    )
            if batch_max > best_max_x_pos:
                best_max_x_pos = batch_max
                best_policy_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                best_policy_optimizer_state = copy.deepcopy(optimizer.state_dict())
                batches_without_improvement = 0
                current_ent_coef = ent_coef
                save_training_checkpoint(os.path.join(checkpoint_dir, "best_ppo_agent_dist.pt"), episode, batch_max, model, optimizer, best_info)
            else:
                batches_without_improvement += 1
                if batches_without_improvement >= exploration_ramp_batches:
                    ramp_batches = batches_without_improvement - exploration_ramp_batches + 1
                    current_ent_coef = min(
                        ent_coef + ramp_batches * exploration_entropy_step,
                        max_ent_coef,
                    )
            state_dir = active_params.get("savestate_dir", os.path.join(checkpoint_dir, "savestates"))
            os.makedirs(state_dir, exist_ok=True)
            for phase, state in best_milestone_states.items():
                stage_path = os.path.join(state_dir, f"stage_{phase}.state")
                if not os.path.exists(stage_path):
                    with open(stage_path, "wb") as stream:
                        stream.write(state)
                phase_path = os.path.join(state_dir, f"phase_{phase}.state")
                if not os.path.exists(phase_path):
                    with open(phase_path, "wb") as stream:
                        stream.write(state)
            if batches_without_improvement >= recovery_window:
                model.load_state_dict(best_policy_state)
                optimizer.load_state_dict(best_policy_optimizer_state)
                batches_without_improvement = 0
                current_ent_coef = ent_coef
                print(f"Recovered best policy after {recovery_window} batches without a new distance record; entropy reset to {current_ent_coef:.4f}")
            if episode % checkpoint_interval == 0 or batch_start == start_episode:
                save_training_checkpoint(os.path.join(checkpoint_dir, "model_weights_latest.pt"), episode, batch_max, model, optimizer, best_info)

            batch_number = len(histories) // worker_count
            if batch_number % eval_interval == 0:
                evaluation_batch = [
                    (index, episode + index, {key: value.cpu() for key, value in model.state_dict().items()}, active_params, True)
                    for index in range(worker_count)
                ]
                # Evaluation runs in the learner process so a worker-pool pipe
                # failure cannot interrupt training at the audit boundary.
                evaluation = [collect_rollout_worker(item) for item in evaluation_batch]
                evaluation_distances = [item[4] for item in evaluation]
                evaluation_reasons = [item[5] for item in evaluation]
                evaluation_stages = [int(item[6].get("stage", 0)) for item in evaluation]
                evaluation_metrics = {
                    "episode": episode,
                    "runs": worker_count,
                    "curriculum_phase": int(active_params.get("curriculum_phase", 0)),
                    "savestate_loaded_runs": sum(bool(item[6].get("savestate_loaded", False)) for item in evaluation),
                    "mean_distance": statistics.mean(evaluation_distances),
                    "median_distance": statistics.median(evaluation_distances),
                    "best_distance": max(evaluation_distances),
                    "mean_stage": statistics.mean(evaluation_stages),
                    "mean_max_stage": statistics.mean(int(item[6].get("max_stage", 0)) for item in evaluation),
                    "completion_rate": evaluation_reasons.count("game_completed") / worker_count,
                    "termination_counts": {reason: evaluation_reasons.count(reason) for reason in set(evaluation_reasons)},
                }
                audit_logger.log_event("policy_evaluation", evaluation_metrics)
                print(f"Evaluation episode {episode}: phase={evaluation_metrics['curriculum_phase']} | savestates={evaluation_metrics['savestate_loaded_runs']}/{worker_count} | mean={evaluation_metrics['mean_distance']:.1f} | stage={evaluation_metrics['mean_stage']:.1f} | max_stage={evaluation_metrics['mean_max_stage']:.1f} | completion={evaluation_metrics['completion_rate']:.0%}")

            if len(histories) >= 2000 and episode >= int(active_params.get("stagnation_min_episodes", 1000)):
                window = histories[-2000:]
                window_mean = statistics.mean(window)
                window_std = statistics.pstdev(window)
                window_trend = window[-1] - window[0]
                if max(window) - window[0] < 20.0:
                    metrics = {
                        "window_size": len(window),
                        "window_first": window[0],
                        "window_last": window[-1],
                        "window_best": max(window),
                        "window_mean": window_mean,
                        "window_median": statistics.median(window),
                        "window_std": window_std,
                        "window_trend": window_trend,
                        "best_distance": best_max_x_pos,
                        "best_gap": best_max_x_pos - max(window),
                        "mean_entropy": statistics.mean(entropy_values),
                    }
                    trigger_early_stopping("STAGNATION_PLATEAU", episode, float(reward_tensor.sum()), batch_max, metrics["mean_entropy"], active_params, audit_logger, exit_code=2, metrics=metrics)

def train_ppo_agent(checkpoint_dir: str = "checkpoints", params_file: str = ACTIVE_PARAMS_FILE):
    os.makedirs(checkpoint_dir, exist_ok=True)
    active_params = load_active_params(params_file)
    worker_count = min(max(1, int(active_params.get("num_workers", 1))), max(1, (os.cpu_count() or 2) - 1))
    if worker_count > 1:
        return train_parallel_ppo_agent(checkpoint_dir, params_file, active_params, worker_count)

    total_episodes = int(active_params.get("total_episodes", 100000))
    checkpoint_interval = int(active_params.get("checkpoint_interval", 5))
    initial_lr = float(active_params.get("initial_lr", 1.5e-4))
    ent_coef = float(active_params.get("ent_coef", 0.05))
    max_ent_coef = max(0.0, float(active_params.get("max_ent_coef", ent_coef)))
    ent_coef = min(max(0.0, ent_coef), max_ent_coef)
    gamma = float(active_params.get("gamma", 0.99))

    # Secure defensive parameters
    lr_floor = float(active_params.get("lr_floor", 1e-6))
    stagnation_frame_budget = int(active_params.get("stagnation_frame_budget", 120))
    stagnation_min_episodes = int(active_params.get("stagnation_min_episodes", 1000))
    stagnation_patience_episodes = int(active_params.get("stagnation_patience_episodes", 2000))
    zero_velocity_penalty = float(active_params.get("zero_velocity_penalty", -10.0))

    print("Connecting to the Castlevania NES environment...")
    env = HeadlessRetroEnv(obs_type="ram", use_retro=True, reward_params=active_params)

    device = "cpu"
    model = ActorCriticPPO(input_dim=15, num_actions=len(env.ACTION_NAMES), is_mlp=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    audit_logger = AuditLogger("training_audit.jsonl")

    ckpt_to_load = find_target_checkpoint(checkpoint_dir)
    start_episode = 1
    best_max_x_pos = 0.0

    if os.path.exists(ckpt_to_load):
        print(f"Loading pre-trained checkpoint weights from {ckpt_to_load}...")
        model.load_checkpoint_weights(ckpt_to_load, optimizer=optimizer)
        try:
            raw_ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
            if isinstance(raw_ckpt, dict):
                if "episode" in raw_ckpt:
                    start_episode = int(raw_ckpt["episode"]) + 1
                if "max_x_pos" in raw_ckpt:
                    best_max_x_pos = float(raw_ckpt["max_x_pos"])
        except Exception as e:
            print(f"Warning reading checkpoint metadata: {e}")

    print(f"Starting PyTorch PPO Platformer Agent Training from episode {start_episode} for {total_episodes} episodes (Best Max X: {best_max_x_pos:.1f})...")

    rolling_max_x_history = []
    rolling_reward_history = []

    for episode in range(start_episode, start_episode + total_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_step_entropy = 0.05
        prev_x_pos = 0.0
        log_probs, rewards, values, dones = [], [], [], []
        entropies = []

        while not done:
            # The wrapper derives completion from the ROM's stage/game-state RAM.
            if info.get("game_completed", False):
                print("🎉 SUCCESS: AI Player completed Castlevania NES autonomously!")
                env.close()
                sys.exit(0) # Exits with Code 0 to flag true final master completion

            # Standard 1D RAM input vector formatting
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action, log_prob, value, entropy = model.get_action(obs_tensor, ent_coef=ent_coef)
            last_step_entropy = float(entropy.item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Apply velocity-sensitive protection bounds inside our active loop
            curr_max_x = float(info.get("max_x_pos", info.get("x_pos", 0.0)))
            if steps > 0 and steps % stagnation_frame_budget == 0:
                if prev_x_pos == curr_max_x:
                    reward += zero_velocity_penalty # Impose movement penalty bounds
                prev_x_pos = curr_max_x

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            entropies.append(entropy)

            total_reward += reward
            steps += 1
            obs = next_obs

        returns = []
        discounted_sum = 0.0
        for reward, finished in zip(reversed(rewards), reversed(dones)):
            if finished:
                discounted_sum = 0.0
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        values_tensor = torch.cat(values).squeeze(-1)
        advantages = returns_tensor - values_tensor.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        value_loss = 0.5 * (returns_tensor - values_tensor).pow(2).mean()
        entropy_bonus = torch.stack(entropies).mean()
        loss = policy_loss + value_loss - ent_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] = max(param_group["lr"], lr_floor)

        # Calculate tracking variables
        curr_max_x = float(info.get("max_x_pos", info.get("x_pos", 0.0)))
        rolling_max_x_history.append(curr_max_x)
        rolling_reward_history.append(total_reward)

        # -------------------------------------------------------------
        # ANTI-REWARD HACKING & STAGATION INTERCEPTORS
        # -------------------------------------------------------------
        if episode >= stagnation_min_episodes and len(rolling_max_x_history) >= stagnation_patience_episodes:
            plateau_window = rolling_max_x_history[-stagnation_patience_episodes:]
            if (max(plateau_window) - plateau_window[0]) < 20.0:
                save_training_checkpoint(
                    os.path.join(checkpoint_dir, "model_weights_latest.pt"),
                    episode,
                    curr_max_x,
                    model,
                    optimizer,
                )
                trigger_early_stopping("STAGNATION_PLATEAU", episode, total_reward, curr_max_x, last_step_entropy, active_params, audit_logger, exit_code=2)

        print(f"Episode {episode}/{start_episode + total_episodes - 1} - Reward: {total_reward:.2f} - Steps: {steps} - Max Distance: {curr_max_x:.1f} - End: {info.get('termination_reason', 'unknown')}")

        # Save Best Model Checkpointing
        if curr_max_x > best_max_x_pos:
            best_max_x_pos = curr_max_x
            save_training_checkpoint(os.path.join(checkpoint_dir, "best_ppo_agent_dist.pt"), episode, curr_max_x, model, optimizer)

        if episode % checkpoint_interval == 0:
            checkpoint = os.path.join(checkpoint_dir, f"ppo_agent_ep{episode}.pt")
            save_training_checkpoint(checkpoint, episode, curr_max_x, model, optimizer)
            save_training_checkpoint(os.path.join(checkpoint_dir, "model_weights_latest.pt"), episode, curr_max_x, model, optimizer)

    env.close()

if __name__ == "__main__":
    train_ppo_agent()
