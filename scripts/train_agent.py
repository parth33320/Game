import sys
import os
import glob
import time
import json
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
    for candidate in (
        RESUME_TARGET_FILE,
        os.path.join(checkpoint_dir, "model_weights_latest.pt"),
        os.path.join(checkpoint_dir, "best_ppo_agent_dist.pt"),
        default_target,
    ):
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(checkpoint_dir):
        ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if ckpts:
            ckpts.sort(key=os.path.getmtime, reverse=True)
            return ckpts[0]
    return default_target

def trigger_early_stopping(failure_reason: str, episode: int, total_reward: float, max_x_pos: float, entropy_val: float, active_params: dict, audit_logger: AuditLogger, exit_code: int = 42):
    print(f"\n=======================================================")
    print(f"EARLY STOPPING TRIGGERED: Failure Reason = '{failure_reason}'")
    print(f"Episode: {episode} | Reward: {total_reward:.2f} | Max X: {max_x_pos:.1f} | Entropy: {entropy_val:.4f}")
    print(f"=======================================================\n")
    audit_payload = {
        "status": "EARLY_STOPPING",
        "failure_reason": failure_reason,
        "episode": episode,
        "total_reward": total_reward,
        "max_x_pos": max_x_pos,
        "entropy": entropy_val,
        "active_params": active_params,
        "timestamp": time.time()
    }
    with open(TRAINING_AUDIT_FILE, "w") as f:
        json.dump(audit_payload, f, indent=2)
    audit_logger.log_event("early_stopping_triggered", audit_payload)
    sys.exit(exit_code)

def save_training_checkpoint(path: str, episode: int, max_x_pos: float, model: ActorCriticPPO, optimizer: optim.Optimizer):
    torch.save({
        "episode": episode,
        "max_x_pos": max_x_pos,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)

def collect_rollout_worker(args):
    worker_id, episode, model_state, active_params = args
    torch.set_num_threads(1)
    env = HeadlessRetroEnv(obs_type="ram", use_retro=True, reward_params=active_params)
    model = ActorCriticPPO(input_dim=15, num_actions=len(env.ACTION_NAMES), is_mlp=True)
    model.load_state_dict(model_state)
    model.eval()

    observations = []
    actions = []
    rewards = []
    obs, info = env.reset(seed=episode * 100 + worker_id)
    done = False
    max_x_pos = 0.0
    termination_reason = "unknown"

    with torch.no_grad():
        while not done:
            observations.append(obs.copy())
            action, _, _, _ = model.get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            done = terminated or truncated
            max_x_pos = max(max_x_pos, float(info.get("max_x_pos", 0.0)))
            termination_reason = info.get("termination_reason", termination_reason)

    env.close()
    return observations, actions, rewards, max_x_pos, termination_reason

def train_parallel_ppo_agent(checkpoint_dir: str, params_file: str, active_params: dict, worker_count: int):
    os.makedirs(checkpoint_dir, exist_ok=True)
    initial_lr = float(active_params.get("initial_lr", 1.5e-4))
    lr_floor = float(active_params.get("lr_floor", 3e-5))
    ent_coef = min(max(0.0, float(active_params.get("ent_coef", 0.05))),
                   max(0.0, float(active_params.get("max_ent_coef", 0.08))))
    gamma = float(active_params.get("gamma", 0.99))
    total_episodes = int(active_params.get("total_episodes", 100000))
    checkpoint_interval = int(active_params.get("checkpoint_interval", 25))
    checkpoint = find_target_checkpoint(checkpoint_dir)

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

    context = mp.get_context("spawn")
    print(f"Starting {worker_count} parallel Simons from episode {start_episode} (best distance: {best_max_x_pos:.1f})")
    histories = []
    audit_logger = AuditLogger("training_audit.jsonl")

    with context.Pool(worker_count) as pool:
        for batch_start in range(start_episode, start_episode + total_episodes, worker_count):
            batch = [(index, batch_start + index, {key: value.cpu() for key, value in model.state_dict().items()}, active_params)
                     for index in range(worker_count)]
            results = pool.map(collect_rollout_worker, batch)
            all_observations, all_actions, all_rewards = [], [], []
            batch_max = 0.0
            for observations, actions, rewards, max_x_pos, termination_reason in results:
                all_observations.extend(observations)
                all_actions.extend(actions)
                all_rewards.extend(rewards)
                batch_max = max(batch_max, max_x_pos)
                histories.append(max_x_pos)

            obs_tensor = torch.as_tensor(np.asarray(all_observations), dtype=torch.float32)
            action_tensor = torch.as_tensor(all_actions, dtype=torch.long)
            reward_tensor = torch.as_tensor(all_rewards, dtype=torch.float32)
            returns = []
            offset = 0
            for observations, _, rewards, _, _ in results:
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
            advantages = returns_tensor - values.squeeze(-1).detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss = (-(log_probs * advantages).mean()
                    + 0.5 * (returns_tensor - values.squeeze(-1)).pow(2).mean()
                    - ent_coef * distribution.entropy().mean())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(param_group["lr"], lr_floor)

            episode = min(batch_start + worker_count - 1, start_episode + total_episodes - 1)
            print(f"Episodes {batch_start}-{episode} | batch max distance: {batch_max:.1f}")
            if batch_max > best_max_x_pos:
                best_max_x_pos = batch_max
                save_training_checkpoint(os.path.join(checkpoint_dir, "best_ppo_agent_dist.pt"), episode, batch_max, model, optimizer)
            if episode % checkpoint_interval == 0 or batch_start == start_episode:
                save_training_checkpoint(os.path.join(checkpoint_dir, "model_weights_latest.pt"), episode, batch_max, model, optimizer)

            if len(histories) >= 2000 and episode >= int(active_params.get("stagnation_min_episodes", 1000)):
                if max(histories[-2000:]) - histories[-2000] < 20.0:
                    trigger_early_stopping("STAGNATION_PLATEAU", episode, float(reward_tensor.sum()), batch_max, 0.0, active_params, audit_logger, exit_code=2)

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
