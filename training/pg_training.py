"""
training/pg_training.py
─────────────────────────────────────────────────────────────
Policy Gradient training for AfAlm CropDroneEnv.
Implements two algorithms:

  1. PPO (Proximal Policy Optimization) via Stable-Baselines3
  2. REINFORCE (vanilla policy gradient) — custom PyTorch

Both run 10 hyperparameter experiments and save results CSV.

Usage
─────
  # Run all PPO experiments
  python training/pg_training.py --algo ppo

  # Run all REINFORCE experiments
  python training/pg_training.py --algo reinforce

  # Run both sequentially (full assignment requirement)
  python training/pg_training.py --algo all

  # Single run by index (1-10)
  python training/pg_training.py --algo ppo --run 4

  # Quick smoke test
  python training/pg_training.py --algo ppo --smoke-test
  python training/pg_training.py --algo reinforce --smoke-test

M1 Pro notes
─────────────
  PPO   : ~3–5 min per run  → ~35–45 min total
  REIN  : ~3–5 min per run  → ~35–45 min total
  export OMP_NUM_THREADS=8 before running for best performance
"""

import os
import sys
import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from environment.custom_env import CropDroneEnv

# ── Directories ───────────────────────────────────────────────
PPO_MODEL_DIR     = "models/pg"
REINFORCE_DIR     = "models/pg"
LOG_DIR_PPO       = "logs/ppo"
LOG_DIR_REINFORCE = "logs/reinforce"
for d in [PPO_MODEL_DIR, LOG_DIR_PPO, LOG_DIR_REINFORCE]:
    os.makedirs(d, exist_ok=True)

PPO_TOTAL_TIMESTEPS       = 150_000   # per run — ~5–8 min on M1 Pro
REINFORCE_TOTAL_EPISODES  = 700       # per run — ~5–7 min on M1 Pro

EVAL_EPISODES = 10
EVAL_FREQ     = 15_000


# ═════════════════════════════════════════════════════════════
# SECTION 1 — PPO HYPERPARAMETER CONFIGURATIONS
# ═════════════════════════════════════════════════════════════
# Axes varied: lr, n_steps, batch_size, n_epochs, gamma,
#              gae_lambda, ent_coef, clip_range
# The environment has episodes of up to 200 steps, sparse-ish
# rewards, and partial observability — entropy is important.
PPO_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Baseline: SB3 defaults adapted for this env
    {
        "run_id": 1,
        "description": "Baseline — lr=3e-4, n_steps=2048, batch=64, ent=0.01",
        "learning_rate":  3e-4,
        "n_steps":        2048,
        "batch_size":     64,
        "n_epochs":       10,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.01,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 2 — Higher entropy: more exploration of spray/scan actions
    {
        "run_id": 2,
        "description": "High entropy — lr=3e-4, ent=0.05, n_steps=2048",
        "learning_rate":  3e-4,
        "n_steps":        2048,
        "batch_size":     64,
        "n_epochs":       10,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.05,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 3 — Very high entropy: forces wide action exploration
    {
        "run_id": 3,
        "description": "Very high entropy — lr=3e-4, ent=0.10, n_steps=2048",
        "learning_rate":  3e-4,
        "n_steps":        2048,
        "batch_size":     64,
        "n_epochs":       10,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.10,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 4 — Low lr: slow but stable convergence
    {
        "run_id": 4,
        "description": "Low LR — lr=1e-4, ent=0.01, n_steps=2048",
        "learning_rate":  1e-4,
        "n_steps":        2048,
        "batch_size":     64,
        "n_epochs":       10,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.01,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 5 — Smaller rollout: more frequent updates
    {
        "run_id": 5,
        "description": "Short rollout — lr=3e-4, n_steps=512, batch=64",
        "learning_rate":  3e-4,
        "n_steps":        512,
        "batch_size":     64,
        "n_epochs":       10,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.01,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 6 — Longer rollout: captures full episode trajectory
    {
        "run_id": 6,
        "description": "Long rollout — lr=3e-4, n_steps=4096, batch=128",
        "learning_rate":  3e-4,
        "n_steps":        4096,
        "batch_size":     128,
        "n_epochs":       10,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.01,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 7 — Tighter clip range: more conservative policy updates
    {
        "run_id": 7,
        "description": "Tight clip — lr=3e-4, clip=0.1, ent=0.01, n_steps=2048",
        "learning_rate":  3e-4,
        "n_steps":        2048,
        "batch_size":     64,
        "n_epochs":       10,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.01,
        "clip_range":     0.1,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 8 — Lower gamma: discounts future more (tests short-horizon behaviour)
    {
        "run_id": 8,
        "description": "Low gamma — lr=3e-4, γ=0.95, gae=0.90, ent=0.01",
        "learning_rate":  3e-4,
        "n_steps":        2048,
        "batch_size":     64,
        "n_epochs":       10,
        "gamma":          0.95,
        "gae_lambda":     0.90,
        "ent_coef":       0.01,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 9 — More epochs per update: extracts more from each rollout
    {
        "run_id": 9,
        "description": "More epochs — lr=3e-4, n_epochs=20, ent=0.01",
        "learning_rate":  3e-4,
        "n_steps":        2048,
        "batch_size":     64,
        "n_epochs":       20,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.01,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 10 — Best guess combo: higher ent + longer rollout + tuned lr
    {
        "run_id": 10,
        "description": "Tuned combo — lr=2e-4, n_steps=4096, ent=0.03, epochs=15",
        "learning_rate":  2e-4,
        "n_steps":        4096,
        "batch_size":     128,
        "n_epochs":       15,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.03,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
    },
]


# ═════════════════════════════════════════════════════════════
# SECTION 2 — REINFORCE HYPERPARAMETER CONFIGURATIONS
# ═════════════════════════════════════════════════════════════
# Axes varied: lr, gamma, hidden_dim, use_baseline (subtract
#              mean return), entropy_coef, normalize_returns
REINFORCE_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Baseline REINFORCE: no baseline subtraction
    {
        "run_id": 1,
        "description": "Baseline REINFORCE — lr=1e-3, γ=0.99, hidden=256, no baseline",
        "learning_rate":      1e-3,
        "gamma":              0.99,
        "hidden_dim":         256,
        "use_baseline":       False,
        "entropy_coef":       0.01,
        "normalize_returns":  False,
    },
    # Run 2 — With baseline (mean return subtracted): reduces variance
    {
        "run_id": 2,
        "description": "With baseline — lr=1e-3, γ=0.99, hidden=256, baseline=True",
        "learning_rate":      1e-3,
        "gamma":              0.99,
        "hidden_dim":         256,
        "use_baseline":       True,
        "entropy_coef":       0.01,
        "normalize_returns":  False,
    },
    # Run 3 — Normalised returns: standardises advantages per episode
    {
        "run_id": 3,
        "description": "Normalised returns — lr=1e-3, γ=0.99, hidden=256, norm=True",
        "learning_rate":      1e-3,
        "gamma":              0.99,
        "hidden_dim":         256,
        "use_baseline":       True,
        "entropy_coef":       0.01,
        "normalize_returns":  True,
    },
    # Run 4 — Higher entropy: encourages scan + mixed treatment actions
    {
        "run_id": 4,
        "description": "High entropy — lr=1e-3, γ=0.99, ent=0.05, norm=True",
        "learning_rate":      1e-3,
        "gamma":              0.99,
        "hidden_dim":         256,
        "use_baseline":       True,
        "entropy_coef":       0.05,
        "normalize_returns":  True,
    },
    # Run 5 — Lower lr: slower but more stable policy gradient steps
    {
        "run_id": 5,
        "description": "Low LR — lr=5e-4, γ=0.99, hidden=256, norm=True",
        "learning_rate":      5e-4,
        "gamma":              0.99,
        "hidden_dim":         256,
        "use_baseline":       True,
        "entropy_coef":       0.01,
        "normalize_returns":  True,
    },
    # Run 6 — Higher lr: tests gradient explosion boundary
    {
        "run_id": 6,
        "description": "High LR — lr=3e-3, γ=0.99, hidden=256, norm=True",
        "learning_rate":      3e-3,
        "gamma":              0.99,
        "hidden_dim":         256,
        "use_baseline":       True,
        "entropy_coef":       0.01,
        "normalize_returns":  True,
    },
    # Run 7 — Smaller network: faster, less capacity
    {
        "run_id": 7,
        "description": "Small net — lr=1e-3, γ=0.99, hidden=128, norm=True",
        "learning_rate":      1e-3,
        "gamma":              0.99,
        "hidden_dim":         128,
        "use_baseline":       True,
        "entropy_coef":       0.01,
        "normalize_returns":  True,
    },
    # Run 8 — Larger network: more policy capacity
    {
        "run_id": 8,
        "description": "Large net — lr=1e-3, γ=0.99, hidden=512, norm=True",
        "learning_rate":      1e-3,
        "gamma":              0.99,
        "hidden_dim":         512,
        "use_baseline":       True,
        "entropy_coef":       0.01,
        "normalize_returns":  True,
    },
    # Run 9 — Lower gamma: short-horizon discounting (penalises long detours)
    {
        "run_id": 9,
        "description": "Low gamma — lr=1e-3, γ=0.95, hidden=256, norm=True",
        "learning_rate":      1e-3,
        "gamma":              0.95,
        "hidden_dim":         256,
        "use_baseline":       True,
        "entropy_coef":       0.01,
        "normalize_returns":  True,
    },
    # Run 10 — Best guess: tuned lr + large net + entropy + baseline
    {
        "run_id": 10,
        "description": "Tuned combo — lr=8e-4, γ=0.99, hidden=256, ent=0.03, norm=True",
        "learning_rate":      8e-4,
        "gamma":              0.99,
        "hidden_dim":         256,
        "use_baseline":       True,
        "entropy_coef":       0.03,
        "normalize_returns":  True,
    },
]


# ═════════════════════════════════════════════════════════════
# SECTION 3 — PPO TRAINING
# ═════════════════════════════════════════════════════════════
class TqdmPPOCallback(BaseCallback):
    """tqdm progress bar for PPO with entropy and value loss readout."""
    def __init__(self, total_timesteps: int, run_id: int, algo: str = "PPO"):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.run_id = run_id
        self.algo = algo
        self.pbar: tqdm = None

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc=f"  {self.algo} Run {self.run_id:>2d}/10",
            unit="step",
            ncols=88,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            colour="cyan",
        )

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()


def make_env_fn(seed=0):
    def _init():
        env = CropDroneEnv(
            grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
            spread_interval=10, spread_prob=0.20, n_infected=5,
            render_mode=None, seed=seed,
        )
        return Monitor(env)
    return _init


def run_ppo_experiment(
    config: Dict[str, Any],
    total_timesteps: int = PPO_TOTAL_TIMESTEPS,
    seed: int = 42,
    enable_tensorboard: bool = True,
    save_best: bool = True,
) -> Dict[str, Any]:
    run_id = config["run_id"]
    print(f"\n{'='*62}")
    print(f"  PPO Run {run_id}/10")
    print(f"  {config['description']}")
    print(f"{'='*62}")

    run_dir = os.path.join(PPO_MODEL_DIR, f"ppo_run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    train_env = make_vec_env(make_env_fn(seed=seed), n_envs=1)
    eval_env  = Monitor(CropDroneEnv(
        grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
        spread_interval=10, spread_prob=0.20, n_infected=5,
        render_mode=None, seed=seed + 200,
    ))

    tb_log = os.path.join(LOG_DIR_PPO, f"run_{run_id:02d}") if enable_tensorboard else None

    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = config["learning_rate"],
        n_steps         = config["n_steps"],
        batch_size      = config["batch_size"],
        n_epochs        = config["n_epochs"],
        gamma           = config["gamma"],
        gae_lambda      = config["gae_lambda"],
        ent_coef        = config["ent_coef"],
        clip_range      = config["clip_range"],
        vf_coef         = config["vf_coef"],
        policy_kwargs   = {"net_arch": config["net_arch"]},
        tensorboard_log = tb_log,
        verbose         = 0,
        seed            = seed,
        device          = "cpu",
    )

    callbacks = [TqdmPPOCallback(total_timesteps=total_timesteps, run_id=run_id, algo="PPO")]
    if save_best:
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path = run_dir,
            log_path             = run_dir,
            eval_freq            = EVAL_FREQ,
            n_eval_episodes      = EVAL_EPISODES,
            deterministic        = True,
            verbose              = 0,
        ))

    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    train_time = time.time() - t_start

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
    )

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)

    # Track global best PPO
    best_path  = os.path.join(PPO_MODEL_DIR, "ppo_best")
    best_txt   = os.path.join(PPO_MODEL_DIR, "ppo_best_reward.txt")
    current_best = -np.inf
    if os.path.exists(best_txt):
        with open(best_txt) as f:
            current_best = float(f.read().strip())
    if mean_reward > current_best:
        model.save(best_path)
        with open(best_txt, "w") as f:
            f.write(str(mean_reward))
        print(f"  ★ New best PPO model (reward={mean_reward:.2f})")

    result = {
        "run_id":          run_id,
        "description":     config["description"],
        "learning_rate":   config["learning_rate"],
        "n_steps":         config["n_steps"],
        "batch_size":      config["batch_size"],
        "n_epochs":        config["n_epochs"],
        "gamma":           config["gamma"],
        "gae_lambda":      config["gae_lambda"],
        "ent_coef":        config["ent_coef"],
        "clip_range":      config["clip_range"],
        "total_timesteps": total_timesteps,
        "mean_reward":     round(mean_reward, 2),
        "std_reward":      round(std_reward,  2),
        "train_time_min":  round(train_time / 60, 2),
    }

    print(f"\n  ✓ PPO Run {run_id} complete  "
          f"reward={mean_reward:.2f}±{std_reward:.2f}  "
          f"time={train_time/60:.1f}m")

    train_env.close()
    eval_env.close()
    return result


# ═════════════════════════════════════════════════════════════
# SECTION 4 — REINFORCE (custom PyTorch implementation)
# ═════════════════════════════════════════════════════════════
class PolicyNetwork(nn.Module):
    """Simple MLP policy for REINFORCE."""
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

    def get_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.forward(obs_t)
        dist  = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns G_t for each timestep."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.FloatTensor(returns)


def run_reinforce_experiment(
    config: Dict[str, Any],
    total_episodes: int = REINFORCE_TOTAL_EPISODES,
    seed: int = 42,
    save_best: bool = True,
) -> Dict[str, Any]:
    run_id = config["run_id"]
    print(f"\n{'='*62}")
    print(f"  REINFORCE Run {run_id}/10")
    print(f"  {config['description']}")
    print(f"{'='*62}")

    run_dir = os.path.join(REINFORCE_DIR, f"reinforce_run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CropDroneEnv(
        grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
        spread_interval=10, spread_prob=0.20, n_infected=5,
        render_mode=None, seed=seed,
    )

    obs_dim   = env.observation_space.shape[0]   # 72
    n_actions = env.action_space.n               # 8
    hidden    = config["hidden_dim"]

    policy    = PolicyNetwork(obs_dim, n_actions, hidden)
    optimizer = optim.Adam(policy.parameters(), lr=config["learning_rate"])

    episode_rewards = []
    best_mean = -np.inf
    t_start   = time.time()

    pbar = tqdm(
        range(total_episodes),
        desc=f"  REINFORCE Run {run_id:>2d}/10",
        unit="ep",
        ncols=88,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        colour="magenta",
    )

    for ep in pbar:
        obs, _ = env.reset()
        log_probs, rewards, entropies = [], [], []
        done = False

        # ── Collect one episode ───────────────────
        while not done:
            action, log_prob, entropy = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            done = terminated or truncated

        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)

        # ── Compute returns ───────────────────────
        returns = compute_returns(rewards, config["gamma"])

        if config["normalize_returns"] and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        if config["use_baseline"]:
            returns = returns - returns.mean()

        # ── Policy gradient loss ──────────────────
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        policy_loss  = -(log_probs_t * returns).mean()
        entropy_loss = -config["entropy_coef"] * entropies_t.mean()
        loss         = policy_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        # ── tqdm postfix update every 50 episodes ─
        if (ep + 1) % 50 == 0:
            recent_mean = float(np.mean(episode_rewards[-50:]))
            pbar.set_postfix(
                reward=f"{recent_mean:.1f}",
                loss=f"{loss.item():.3f}",
                refresh=False,
            )
            # Save best checkpoint
            if save_best and recent_mean > best_mean:
                best_mean = recent_mean
                torch.save(
                    policy.state_dict(),
                    os.path.join(run_dir, "best_policy.pt"),
                )

    pbar.close()

    env.close()
    train_time = time.time() - t_start

    # ── Final evaluation (greedy) ─────────────────
    eval_env = CropDroneEnv(
        grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
        spread_interval=10, spread_prob=0.20, n_infected=5,
        render_mode=None, seed=seed + 300,
    )
    eval_rewards = []
    policy.eval()
    with torch.no_grad():
        for _ in range(EVAL_EPISODES):
            obs, _ = eval_env.reset()
            ep_r, done = 0.0, False
            while not done:
                obs_t  = torch.FloatTensor(obs).unsqueeze(0)
                probs  = policy(obs_t)
                action = probs.argmax(dim=-1).item()   # greedy
                obs, r, terminated, truncated, _ = eval_env.step(action)
                ep_r += r
                done = terminated or truncated
            eval_rewards.append(ep_r)
    eval_env.close()

    mean_reward = float(np.mean(eval_rewards))
    std_reward  = float(np.std(eval_rewards))

    # Save final weights
    torch.save(policy.state_dict(), os.path.join(run_dir, "final_policy.pt"))

    # Track global best REINFORCE
    best_global_path = os.path.join(REINFORCE_DIR, "reinforce_best.pt")
    best_txt         = os.path.join(REINFORCE_DIR, "reinforce_best_reward.txt")
    current_best = -np.inf
    if os.path.exists(best_txt):
        with open(best_txt) as f:
            current_best = float(f.read().strip())
    if mean_reward > current_best:
        torch.save(policy.state_dict(), best_global_path)
        with open(best_txt, "w") as f:
            f.write(str(mean_reward))
        print(f"  ★ New best REINFORCE model (reward={mean_reward:.2f})")

    result = {
        "run_id":             run_id,
        "description":        config["description"],
        "learning_rate":      config["learning_rate"],
        "gamma":              config["gamma"],
        "hidden_dim":         config["hidden_dim"],
        "use_baseline":       config["use_baseline"],
        "entropy_coef":       config["entropy_coef"],
        "normalize_returns":  config["normalize_returns"],
        "total_episodes":     total_episodes,
        "mean_reward":        round(mean_reward, 2),
        "std_reward":         round(std_reward,  2),
        "train_time_min":     round(train_time / 60, 2),
    }

    print(f"\n  ✓ REINFORCE Run {run_id} complete  "
          f"reward={mean_reward:.2f}±{std_reward:.2f}  "
          f"time={train_time/60:.1f}m")
    return result


# ═════════════════════════════════════════════════════════════
# SECTION 5 — UTILITIES
# ═════════════════════════════════════════════════════════════
def save_csv(results: List[Dict], path: str):
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved → {path}")


def print_table(results: List[Dict], algo: str):
    print(f"\n{'='*70}")
    print(f"  {algo.upper()} HYPERPARAMETER RESULTS")
    print(f"{'='*70}")
    best = max(r["mean_reward"] for r in results)
    for r in results:
        m = " ★" if r["mean_reward"] == best else "  "
        print(f"  Run {r['run_id']:>2d}{m}  "
              f"reward={r['mean_reward']:>8.2f}±{r['std_reward']:<6.2f}  "
              f"time={r['train_time_min']:.1f}m  |  {r['description']}")
    print(f"{'='*70}\n")


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy Gradient Training — AfAlm")
    parser.add_argument("--algo",         choices=["ppo", "reinforce", "all"],
                        default="all",    help="Which algorithm to train")
    parser.add_argument("--run",          type=int, default=None,
                        help="Single run index (1-10). Omit to run all.")
    parser.add_argument("--ppo-steps",    type=int, default=PPO_TOTAL_TIMESTEPS)
    parser.add_argument("--rein-eps",     type=int, default=REINFORCE_TOTAL_EPISODES)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--smoke-test",   action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-save-best",   action="store_true")
    args = parser.parse_args()

    tb         = not args.no_tensorboard
    save_best  = not args.no_save_best
    smoke      = args.smoke_test

    print(f"\n  AfAlm — Policy Gradient Training")
    print(f"  Algorithm  : {args.algo.upper()}")
    print(f"  Device     : CPU (M1 optimised)")
    print(f"  Start time : {datetime.now().strftime('%H:%M:%S')}")

    def pick_configs(all_cfgs):
        if args.run is not None:
            idx = args.run - 1
            assert 0 <= idx < len(all_cfgs), f"--run must be 1-{len(all_cfgs)}"
            return [all_cfgs[idx]]
        return all_cfgs

    # ── PPO ───────────────────────────────────────
    if args.algo in ("ppo", "all"):
        ppo_steps = 2_000 if smoke else args.ppo_steps
        cfgs = pick_configs(PPO_CONFIGS)
        ppo_results = []
        for cfg in cfgs:
            r = run_ppo_experiment(cfg, total_timesteps=ppo_steps,
                                   seed=args.seed, enable_tensorboard=tb,
                                   save_best=save_best)
            ppo_results.append(r)
        if len(ppo_results) > 1:
            print_table(ppo_results, "PPO")
            save_csv(ppo_results, os.path.join(PPO_MODEL_DIR, "ppo_results.csv"))
        if tb:
            print(f"  TensorBoard: tensorboard --logdir {LOG_DIR_PPO}")

    # ── REINFORCE ─────────────────────────────────
    if args.algo in ("reinforce", "all"):
        rein_eps = 30 if smoke else args.rein_eps
        cfgs = pick_configs(REINFORCE_CONFIGS)
        rein_results = []
        for cfg in cfgs:
            r = run_reinforce_experiment(cfg, total_episodes=rein_eps,
                                         seed=args.seed, save_best=save_best)
            rein_results.append(r)
        if len(rein_results) > 1:
            print_table(rein_results, "REINFORCE")
            save_csv(rein_results, os.path.join(REINFORCE_DIR, "reinforce_results.csv"))

    print(f"\n  Done. Models saved in: {PPO_MODEL_DIR}/")
