"""
training/dqn_training.py
─────────────────────────────────────────────────────────────
DQN (Deep Q-Network) training for AfAlm CropDroneEnv.
Runs 10 experiments with varied hyperparameters and saves
a results CSV + the best model checkpoint.

Usage
─────
  # Run all 10 experiments automatically (recommended)
  python training/dqn_training.py

  # Run a single experiment by index (0-9)
  python training/dqn_training.py --run 3

  # Quick smoke-test (very short, just checks everything loads)
  python training/dqn_training.py --smoke-test

  # Disable rendering during eval
  python training/dqn_training.py --no-render

M1 Pro notes
─────────────
  Each run ≈ 80 000 timesteps.
  Estimated time per run: 3–5 min → ~35–45 min total for all 10.
  PyTorch will use CPU (SB3 MPS support is still experimental).
  Set OMP_NUM_THREADS=8 before running to maximise M1 efficiency:
      export OMP_NUM_THREADS=8 && python training/dqn_training.py
"""

import os
import sys
import time
import argparse
import csv
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# ── Path fix so we can import from project root ──────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import CropDroneEnv

# ── Directories ───────────────────────────────────────────────
MODEL_DIR  = "models/dqn"
LOG_DIR    = "logs/dqn"
CSV_PATH   = "models/dqn/hyperparameter_results.csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 10 HYPERPARAMETER CONFIGURATIONS
# Each entry maps directly to what the report table needs.
# Axes varied: learning_rate, gamma, buffer_size, batch_size,
#              exploration_fraction, target_update_interval, net_arch
# ─────────────────────────────────────────────────────────────
DQN_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Baseline: moderate lr, standard net, balanced exploration
    {
        "run_id": 1,
        "description": "Baseline — lr=1e-4, γ=0.99, buf=50k, batch=64, expl=0.2",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.20,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "net_arch": [256, 256],
        "learning_starts": 1_000,
    },
    # Run 2 — Higher lr: faster convergence but risk of instability
    {
        "run_id": 2,
        "description": "High LR — lr=5e-4, γ=0.99, buf=50k, batch=64, expl=0.2",
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.20,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "net_arch": [256, 256],
        "learning_starts": 1_000,
    },
    # Run 3 — Very high lr: tests stability ceiling
    {
        "run_id": 3,
        "description": "Very high LR — lr=1e-3, γ=0.99, buf=50k, batch=64, expl=0.2",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.20,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "net_arch": [256, 256],
        "learning_starts": 1_000,
    },
    # Run 4 — Lower gamma: values near-term rewards more (hurts long-horizon planning)
    {
        "run_id": 4,
        "description": "Low γ — lr=1e-4, γ=0.90, buf=50k, batch=64, expl=0.2",
        "learning_rate": 1e-4,
        "gamma": 0.90,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.20,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "net_arch": [256, 256],
        "learning_starts": 1_000,
    },
    # Run 5 — Small buffer: less diverse replay, faster memory, higher bias
    {
        "run_id": 5,
        "description": "Small buffer — lr=1e-4, γ=0.99, buf=10k, batch=64, expl=0.2",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 10_000,
        "batch_size": 64,
        "exploration_fraction": 0.20,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "net_arch": [256, 256],
        "learning_starts": 500,
    },
    # Run 6 — Large batch: more stable gradient estimates
    {
        "run_id": 6,
        "description": "Large batch — lr=1e-4, γ=0.99, buf=50k, batch=128, expl=0.2",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 128,
        "exploration_fraction": 0.20,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "net_arch": [256, 256],
        "learning_starts": 1_000,
    },
    # Run 7 — Long exploration: agent explores more before exploiting
    {
        "run_id": 7,
        "description": "Long exploration — lr=1e-4, γ=0.99, buf=50k, expl=0.40",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.40,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "net_arch": [256, 256],
        "learning_starts": 1_000,
    },
    # Run 8 — Short exploration: exploit earlier (risks local optima)
    {
        "run_id": 8,
        "description": "Short exploration — lr=1e-4, γ=0.99, buf=50k, expl=0.10",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.10,
        "exploration_final_eps": 0.02,
        "target_update_interval": 500,
        "net_arch": [256, 256],
        "learning_starts": 1_000,
    },
    # Run 9 — Smaller network: faster forward pass, less capacity
    {
        "run_id": 9,
        "description": "Small net — lr=1e-4, γ=0.99, buf=50k, arch=[128,128]",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.20,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "net_arch": [128, 128],
        "learning_starts": 1_000,
    },
    # Run 10 — Best guess combo: tuned lr + large buffer + frequent target updates
    {
        "run_id": 10,
        "description": "Tuned combo — lr=2e-4, γ=0.99, buf=100k, batch=128, arch=[256,256,128]",
        "learning_rate": 2e-4,
        "gamma": 0.99,
        "buffer_size": 100_000,
        "batch_size": 128,
        "exploration_fraction": 0.25,
        "exploration_final_eps": 0.04,
        "target_update_interval": 300,
        "net_arch": [256, 256, 128],
        "learning_starts": 2_000,
    },
]

TOTAL_TIMESTEPS = 120_000   # per run — ~5–7 min on M1 Pro (~55–70 min total)
EVAL_EPISODES   = 10        # episodes per evaluation
EVAL_FREQ       = 10_000    # evaluate every N training steps


# ─────────────────────────────────────────────────────────────
class TqdmCallback(BaseCallback):
    """tqdm progress bar that updates every step and shows key training stats."""
    def __init__(self, total_timesteps: int, run_id: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.run_id = run_id
        self.pbar: tqdm = None

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc=f"  DQN Run {self.run_id:>2d}/10",
            unit="step",
            ncols=88,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            colour="green",
        )

    def _on_step(self) -> bool:
        self.pbar.update(1)
        if self.num_timesteps % 5_000 == 0:
            eps = self.model.exploration_rate
            self.pbar.set_postfix(eps=f"{eps:.3f}", refresh=False)
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()


def make_env(seed: int = 0, render_mode=None):
    def _init():
        env = CropDroneEnv(
            grid_size=8,
            max_steps=200,
            max_fuel=100,
            max_payload=3,
            spread_interval=10,
            spread_prob=0.20,
            n_infected=5,
            render_mode=render_mode,
            seed=seed,
        )
        return Monitor(env)
    return _init


def run_dqn_experiment(
    config: Dict[str, Any],
    total_timesteps: int = TOTAL_TIMESTEPS,
    seed: int = 42,
    enable_tensorboard: bool = True,
    save_best: bool = True,
) -> Dict[str, Any]:
    """Train one DQN run and return metrics."""
    run_id = config["run_id"]
    print(f"\n{'='*62}")
    print(f"  DQN Run {run_id}/10")
    print(f"  {config['description']}")
    print(f"{'='*62}")

    run_dir = os.path.join(MODEL_DIR, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    # Training env (no rendering — fast)
    train_env = make_vec_env(make_env(seed=seed), n_envs=1)

    # Eval env
    eval_env = Monitor(CropDroneEnv(
        grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
        spread_interval=10, spread_prob=0.20, n_infected=5,
        render_mode=None, seed=seed + 100,
    ))

    policy_kwargs = {"net_arch": config["net_arch"]}
    tb_log = os.path.join(LOG_DIR, f"run_{run_id:02d}") if enable_tensorboard else None

    model = DQN(
        policy              = "MlpPolicy",
        env                 = train_env,
        learning_rate       = config["learning_rate"],
        buffer_size         = config["buffer_size"],
        learning_starts     = config["learning_starts"],
        batch_size          = config["batch_size"],
        tau                 = 1.0,
        gamma               = config["gamma"],
        train_freq          = 4,
        gradient_steps      = 1,
        target_update_interval = config["target_update_interval"],
        exploration_fraction   = config["exploration_fraction"],
        exploration_initial_eps= 1.0,
        exploration_final_eps  = config["exploration_final_eps"],
        policy_kwargs       = policy_kwargs,
        tensorboard_log     = tb_log,
        verbose             = 0,
        seed                = seed,
        device              = "cpu",   # MPS not fully stable with SB3 on M1
    )

    callbacks = [TqdmCallback(total_timesteps=total_timesteps, run_id=run_id)]

    if save_best:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path = run_dir,
            log_path             = run_dir,
            eval_freq            = EVAL_FREQ,
            n_eval_episodes      = EVAL_EPISODES,
            deterministic        = True,
            verbose              = 0,
        )
        callbacks.append(eval_cb)

    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    train_time = time.time() - t_start

    # Final evaluation
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
    )

    # Save final model
    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)

    # Track best model globally
    best_path = os.path.join(MODEL_DIR, "best_model")
    best_results_path = os.path.join(MODEL_DIR, "best_reward.txt")
    current_best = -np.inf
    if os.path.exists(best_results_path):
        with open(best_results_path) as f:
            current_best = float(f.read().strip())
    if mean_reward > current_best:
        model.save(best_path)
        with open(best_results_path, "w") as f:
            f.write(str(mean_reward))
        print(f"  ★ New best model saved (reward={mean_reward:.2f})")

    result = {
        "run_id":                  run_id,
        "description":             config["description"],
        "learning_rate":           config["learning_rate"],
        "gamma":                   config["gamma"],
        "buffer_size":             config["buffer_size"],
        "batch_size":              config["batch_size"],
        "exploration_fraction":    config["exploration_fraction"],
        "exploration_final_eps":   config["exploration_final_eps"],
        "target_update_interval":  config["target_update_interval"],
        "net_arch":                str(config["net_arch"]),
        "total_timesteps":         total_timesteps,
        "mean_reward":             round(mean_reward, 2),
        "std_reward":              round(std_reward,  2),
        "train_time_min":          round(train_time / 60, 2),
    }

    print(f"\n  ✓ Run {run_id} complete")
    print(f"    Mean reward : {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"    Train time  : {train_time/60:.1f} min")

    train_env.close()
    eval_env.close()
    return result


def save_results_csv(results: List[Dict], path: str = CSV_PATH):
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results table saved → {path}")


def print_summary_table(results: List[Dict]):
    print(f"\n{'='*72}")
    print(f"  DQN HYPERPARAMETER RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Run':>3}  {'LR':>8}  {'γ':>5}  {'Buf':>7}  "
          f"{'Batch':>5}  {'Expl':>5}  "
          f"{'Mean Rew':>10}  {'Std':>8}  {'Time(m)':>7}")
    print(f"  {'─'*68}")
    best_reward = max(r["mean_reward"] for r in results)
    for r in results:
        marker = " ★" if r["mean_reward"] == best_reward else "  "
        print(
            f"  {r['run_id']:>3}{marker}"
            f"  {r['learning_rate']:>8.0e}"
            f"  {r['gamma']:>5.2f}"
            f"  {r['buffer_size']:>7d}"
            f"  {r['batch_size']:>5d}"
            f"  {r['exploration_fraction']:>5.2f}"
            f"  {r['mean_reward']:>10.2f}"
            f"  {r['std_reward']:>8.2f}"
            f"  {r['train_time_min']:>7.1f}"
        )
    print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training — AfAlm CropDrone")
    parser.add_argument("--run",        type=int, default=None,
                        help="Run a single experiment (1-10). Omit to run all.")
    parser.add_argument("--timesteps",  type=int, default=TOTAL_TIMESTEPS,
                        help=f"Timesteps per run (default: {TOTAL_TIMESTEPS})")    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 2000-step test to verify everything loads")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-save-best",   action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        print("\n  Smoke test — 2000 steps, run 1 only")
        result = run_dqn_experiment(
            DQN_CONFIGS[0], total_timesteps=2_000,
            enable_tensorboard=False, save_best=False
        )
        print(f"\n  ✓ Smoke test passed  reward={result['mean_reward']}")
        sys.exit(0)

    timesteps    = args.timesteps
    tb_enabled   = not args.no_tensorboard
    save_best    = not args.no_save_best

    print(f"\n  AfAlm — DQN Training")
    print(f"  Timesteps per run : {timesteps:,}")
    print(f"  TensorBoard       : {tb_enabled}")
    print(f"  Device            : CPU (M1 optimised)")
    print(f"  Start time        : {datetime.now().strftime('%H:%M:%S')}")

    configs_to_run = DQN_CONFIGS
    if args.run is not None:
        idx = args.run - 1
        if not 0 <= idx < len(DQN_CONFIGS):
            print(f"  Error: --run must be 1-{len(DQN_CONFIGS)}")
            sys.exit(1)
        configs_to_run = [DQN_CONFIGS[idx]]

    all_results = []
    for cfg in configs_to_run:
        result = run_dqn_experiment(
            cfg,
            total_timesteps  = timesteps,
            seed             = args.seed,
            enable_tensorboard = tb_enabled,
            save_best        = save_best,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print_summary_table(all_results)
        save_results_csv(all_results)

    print(f"\n  All runs complete. Models saved in: {MODEL_DIR}/")
    print(f"  Best model: {MODEL_DIR}/best_model.zip")
    if tb_enabled:
        print(f"  TensorBoard: tensorboard --logdir {LOG_DIR}")
