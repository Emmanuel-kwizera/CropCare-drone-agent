"""
training/improved/improved_dqn.py
─────────────────────────────────────────────────────────────
Improved DQN training based on analysis of initial results.

Key changes from initial runs:
  1. 400k timesteps (vs 120k) — main bottleneck was under-training
  2. learning_starts=8000 — let buffer fill before learning
  3. gradient_steps=4, train_freq=1 — learn more per env step
  4. Larger replay buffer (100k–200k)
  5. Lower final epsilon (0.02) — more exploitation at end
  6. 5 focused runs targeting the most impactful axes

Usage:
    python training/improved/improved_dqn.py
    python training/improved/improved_dqn.py --run 2
    python training/improved/improved_dqn.py --smoke-test
"""

import os, sys, csv, time, argparse
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from environment.custom_env import CropDroneEnv

MODEL_DIR = "models/improved/dqn"
LOG_DIR   = "logs/improved/dqn"
CSV_PATH  = "models/improved/dqn/improved_dqn_results.csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

TOTAL_TIMESTEPS = 400_000   
EVAL_EPISODES   = 10
EVAL_FREQ       = 20_000

# ─────────────────────────────────────────────────────────────
# 5 FOCUSED CONFIGS — each targets one specific insight
# from the initial poor results
# ─────────────────────────────────────────────────────────────
IMPROVED_DQN_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Conservative warmup
    # Key insight: initial runs never warmed up the buffer properly.
    # 8000 learning_starts + 2 gradient steps gives stable early learning.
    {
        "run_id": 1,
        "description": "Conservative warmup — lr=1e-4, buf=100k, starts=8k, grad_steps=2",
        "learning_rate":            1e-4,
        "buffer_size":              100_000,
        "learning_starts":          8_000,
        "batch_size":               64,
        "gamma":                    0.99,
        "exploration_fraction":     0.30,
        "exploration_final_eps":    0.02,
        "target_update_interval":   500,
        "gradient_steps":           2,
        "train_freq":               4,
        "net_arch":                 [256, 256],
    },
    # Run 2 — Aggressive learning (more gradient steps per env step)
    # Key insight: DQN only updates once per 4 env steps by default.
    # gradient_steps=4 + train_freq=1 means 4x more learning per step.
    {
        "run_id": 2,
        "description": "Aggressive learning — lr=1e-4, grad_steps=4, train_freq=1, buf=100k",
        "learning_rate":            1e-4,
        "buffer_size":              100_000,
        "learning_starts":          5_000,
        "batch_size":               64,
        "gamma":                    0.99,
        "exploration_fraction":     0.25,
        "exploration_final_eps":    0.02,
        "target_update_interval":   300,
        "gradient_steps":           4,
        "train_freq":               1,
        "net_arch":                 [256, 256],
    },
    # Run 3 — Longer exploration phase
    # Key insight: with partial observability the agent needs longer
    # to discover the scan-then-treat sequence before exploiting.
    {
        "run_id": 3,
        "description": "Long exploration — lr=1e-4, expl=0.45, buf=100k, starts=8k",
        "learning_rate":            1e-4,
        "buffer_size":              100_000,
        "learning_starts":          8_000,
        "batch_size":               64,
        "gamma":                    0.99,
        "exploration_fraction":     0.45,
        "exploration_final_eps":    0.03,
        "target_update_interval":   500,
        "gradient_steps":           2,
        "train_freq":               4,
        "net_arch":                 [256, 256],
    },
    # Run 4 — Large buffer + very slow start
    # Key insight: bigger buffer = more diverse experience = better
    # generalisation. Slow start prevents learning from garbage data.
    {
        "run_id": 4,
        "description": "Large buffer — lr=5e-5, buf=200k, starts=15k, grad_steps=2",
        "learning_rate":            5e-5,
        "buffer_size":              200_000,
        "learning_starts":          15_000,
        "batch_size":               64,
        "gamma":                    0.99,
        "exploration_fraction":     0.35,
        "exploration_final_eps":    0.02,
        "target_update_interval":   600,
        "gradient_steps":           2,
        "train_freq":               4,
        "net_arch":                 [256, 256],
    },
    # Run 5 — Best guess combo
    # Combines: moderate lr + big buffer + aggressive learning + small batch
    # Small batch (32) gives noisier but more frequent gradient updates.
    {
        "run_id": 5,
        "description": "Best combo — lr=1e-4, buf=100k, batch=32, grad_steps=4, expl=0.30",
        "learning_rate":            1e-4,
        "buffer_size":              100_000,
        "learning_starts":          5_000,
        "batch_size":               32,
        "gamma":                    0.99,
        "exploration_fraction":     0.30,
        "exploration_final_eps":    0.02,
        "target_update_interval":   400,
        "gradient_steps":           4,
        "train_freq":               1,
        "net_arch":                 [256, 256],
    },
]


class TqdmCallback(BaseCallback):
    def __init__(self, total: int, run_id: int):
        super().__init__()
        self.total = total
        self.run_id = run_id
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total,
            desc=f"  Improved DQN Run {self.run_id}/5",
            unit="step", ncols=88, colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def _on_step(self):
        self.pbar.update(1)
        if self.num_timesteps % 20_000 == 0:
            self.pbar.set_postfix(eps=f"{self.model.exploration_rate:.3f}", refresh=False)
        return True

    def _on_training_end(self):
        if self.pbar: self.pbar.close()


def make_env_fn(seed=0):
    def _init():
        env = CropDroneEnv(
            grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
            spread_interval=10, spread_prob=0.20, n_infected=5,
            render_mode=None, seed=seed,
        )
        return Monitor(env)
    return _init


def run_experiment(config: Dict, timesteps: int, seed: int) -> Dict:
    run_id = config["run_id"]
    print(f"\n{'='*60}")
    print(f"  Improved DQN — Run {run_id}/5")
    print(f"  {config['description']}")
    print(f"{'='*60}")

    run_dir = os.path.join(MODEL_DIR, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    train_env = make_vec_env(make_env_fn(seed=seed), n_envs=1)
    eval_env  = Monitor(CropDroneEnv(
        grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
        spread_interval=10, spread_prob=0.20, n_infected=5,
        render_mode=None, seed=seed+100,
    ))

    model = DQN(
        policy                  = "MlpPolicy",
        env                     = train_env,
        learning_rate           = config["learning_rate"],
        buffer_size             = config["buffer_size"],
        learning_starts         = config["learning_starts"],
        batch_size              = config["batch_size"],
        gamma                   = config["gamma"],
        train_freq              = config["train_freq"],
        gradient_steps          = config["gradient_steps"],
        target_update_interval  = config["target_update_interval"],
        exploration_fraction    = config["exploration_fraction"],
        exploration_initial_eps = 1.0,
        exploration_final_eps   = config["exploration_final_eps"],
        policy_kwargs           = {"net_arch": config["net_arch"]},
        tensorboard_log         = os.path.join(LOG_DIR, f"run_{run_id:02d}"),
        verbose                 = 0,
        seed                    = seed,
        device                  = "cpu",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = run_dir,
        log_path             = run_dir,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,
        verbose              = 0,
    )

    t0 = time.time()
    model.learn(
        total_timesteps = timesteps,
        callback        = [TqdmCallback(timesteps, run_id), eval_cb],
        progress_bar    = False,
    )
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True)
    model.save(os.path.join(run_dir, "final_model"))

    # Track global best
    best_path = os.path.join(MODEL_DIR, "best_model")
    best_txt  = os.path.join(MODEL_DIR, "best_reward.txt")
    current_best = -np.inf
    if os.path.exists(best_txt):
        with open(best_txt) as f:
            try: current_best = float(f.read())
            except: pass
    if mean_r > current_best:
        model.save(best_path)
        with open(best_txt, "w") as f: f.write(str(mean_r))
        print(f"  ★ New best DQN (reward={mean_r:.2f})")

    print(f"\n  ✓ Run {run_id} — reward={mean_r:.2f}±{std_r:.2f}  time={elapsed/60:.1f}m")
    train_env.close(); eval_env.close()

    return {
        "run_id":                 run_id,
        "description":            config["description"],
        "learning_rate":          config["learning_rate"],
        "buffer_size":            config["buffer_size"],
        "learning_starts":        config["learning_starts"],
        "batch_size":             config["batch_size"],
        "gamma":                  config["gamma"],
        "exploration_fraction":   config["exploration_fraction"],
        "exploration_final_eps":  config["exploration_final_eps"],
        "gradient_steps":         config["gradient_steps"],
        "train_freq":             config["train_freq"],
        "total_timesteps":        timesteps,
        "mean_reward":            round(mean_r, 2),
        "std_reward":             round(std_r, 2),
        "train_time_min":         round(elapsed / 60, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",        type=int, default=None)
    parser.add_argument("--timesteps",  type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        run_experiment(IMPROVED_DQN_CONFIGS[0], timesteps=3_000, seed=42)
        print("  ✓ Smoke test passed"); sys.exit(0)

    configs = IMPROVED_DQN_CONFIGS if args.run is None else [IMPROVED_DQN_CONFIGS[args.run - 1]]
    results = []

    print(f"\n  Improved DQN Training  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Timesteps per run: {args.timesteps:,}  |  Device: CPU (M1)")

    for cfg in configs:
        results.append(run_experiment(cfg, args.timesteps, args.seed))

    if len(results) > 1:
        print(f"\n{'='*60}  IMPROVED DQN SUMMARY")
        best = max(r["mean_reward"] for r in results)
        for r in results:
            m = " ★" if r["mean_reward"] == best else "  "
            print(f"  Run {r['run_id']}{m}  reward={r['mean_reward']:>8.2f}±{r['std_reward']:<6.2f}  {r['description']}")
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)
        print(f"\n  Results → {CSV_PATH}")
        print(f"  Best model → {MODEL_DIR}/best_model.zip")
        print(f"  TensorBoard → tensorboard --logdir {LOG_DIR}")
