"""
training/improved/improved_ppo_v2.py
─────────────────────────────────────────────────────────────
PPO v2 — targets -200 to -100 reward range.

Root cause analysis of v1 failure:
  - All runs: identical std_reward=177.62 → policy not changing
  - 500k steps in ~2 min → episodes terminating in ~20 steps (crashes)
  - Agent crashes on fuel before ever reaching an infected cell
  - No gradient signal → PPO can't learn anything

Three-layer fix:
  1. EasierEnv wrapper — fuel=200, n_infected=2, spread=off,
     step_penalty=-0.1, partial_obs=off (auto-reveal disease type)
  2. RewardShapingWrapper — dense proximity reward guides agent
     toward infected cells before it learns to treat them
  3. VecNormalize — normalises observations AND returns,
     critical for stable PPO on environments with large reward variance

Usage:
    python training/improved/improved_ppo_v2.py
    python training/improved/improved_ppo_v2.py --run 3
    python training/improved/improved_ppo_v2.py --smoke-test
"""

import os, sys, csv, time, argparse
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import gymnasium as gym
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from environment.custom_env import CropDroneEnv, AT_RISK, FUNGAL, PEST, TREATED

MODEL_DIR = "models/improved/ppo_v2"
LOG_DIR   = "logs/improved/ppo_v2"
CSV_PATH  = "models/improved/ppo_v2/ppo_v2_results.csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

TOTAL_TIMESTEPS = 600_000   # ~10–16 min per run on M1 Pro
EVAL_EPISODES   = 10
EVAL_FREQ       = 25_000
N_ENVS          = 4


# ─────────────────────────────────────────────────────────────
# LAYER 1 — Easier environment parameters
# Start simpler so the agent can find positive rewards,
# then we can tighten constraints once learning is confirmed.
# ─────────────────────────────────────────────────────────────
EASY_ENV_KWARGS = dict(
    grid_size       = 8,
    max_steps       = 200,
    max_fuel        = 200,       # was 100 — agent was crashing constantly
    max_payload     = 10,        # was 3  — removes payload constraint initially
    spread_interval = 999,       # effectively disabled — removes moving target
    spread_prob     = 0.0,       # no spread — static disease map
    n_infected      = 3,         # was 5  — fewer targets = easier
)


# ─────────────────────────────────────────────────────────────
# LAYER 2 — Reward shaping wrapper
# Adds dense proximity reward so agent gets signal for
# moving toward infected cells, not just on treatment.
# ─────────────────────────────────────────────────────────────
class RewardShapingWrapper(gym.Wrapper):
    """
    Adds shaped rewards on top of the base environment:
      +proximity_reward  for each step closer to nearest infected cell
      -proximity_reward  for each step further away
      -0.1 per step      (replaces -0.5 — much less harsh)
      Auto-reveals disease type (removes partial observability)
    """

    def __init__(self, env: CropDroneEnv, proximity_scale: float = 2.0):
        super().__init__(env)
        self.proximity_scale = proximity_scale
        self._prev_dist      = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Auto-reveal all disease types — remove partial observability
        # so agent learns treatment before learning to scan
        self._reveal_all()
        self._prev_dist = self._nearest_infected_dist()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Replace harsh step penalty with lighter one
        # (base env applies -0.5, we add +0.4 to make it -0.1 effective)
        reward += 0.4

        # Auto-reveal on every step (disease may spread in base env)
        self._reveal_all()

        # Proximity shaping — reward for moving closer to nearest infection
        curr_dist = self._nearest_infected_dist()
        if self._prev_dist is not None and curr_dist is not None:
            delta = self._prev_dist - curr_dist   # positive = got closer
            reward += self.proximity_scale * delta
        self._prev_dist = curr_dist

        return obs, reward, terminated, truncated, info

    def _reveal_all(self):
        """Force all infected cells to show their true disease type."""
        env = self.env
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if env.grid[r, c] == AT_RISK and env.hidden_disease[r, c] in (FUNGAL, PEST):
                    env.grid[r, c] = env.hidden_disease[r, c]
                    env.scanned[r, c] = True

    def _nearest_infected_dist(self):
        """Manhattan distance from drone to nearest infected cell."""
        env = self.env
        dr, dc = env.drone_row, env.drone_col
        infected = np.argwhere(
            (env.grid == AT_RISK) | (env.grid == FUNGAL) | (env.grid == PEST)
        )
        if len(infected) == 0:
            return 0.0
        dists = np.abs(infected[:, 0] - dr) + np.abs(infected[:, 1] - dc)
        return float(dists.min())


# ─────────────────────────────────────────────────────────────
# 5 CONFIGS — vary PPO hyperparameters on the fixed env
# ─────────────────────────────────────────────────────────────
PPO_V2_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Baseline on easy env + reward shaping
    {
        "run_id": 1,
        "description": "Easy env + shaping — lr=1e-4, n_steps=512, ent=0.01, epochs=5",
        "learning_rate": 1e-4,
        "n_steps":       512,
        "batch_size":    64,
        "n_epochs":      5,
        "gamma":         0.99,
        "gae_lambda":    0.95,
        "ent_coef":      0.01,
        "clip_range":    0.2,
        "vf_coef":       0.5,
        "net_arch":      [dict(pi=[256, 256], vf=[256, 256])],
        "proximity_scale": 2.0,
    },
    # Run 2 — Higher proximity reward: stronger signal toward targets
    {
        "run_id": 2,
        "description": "Strong proximity — lr=1e-4, n_steps=512, prox=4.0, ent=0.01",
        "learning_rate": 1e-4,
        "n_steps":       512,
        "batch_size":    64,
        "n_epochs":      5,
        "gamma":         0.99,
        "gae_lambda":    0.95,
        "ent_coef":      0.01,
        "clip_range":    0.2,
        "vf_coef":       0.5,
        "net_arch":      [dict(pi=[256, 256], vf=[256, 256])],
        "proximity_scale": 4.0,
    },
    # Run 3 — Longer rollout to capture full treatment sequences
    {
        "run_id": 3,
        "description": "Long rollout — lr=1e-4, n_steps=1024, prox=2.0, ent=0.01",
        "learning_rate": 1e-4,
        "n_steps":       1024,
        "batch_size":    128,
        "n_epochs":      5,
        "gamma":         0.99,
        "gae_lambda":    0.95,
        "ent_coef":      0.01,
        "clip_range":    0.2,
        "vf_coef":       0.5,
        "net_arch":      [dict(pi=[256, 256], vf=[256, 256])],
        "proximity_scale": 2.0,
    },
    # Run 4 — Higher entropy: explore treatment actions more
    {
        "run_id": 4,
        "description": "Higher entropy — lr=1e-4, n_steps=512, prox=2.0, ent=0.05",
        "learning_rate": 1e-4,
        "n_steps":       512,
        "batch_size":    64,
        "n_epochs":      5,
        "gamma":         0.99,
        "gae_lambda":    0.95,
        "ent_coef":      0.05,
        "clip_range":    0.2,
        "vf_coef":       0.5,
        "net_arch":      [dict(pi=[256, 256], vf=[256, 256])],
        "proximity_scale": 2.0,
    },
    # Run 5 — Best guess: moderate proximity + longer rollout + entropy
    {
        "run_id": 5,
        "description": "Best combo — lr=1e-4, n_steps=1024, prox=3.0, ent=0.02, epochs=8",
        "learning_rate": 1e-4,
        "n_steps":       1024,
        "batch_size":    128,
        "n_epochs":      8,
        "gamma":         0.99,
        "gae_lambda":    0.95,
        "ent_coef":      0.02,
        "clip_range":    0.2,
        "vf_coef":       0.5,
        "net_arch":      [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
        "proximity_scale": 3.0,
    },
]


# ─────────────────────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────────────────────
def make_env_fn(seed: int = 0, proximity_scale: float = 2.0, render_mode=None):
    def _init():
        base = CropDroneEnv(**EASY_ENV_KWARGS, render_mode=render_mode, seed=seed)
        env  = RewardShapingWrapper(base, proximity_scale=proximity_scale)
        return Monitor(env)
    return _init


# ─────────────────────────────────────────────────────────────
# TQDM CALLBACK
# ─────────────────────────────────────────────────────────────
class TqdmCB(BaseCallback):
    def __init__(self, total: int, run_id: int):
        super().__init__()
        self.total = total; self.run_id = run_id; self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total,
            desc=f"  PPO v2 Run {self.run_id}/5",
            unit="step", ncols=92, colour="cyan",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def _on_step(self):
        self.pbar.update(1); return True

    def _on_training_end(self):
        if self.pbar: self.pbar.close()


# ─────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────
def run_experiment(config: Dict, timesteps: int, seed: int) -> Dict:
    run_id = config["run_id"]
    prox   = config["proximity_scale"]

    print(f"\n{'='*62}")
    print(f"  PPO v2 — Run {run_id}/5")
    print(f"  {config['description']}")
    print(f"  proximity_scale={prox}  |  easy_env=True  |  VecNormalize=True")
    print(f"{'='*62}")

    run_dir = os.path.join(MODEL_DIR, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    # LAYER 3 — VecNormalize wraps the vectorised env
    # Normalises observations (mean=0, std=1) AND reward scale
    # Critical for PPO stability on high-variance reward environments
    train_vec = make_vec_env(make_env_fn(seed=seed, proximity_scale=prox), n_envs=N_ENVS)
    train_env = VecNormalize(train_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval env — normalise obs but NOT reward (so we see true reward)
    eval_vec  = make_vec_env(make_env_fn(seed=seed+100, proximity_scale=prox), n_envs=1)
    eval_env  = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False)

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
        normalize_advantage = True,
        policy_kwargs   = {"net_arch": config["net_arch"]},
        tensorboard_log = os.path.join(LOG_DIR, f"run_{run_id:02d}"),
        verbose         = 0,
        seed            = seed,
        device          = "cpu",
    )

    # Sync VecNormalize stats between train and eval
    def sync_vecnorm(locals_, globals_):
        eval_env.obs_rms = train_env.obs_rms

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = run_dir,
        log_path             = run_dir,
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,
        verbose              = 0,
        callback_after_eval  = None,
    )

    t0 = time.time()
    model.learn(
        total_timesteps = timesteps,
        callback        = [TqdmCB(timesteps, run_id), eval_cb],
        progress_bar    = False,
    )
    elapsed = time.time() - t0

    # Sync stats before final eval
    eval_env.obs_rms = train_env.obs_rms
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
    )

    model.save(os.path.join(run_dir, "final_model"))
    # Save VecNormalize stats (needed to load model later)
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))

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
        train_env.save(os.path.join(MODEL_DIR, "vecnormalize_best.pkl"))
        with open(best_txt, "w") as f: f.write(str(mean_r))
        print(f"  ★ New best PPO v2 (reward={mean_r:.2f})")

    train_env.close(); eval_env.close()
    print(f"\n  ✓ Run {run_id} — reward={mean_r:.2f}±{std_r:.2f}  time={elapsed/60:.1f}m")

    return {
        "run_id":           run_id,
        "description":      config["description"],
        "learning_rate":    config["learning_rate"],
        "n_steps":          config["n_steps"],
        "n_envs":           N_ENVS,
        "batch_size":       config["batch_size"],
        "n_epochs":         config["n_epochs"],
        "gamma":            config["gamma"],
        "ent_coef":         config["ent_coef"],
        "proximity_scale":  prox,
        "vec_normalize":    True,
        "easy_env":         True,
        "total_timesteps":  timesteps,
        "mean_reward":      round(mean_r, 2),
        "std_reward":       round(std_r, 2),
        "train_time_min":   round(elapsed / 60, 2),
    }


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",        type=int,  default=None)
    parser.add_argument("--timesteps",  type=int,  default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        print("\n  Smoke test — 3000 steps, Run 1")
        r = run_experiment(PPO_V2_CONFIGS[0], timesteps=3_000, seed=42)
        print(f"  ✓ Smoke test passed  reward={r['mean_reward']}")
        import sys; sys.exit(0)

    configs = PPO_V2_CONFIGS if args.run is None else [PPO_V2_CONFIGS[args.run - 1]]
    results = []

    print(f"\n  PPO v2 Training  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Timesteps: {args.timesteps:,}  |  n_envs={N_ENVS}  |  VecNormalize=ON")
    print(f"  Easy env: fuel=200, n_infected=3, spread=OFF, step_penalty=-0.1")

    for cfg in configs:
        results.append(run_experiment(cfg, args.timesteps, args.seed))

    if len(results) > 1:
        print(f"\n{'='*62}  PPO v2 SUMMARY")
        best_r = max(r["mean_reward"] for r in results)
        for r in results:
            m = " ★" if r["mean_reward"] == best_r else "  "
            print(f"  Run {r['run_id']}{m}  reward={r['mean_reward']:>8.2f}±{r['std_reward']:<7.2f}  {r['description']}")

        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)

        print(f"\n  Results → {CSV_PATH}")
        print(f"  Best model → {MODEL_DIR}/best_model.zip")
        print(f"  TensorBoard → tensorboard --logdir {LOG_DIR}")
