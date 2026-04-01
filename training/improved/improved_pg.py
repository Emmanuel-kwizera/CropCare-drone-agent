"""
training/improved/improved_pg.py
─────────────────────────────────────────────────────────────
Improved PPO and REINFORCE training based on analysis of
initial poor results.

Key changes from initial runs:

PPO:
  1. n_envs=4 parallel environments — biggest single improvement for PPO
  2. lr=1e-4 (Run 4 was best — keep low lr)
  3. 500k timesteps (vs 150k)
  4. Lower entropy (0.005) — initial results showed high entropy hurts
  5. Fewer epochs (5) — Run 9 showed 20 epochs causes policy collapse

REINFORCE:
  1. 2000 episodes (vs 700) — 700 episodes in 18 seconds is massively under-trained
  2. No baseline (Run 1 outperformed runs with baseline)
  3. hidden=128 (Run 7 was best — smaller net more stable)
  4. Episode batching — accumulate gradients over N episodes before updating

Usage:
    python training/improved/improved_pg.py --algo ppo
    python training/improved/improved_pg.py --algo reinforce
    python training/improved/improved_pg.py --algo all
    python training/improved/improved_pg.py --algo ppo --run 2
    python training/improved/improved_pg.py --algo ppo --smoke-test
"""

import os, sys, csv, time, argparse
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from environment.custom_env import CropDroneEnv

PPO_DIR       = "models/improved/pg"
REINFORCE_DIR = "models/improved/pg"
LOG_PPO       = "logs/improved/ppo"
LOG_REINFORCE = "logs/improved/reinforce"
for d in [PPO_DIR, LOG_PPO, LOG_REINFORCE]:
    os.makedirs(d, exist_ok=True)

PPO_TIMESTEPS      = 500_000   
REINFORCE_EPISODES = 2_000     
EVAL_EPISODES      = 10
EVAL_FREQ          = 25_000

N_ENVS = 4   # parallel environments for PPO — biggest single improvement


# ═════════════════════════════════════════════════════════════
# IMPROVED PPO CONFIGS — 5 focused runs
# ═════════════════════════════════════════════════════════════
IMPROVED_PPO_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Parallel envs + low LR (core improvement)
    # Run 4 (lr=1e-4) was best. Adding 4 parallel envs gives
    # 4x more diverse rollout data per update.
    {
        "run_id": 1,
        "description": "Parallel envs — lr=1e-4, n_envs=4, n_steps=512, ent=0.005, epochs=5",
        "learning_rate":  1e-4,
        "n_steps":        512,      # per env — 4×512=2048 effective rollout
        "batch_size":     64,
        "n_epochs":       5,        # fewer epochs — avoids policy collapse
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.005,    # lower entropy — high ent hurt in Run 3
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 2 — Even lower LR + more rollout data
    # Test if going even slower helps stability on partial obs env.
    {
        "run_id": 2,
        "description": "Very low LR — lr=5e-5, n_envs=4, n_steps=512, ent=0.005, epochs=5",
        "learning_rate":  5e-5,
        "n_steps":        512,
        "batch_size":     64,
        "n_epochs":       5,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.005,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 3 — Longer effective rollout with parallel envs
    # n_steps=1024 per env × 4 envs = 4096 effective rollout.
    # Longer rollouts capture full episode sequences better.
    {
        "run_id": 3,
        "description": "Long effective rollout — lr=1e-4, n_envs=4, n_steps=1024, epochs=5",
        "learning_rate":  1e-4,
        "n_steps":        1024,
        "batch_size":     128,
        "n_epochs":       5,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.008,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Run 4 — Linear LR schedule (decays to zero by end of training)
    # Aggressive early, conservative late — helps stabilise convergence.
    {
        "run_id": 4,
        "description": "LR schedule — lr=linear(1e-4→0), n_envs=4, n_steps=512, epochs=5",
        "learning_rate":  "linear",   # SB3 supports string 'linear' for linear decay
        "n_steps":        512,
        "batch_size":     64,
        "n_epochs":       5,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.005,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256], vf=[256, 256])],
        "_lr_init":        1e-4,  # used when learning_rate == 'linear'
    },
    # Run 5 — Best guess combo
    # Low LR + parallel envs + moderate entropy + 8 epochs (sweet spot)
    # between too few (Run 1) and too many (original Run 9).
    {
        "run_id": 5,
        "description": "Best combo — lr=1e-4, n_envs=4, n_steps=512, ent=0.01, epochs=8",
        "learning_rate":  1e-4,
        "n_steps":        512,
        "batch_size":     64,
        "n_epochs":       8,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "ent_coef":       0.01,
        "clip_range":     0.2,
        "vf_coef":        0.5,
        "net_arch":       [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
    },
]


# ═════════════════════════════════════════════════════════════
# IMPROVED REINFORCE CONFIGS — 5 focused runs
# ═════════════════════════════════════════════════════════════
IMPROVED_REINFORCE_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — No baseline + small net (best from initial)
    # Run 7 (hidden=128, no baseline) was best. Scale up episodes.
    {
        "run_id": 1,
        "description": "No baseline + small net — lr=1e-3, hidden=128, batch_eps=1",
        "learning_rate":     1e-3,
        "gamma":             0.99,
        "hidden_dim":        128,
        "use_baseline":      False,
        "entropy_coef":      0.005,
        "normalize_returns": False,
        "batch_episodes":    1,       # update every episode
    },
    # Run 2 — Episode batching (accumulate 4 episodes before update)
    # Reduces gradient variance by averaging over multiple episodes.
    {
        "run_id": 2,
        "description": "Episode batch=4 — lr=1e-3, hidden=128, no baseline, batch_eps=4",
        "learning_rate":     1e-3,
        "gamma":             0.99,
        "hidden_dim":        128,
        "use_baseline":      False,
        "entropy_coef":      0.005,
        "normalize_returns": False,
        "batch_episodes":    4,
    },
    # Run 3 — Lower LR + episode batching
    # Slower updates with more data per update = more stable.
    {
        "run_id": 3,
        "description": "Low LR + batch — lr=5e-4, hidden=128, no baseline, batch_eps=4",
        "learning_rate":     5e-4,
        "gamma":             0.99,
        "hidden_dim":        128,
        "use_baseline":      False,
        "entropy_coef":      0.005,
        "normalize_returns": False,
        "batch_episodes":    4,
    },
    # Run 4 — Tiny network (64 hidden)
    # Even smaller network = less overfitting to noisy episode returns.
    {
        "run_id": 4,
        "description": "Tiny net — lr=1e-3, hidden=64, no baseline, batch_eps=2",
        "learning_rate":     1e-3,
        "gamma":             0.99,
        "hidden_dim":        64,
        "use_baseline":      False,
        "entropy_coef":      0.01,
        "normalize_returns": False,
        "batch_episodes":    2,
    },
    # Run 5 — Best guess combo
    # Moderate LR + small net + episode batching + slight entropy
    {
        "run_id": 5,
        "description": "Best combo — lr=8e-4, hidden=128, no baseline, batch_eps=4, ent=0.008",
        "learning_rate":     8e-4,
        "gamma":             0.99,
        "hidden_dim":        128,
        "use_baseline":      False,
        "entropy_coef":      0.008,
        "normalize_returns": False,
        "batch_episodes":    4,
    },
]


# ═════════════════════════════════════════════════════════════
# PPO TRAINING
# ═════════════════════════════════════════════════════════════
class TqdmPPO(BaseCallback):
    def __init__(self, total: int, run_id: int):
        super().__init__()
        self.total = total; self.run_id = run_id; self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total, desc=f"  Improved PPO Run {self.run_id}/5",
                         unit="step", ncols=88, colour="cyan",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    def _on_step(self):
        self.pbar.update(1); return True

    def _on_training_end(self):
        if self.pbar: self.pbar.close()


def make_ppo_env(seed=0):
    def _init():
        env = CropDroneEnv(grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
                           spread_interval=10, spread_prob=0.20, n_infected=5,
                           render_mode=None, seed=seed)
        return Monitor(env)
    return _init


def run_ppo_experiment(config: Dict, timesteps: int, seed: int) -> Dict:
    run_id = config["run_id"]
    print(f"\n{'='*60}")
    print(f"  Improved PPO — Run {run_id}/5")
    print(f"  {config['description']}")
    print(f"{'='*60}")

    run_dir = os.path.join(PPO_DIR, f"ppo_run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    # n_envs parallel environments — biggest PPO improvement
    train_env = make_vec_env(make_ppo_env(seed=seed), n_envs=N_ENVS)
    eval_env  = Monitor(CropDroneEnv(grid_size=8, max_steps=200, max_fuel=100,
                                     max_payload=3, spread_interval=10, spread_prob=0.20,
                                     n_infected=5, render_mode=None, seed=seed+200))

    # Handle linear LR schedule
    lr = config["learning_rate"]
    if lr == "linear":
        lr_init = config["_lr_init"]
        lr = lambda progress: lr_init * progress   # SB3 passes remaining fraction

    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = lr,
        n_steps         = config["n_steps"],
        batch_size      = config["batch_size"],
        n_epochs        = config["n_epochs"],
        gamma           = config["gamma"],
        gae_lambda      = config["gae_lambda"],
        ent_coef        = config["ent_coef"],
        clip_range      = config["clip_range"],
        vf_coef         = config["vf_coef"],
        policy_kwargs   = {"net_arch": config["net_arch"]},
        tensorboard_log = os.path.join(LOG_PPO, f"run_{run_id:02d}"),
        verbose         = 0,
        seed            = seed,
        device          = "cpu",
    )

    eval_cb = EvalCallback(eval_env, best_model_save_path=run_dir, log_path=run_dir,
                           eval_freq=max(EVAL_FREQ // N_ENVS, 1),
                           n_eval_episodes=EVAL_EPISODES, deterministic=True, verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=timesteps,
                callback=[TqdmPPO(timesteps, run_id), eval_cb],
                progress_bar=False)
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True)
    model.save(os.path.join(run_dir, "final_model"))

    best_path = os.path.join(PPO_DIR, "ppo_best")
    best_txt  = os.path.join(PPO_DIR, "ppo_best_reward.txt")
    current_best = -np.inf
    if os.path.exists(best_txt):
        with open(best_txt) as f:
            try: current_best = float(f.read())
            except: pass
    if mean_r > current_best:
        model.save(best_path)
        with open(best_txt, "w") as f: f.write(str(mean_r))
        print(f"  ★ New best PPO (reward={mean_r:.2f})")

    print(f"\n  ✓ PPO Run {run_id} — reward={mean_r:.2f}±{std_r:.2f}  time={elapsed/60:.1f}m")
    train_env.close(); eval_env.close()

    return {
        "run_id": run_id, "description": config["description"],
        "learning_rate": config["learning_rate"] if config["learning_rate"] != "linear" else "linear(1e-4→0)",
        "n_steps": config["n_steps"], "n_envs": N_ENVS,
        "batch_size": config["batch_size"], "n_epochs": config["n_epochs"],
        "gamma": config["gamma"], "ent_coef": config["ent_coef"],
        "clip_range": config["clip_range"], "total_timesteps": timesteps,
        "mean_reward": round(mean_r, 2), "std_reward": round(std_r, 2),
        "train_time_min": round(elapsed / 60, 2),
    }


# ═════════════════════════════════════════════════════════════
# IMPROVED REINFORCE
# ═════════════════════════════════════════════════════════════
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def act(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.forward(obs_t)
        dist  = Categorical(probs)
        a     = dist.sample()
        return a.item(), dist.log_prob(a), dist.entropy()


def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    G, ret = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        ret.insert(0, G)
    return torch.FloatTensor(ret)


def run_reinforce_experiment(config: Dict, total_episodes: int, seed: int) -> Dict:
    run_id = config["run_id"]
    print(f"\n{'='*60}")
    print(f"  Improved REINFORCE — Run {run_id}/5")
    print(f"  {config['description']}")
    print(f"{'='*60}")

    run_dir = os.path.join(REINFORCE_DIR, f"reinforce_run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed); np.random.seed(seed)

    env = CropDroneEnv(grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
                       spread_interval=10, spread_prob=0.20, n_infected=5,
                       render_mode=None, seed=seed)

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy    = PolicyNet(obs_dim, n_actions, config["hidden_dim"])
    optimizer = optim.Adam(policy.parameters(), lr=config["learning_rate"])

    ep_rewards = []
    best_mean  = -np.inf
    t0 = time.time()
    batch_size = config["batch_episodes"]

    pbar = tqdm(range(0, total_episodes, batch_size),
                desc=f"  Improved REINFORCE Run {run_id}/5",
                unit=f"batch({batch_size}ep)", ncols=88, colour="magenta",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

    for batch_start in pbar:
        # Collect batch_size episodes before updating
        batch_log_probs, batch_returns, batch_entropies = [], [], []

        for _ in range(batch_size):
            obs, _ = env.reset()
            log_probs, rewards, entropies = [], [], []
            done = False
            while not done:
                a, lp, ent = policy.act(obs)
                obs, r, terminated, truncated, _ = env.step(a)
                log_probs.append(lp); rewards.append(r); entropies.append(ent)
                done = terminated or truncated

            ep_rewards.append(sum(rewards))
            returns = compute_returns(rewards, config["gamma"])

            if config["use_baseline"]:
                returns = returns - returns.mean()

            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns.tolist())
            batch_entropies.extend(entropies)

        # Gradient update over entire batch
        lp_t  = torch.stack(batch_log_probs)
        ret_t = torch.FloatTensor(batch_returns)
        ent_t = torch.stack(batch_entropies)

        loss = -(lp_t * ret_t).mean() - config["entropy_coef"] * ent_t.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        # Update tqdm every 10 batches
        ep_done = min(batch_start + batch_size, total_episodes)
        if ep_done % max(50, batch_size * 10) == 0 or ep_done >= total_episodes:
            recent = float(np.mean(ep_rewards[-max(50, batch_size * 10):]))
            pbar.set_postfix(reward=f"{recent:.1f}", loss=f"{loss.item():.3f}", refresh=False)
            if recent > best_mean:
                best_mean = recent
                torch.save(policy.state_dict(), os.path.join(run_dir, "best_policy.pt"))

    pbar.close()
    elapsed = time.time() - t0

    # Greedy evaluation
    eval_env = CropDroneEnv(grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
                            spread_interval=10, spread_prob=0.20, n_infected=5,
                            render_mode=None, seed=seed+300)
    eval_rs = []
    policy.eval()
    with torch.no_grad():
        for _ in range(EVAL_EPISODES):
            obs, _ = eval_env.reset()
            ep_r, done = 0.0, False
            while not done:
                probs  = policy(torch.FloatTensor(obs).unsqueeze(0))
                action = probs.argmax(dim=-1).item()
                obs, r, te, tr, _ = eval_env.step(action)
                ep_r += r; done = te or tr
            eval_rs.append(ep_r)
    eval_env.close()

    mean_r = float(np.mean(eval_rs))
    std_r  = float(np.std(eval_rs))
    torch.save(policy.state_dict(), os.path.join(run_dir, "final_policy.pt"))

    best_path = os.path.join(REINFORCE_DIR, "reinforce_best.pt")
    best_txt  = os.path.join(REINFORCE_DIR, "reinforce_best_reward.txt")
    current_best = -np.inf
    if os.path.exists(best_txt):
        with open(best_txt) as f:
            try: current_best = float(f.read())
            except: pass
    if mean_r > current_best:
        torch.save(policy.state_dict(), best_path)
        with open(best_txt, "w") as f: f.write(str(mean_r))
        print(f"  ★ New best REINFORCE (reward={mean_r:.2f})")

    print(f"\n  ✓ REINFORCE Run {run_id} — reward={mean_r:.2f}±{std_r:.2f}  time={elapsed/60:.1f}m")
    env.close()

    return {
        "run_id": run_id, "description": config["description"],
        "learning_rate": config["learning_rate"], "gamma": config["gamma"],
        "hidden_dim": config["hidden_dim"], "use_baseline": config["use_baseline"],
        "entropy_coef": config["entropy_coef"], "batch_episodes": config["batch_episodes"],
        "total_episodes": total_episodes, "mean_reward": round(mean_r, 2),
        "std_reward": round(std_r, 2), "train_time_min": round(elapsed / 60, 2),
    }


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       choices=["ppo", "reinforce", "all"], default="all")
    parser.add_argument("--run",        type=int, default=None)
    parser.add_argument("--ppo-steps",  type=int, default=PPO_TIMESTEPS)
    parser.add_argument("--rein-eps",   type=int, default=REINFORCE_EPISODES)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    smoke = args.smoke_test

    def pick(cfgs):
        if args.run is not None:
            idx = args.run - 1
            assert 0 <= idx < len(cfgs), f"--run must be 1-{len(cfgs)}"
            return [cfgs[idx]]
        return cfgs

    print(f"\n  Improved PG Training  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Algorithm: {args.algo.upper()}  |  PPO n_envs={N_ENVS}  |  Device: CPU")

    if args.algo in ("ppo", "all"):
        steps = 3_000 if smoke else args.ppo_steps
        ppo_results = [run_ppo_experiment(c, steps, args.seed) for c in pick(IMPROVED_PPO_CONFIGS)]
        if len(ppo_results) > 1:
            print(f"\n{'='*60}  IMPROVED PPO SUMMARY")
            best = max(r["mean_reward"] for r in ppo_results)
            for r in ppo_results:
                m = " ★" if r["mean_reward"] == best else "  "
                print(f"  Run {r['run_id']}{m}  reward={r['mean_reward']:>8.2f}  {r['description']}")
            path = os.path.join(PPO_DIR, "improved_ppo_results.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=ppo_results[0].keys())
                w.writeheader(); w.writerows(ppo_results)
            print(f"  Results → {path}")

    if args.algo in ("reinforce", "all"):
        eps = 30 if smoke else args.rein_eps
        rein_results = [run_reinforce_experiment(c, eps, args.seed) for c in pick(IMPROVED_REINFORCE_CONFIGS)]
        if len(rein_results) > 1:
            print(f"\n{'='*60}  IMPROVED REINFORCE SUMMARY")
            best = max(r["mean_reward"] for r in rein_results)
            for r in rein_results:
                m = " ★" if r["mean_reward"] == best else "  "
                print(f"  Run {r['run_id']}{m}  reward={r['mean_reward']:>8.2f}  {r['description']}")
            path = os.path.join(REINFORCE_DIR, "improved_reinforce_results.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rein_results[0].keys())
                w.writeheader(); w.writerows(rein_results)
            print(f"  Results → {path}")

    print(f"\n  Done. Best models in: {PPO_DIR}/")
    print(f"  TensorBoard: tensorboard --logdir logs/improved/")
