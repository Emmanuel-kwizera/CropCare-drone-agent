"""
training/improved/improved_reinforce_v2.py
─────────────────────────────────────────────────────────────
REINFORCE v2 — targets positive reward territory.

Root cause of v1 failure:
  - Runs 1,2,4,5 identical (−440.8) → policy never updated meaningfully
  - Run 3 collapsed to −1149.8 → gradient exploded on negative returns
  - 2000 episodes in 42 seconds → episodes ~1.2s each → crash at ~step 20
  - With max_fuel=100, agent depleted fuel before finding any infected cell
  - REINFORCE on all-negative returns → pushes policy toward inaction/walls

Three-layer fix (mirrors PPO v2):
  1. Easy env  — fuel=200, payload=10, spread=OFF, n_infected=3
                 agent can complete episodes without crashing
  2. RewardShapingWrapper — proximity reward guides toward infected cells
                            even before treatment is discovered
  3. Return normalisation per episode — standardises scale of gradient
     signal so learning rate behaves consistently

Additional REINFORCE-specific improvements:
  - Reward clipping at ±50 before return computation
  - Larger batch accumulation (8-16 episodes) for stable gradient estimate
  - Grad clipping at 1.0 (was 0.5 — too tight, blocked learning)
  - 4000 episodes total (was 2000 — still too few given fast episodes)

Usage:
    python training/improved/improved_reinforce_v2.py
    python training/improved/improved_reinforce_v2.py --run 3
    python training/improved/improved_reinforce_v2.py --smoke-test
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

from environment.custom_env import CropDroneEnv
from training.improved.improved_ppo_v2 import RewardShapingWrapper, EASY_ENV_KWARGS

MODEL_DIR = "models/improved/reinforce_v2"
LOG_DIR   = "logs/improved/reinforce_v2"
CSV_PATH  = "models/improved/reinforce_v2/reinforce_v2_results.csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

TOTAL_EPISODES = 4_000   
EVAL_EPISODES  = 15
REWARD_CLIP    = 50.0    


# ─────────────────────────────────────────────────────────────
# 5 FOCUSED CONFIGS
# All use easy env + reward shaping.
# Vary: lr, batch_episodes, entropy, hidden_dim, baseline
# ─────────────────────────────────────────────────────────────
CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Easy env + large batch (16 eps)
    # Larger batch averages gradient over more episodes → much lower variance
    {
        "run_id":          1,
        "description":     "Easy env + batch=16 — lr=1e-3, hidden=128, ent=0.01",
        "learning_rate":   1e-3,
        "gamma":           0.99,
        "hidden_dim":      128,
        "entropy_coef":    0.01,
        "batch_episodes":  16,
        "reward_clip":     REWARD_CLIP,
        "normalize":       True,
        "grad_clip":       1.0,
    },
    # Run 2 — Medium batch + stronger entropy
    # More entropy → more exploration of treatment actions
    {
        "run_id":          2,
        "description":     "Strong entropy — lr=1e-3, hidden=128, ent=0.05, batch=8",
        "learning_rate":   1e-3,
        "gamma":           0.99,
        "hidden_dim":      128,
        "entropy_coef":    0.05,
        "batch_episodes":  8,
        "reward_clip":     REWARD_CLIP,
        "normalize":       True,
        "grad_clip":       1.0,
    },
    # Run 3 — Lower LR + large batch (fix for v1 Run 3 collapse)
    # v1 Run 3 (lr=5e-4) collapsed because returns were still all-negative
    # With easy env the returns are mixed → lower lr becomes safe + stable
    {
        "run_id":          3,
        "description":     "Low LR stable — lr=3e-4, hidden=128, ent=0.01, batch=16",
        "learning_rate":   3e-4,
        "gamma":           0.99,
        "hidden_dim":      128,
        "entropy_coef":    0.01,
        "batch_episodes":  16,
        "reward_clip":     REWARD_CLIP,
        "normalize":       True,
        "grad_clip":       1.0,
    },
    # Run 4 — Larger network + moderate batch
    # More capacity to represent the treatment-navigation policy
    {
        "run_id":          4,
        "description":     "Larger net — lr=1e-3, hidden=256, ent=0.01, batch=8",
        "learning_rate":   1e-3,
        "gamma":           0.99,
        "hidden_dim":      256,
        "entropy_coef":    0.01,
        "batch_episodes":  8,
        "reward_clip":     REWARD_CLIP,
        "normalize":       True,
        "grad_clip":       1.0,
    },
    # Run 5 — Best guess combo
    # Moderate lr + large batch + moderate entropy + clipping
    {
        "run_id":          5,
        "description":     "Best combo — lr=8e-4, hidden=128, ent=0.02, batch=16",
        "learning_rate":   8e-4,
        "gamma":           0.99,
        "hidden_dim":      128,
        "entropy_coef":    0.02,
        "batch_episodes":  16,
        "reward_clip":     REWARD_CLIP,
        "normalize":       True,
        "grad_clip":       1.0,
    },
]


# ─────────────────────────────────────────────────────────────
# POLICY NETWORK
# ─────────────────────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

    def act(self, obs: np.ndarray):
        t = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.forward(t)
        dist  = Categorical(probs)
        a     = dist.sample()
        return a.item(), dist.log_prob(a), dist.entropy()


def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    G, ret = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        ret.insert(0, G)
    return torch.FloatTensor(ret)


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
def make_env(seed: int = 0) -> CropDroneEnv:
    base = CropDroneEnv(**EASY_ENV_KWARGS, render_mode=None, seed=seed)
    return RewardShapingWrapper(base, proximity_scale=2.0)


def run_experiment(config: Dict, total_episodes: int, seed: int) -> Dict:
    run_id = config["run_id"]
    print(f"\n{'='*62}")
    print(f"  REINFORCE v2 — Run {run_id}/5")
    print(f"  {config['description']}")
    print(f"  easy_env=True  |  reward_shaping=True  |  reward_clip={config['reward_clip']}")
    print(f"{'='*62}")

    run_dir = os.path.join(MODEL_DIR, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(seed=seed)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy    = PolicyNet(obs_dim, n_actions, config["hidden_dim"])
    optimizer = optim.Adam(policy.parameters(), lr=config["learning_rate"])

    batch_size = config["batch_episodes"]
    ep_rewards = []
    best_mean  = -np.inf
    t0 = time.time()

    n_batches = total_episodes // batch_size
    pbar = tqdm(
        range(n_batches),
        desc=f"  REINFORCE v2 Run {run_id}/5",
        unit=f"batch({batch_size})",
        ncols=92,
        colour="magenta",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for batch_idx in pbar:
        all_log_probs = []
        all_returns   = []
        all_entropies = []

        for _ in range(batch_size):
            obs, _ = env.reset()
            lps, rwds, ents = [], [], []
            done = False

            while not done:
                a, lp, ent = policy.act(obs)
                obs, r, terminated, truncated, _ = env.step(a)
                # Clip reward before storing (prevents exploding returns)
                rwds.append(float(np.clip(r, -config["reward_clip"], config["reward_clip"])))
                lps.append(lp)
                ents.append(ent)
                done = terminated or truncated

            ep_rewards.append(sum(rwds))

            returns = compute_returns(rwds, config["gamma"])

            # Per-episode return normalisation
            if config["normalize"] and returns.std() > 1e-6:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            all_log_probs.extend(lps)
            all_returns.extend(returns.tolist())
            all_entropies.extend(ents)

        # ── Batch gradient update ─────────────────
        lp_t  = torch.stack(all_log_probs)
        ret_t = torch.FloatTensor(all_returns)
        ent_t = torch.stack(all_entropies)

        loss = -(lp_t * ret_t).mean() - config["entropy_coef"] * ent_t.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config["grad_clip"])
        optimizer.step()

        # ── Update tqdm ───────────────────────────
        window = min(len(ep_rewards), batch_size * 5)
        recent = float(np.mean(ep_rewards[-window:]))
        pbar.set_postfix(reward=f"{recent:.1f}", loss=f"{loss.item():.3f}", refresh=False)

        if recent > best_mean:
            best_mean = recent
            torch.save(policy.state_dict(), os.path.join(run_dir, "best_policy.pt"))

    pbar.close()
    elapsed = time.time() - t0

    # ── Greedy evaluation ─────────────────────────
    eval_env = make_env(seed=seed + 300)
    eval_rs  = []
    policy.eval()
    with torch.no_grad():
        for _ in range(EVAL_EPISODES):
            obs, _ = eval_env.reset()
            ep_r, done = 0.0, False
            while not done:
                probs  = policy(torch.FloatTensor(obs).unsqueeze(0))
                action = probs.argmax(dim=-1).item()
                obs, r, te, tr, _ = eval_env.step(action)
                ep_r += r
                done  = te or tr
            eval_rs.append(ep_r)
    eval_env.close()
    env.close()

    mean_r = float(np.mean(eval_rs))
    std_r  = float(np.std(eval_rs))

    # Save final weights
    torch.save(policy.state_dict(), os.path.join(run_dir, "final_policy.pt"))

    # Track global best
    best_global = os.path.join(MODEL_DIR, "reinforce_v2_best.pt")
    best_txt    = os.path.join(MODEL_DIR, "reinforce_v2_best_reward.txt")
    cur_best = -np.inf
    if os.path.exists(best_txt):
        try: cur_best = float(open(best_txt).read())
        except: pass
    if mean_r > cur_best:
        torch.save(policy.state_dict(), best_global)
        open(best_txt, "w").write(str(mean_r))
        print(f"  ★ New best REINFORCE v2 (reward={mean_r:.2f})")

    print(f"\n  ✓ Run {run_id} — reward={mean_r:.2f}±{std_r:.2f}  time={elapsed/60:.1f}m")

    return {
        "run_id":          run_id,
        "description":     config["description"],
        "learning_rate":   config["learning_rate"],
        "gamma":           config["gamma"],
        "hidden_dim":      config["hidden_dim"],
        "entropy_coef":    config["entropy_coef"],
        "batch_episodes":  config["batch_episodes"],
        "reward_clip":     config["reward_clip"],
        "normalize":       config["normalize"],
        "easy_env":        True,
        "reward_shaping":  True,
        "total_episodes":  total_episodes,
        "mean_reward":     round(mean_r, 2),
        "std_reward":      round(std_r, 2),
        "train_time_min":  round(elapsed / 60, 2),
    }


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",        type=int, default=None)
    parser.add_argument("--episodes",   type=int, default=TOTAL_EPISODES)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        print("\n  Smoke test — 32 episodes, Run 1")
        r = run_experiment(CONFIGS[0], total_episodes=32, seed=42)
        print(f"  ✓ Smoke test passed  reward={r['mean_reward']}")
        sys.exit(0)

    configs = CONFIGS if args.run is None else [CONFIGS[args.run - 1]]
    results = []

    print(f"\n  REINFORCE v2 Training  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Episodes/run: {args.episodes:,}  |  Easy env + reward shaping")

    for cfg in configs:
        results.append(run_experiment(cfg, args.episodes, args.seed))

    if len(results) > 1:
        print(f"\n{'='*62}  REINFORCE v2 SUMMARY")
        best_r = max(r["mean_reward"] for r in results)
        for r in results:
            m = " ★" if r["mean_reward"] == best_r else "  "
            print(f"  Run {r['run_id']}{m}  reward={r['mean_reward']:>8.2f}±{r['std_reward']:<7.2f}  {r['description']}")

        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)
        print(f"\n  Results → {CSV_PATH}")
        print(f"  Best model → {MODEL_DIR}/reinforce_v2_best.pt")
