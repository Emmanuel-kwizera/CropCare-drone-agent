"""
main.py — AfAlm Drone Crop Disease Management
----------------------------------------------
Entry point for running and visualising the environment.

Usage:
    # Random agent (no model — just shows the environment)
    python main.py --mode random

    # Run best saved model
    python main.py --mode best

    # Run specific algorithm
    python main.py --mode dqn
    python main.py --mode ppo
    python main.py --mode a2c
"""

import argparse
import os
import time
import numpy as np

from environment.custom_env import CropDroneEnv


def run_random(env: CropDroneEnv, episodes: int = 3):
    """Demonstrate environment with a random agent (no training)."""
    print("\n" + "="*55)
    print("  AfAlm — Random Agent Demo (no model)")
    print("="*55)
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        print(f"\n  Episode {ep + 1}")
        print(f"  {'─'*40}")
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            print(
                f"  Step {info['step']:>3d} | "
                f"Action: {info['last_action']:<20s} | "
                f"Reward: {reward:>7.1f} | "
                f"Fuel: {info['fuel']:>3d} | "
                f"Infected: {info['infected_count']:>2d} | "
                f"{info['last_event']}"
            )
            time.sleep(0.05)
        print(f"\n  Episode {ep+1} finished — Total reward: {total_reward:.1f}")
        print(f"  Treated: {info['treated_count']}  Dead: {info['dead_count']}")
    env.close()


def run_sb3_model(env: CropDroneEnv, algo: str):
    """Run a saved Stable-Baselines3 model (DQN or PPO)."""
    from stable_baselines3 import DQN, PPO

    model_map = {
        "dqn": (DQN, "models/dqn/best_model"),
        "ppo": (PPO, "models/pg/ppo_best"),
    }

    if algo not in model_map:
        print(f"  Unknown SB3 algorithm: {algo}. Choose from: dqn, ppo")
        return

    ModelClass, model_path = model_map[algo]
    path = model_path + ".zip"

    if not os.path.exists(path):
        print(f"  Model not found at {path}.")
        print(f"  Train it first:  python training/dqn_training.py" if algo == "dqn"
              else f"  Train it first:  python training/pg_training.py --algo ppo")
        return

    print(f"\n  Loading {algo.upper()} model from {path}")
    model = ModelClass.load(model_path, env=env)

    print("\n" + "="*60)
    print(f"  AfAlm — {algo.upper()} Agent")
    print("="*60)

    obs, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated
        print(
            f"  Step {info['step']:>3d} | "
            f"Action: {info['last_action']:<20s} | "
            f"Reward: {reward:>7.1f} | "
            f"Fuel: {info['fuel']:>3d} | "
            f"Infected: {info['infected_count']:>2d} | "
            f"{info['last_event']}"
        )

    print(f"\n  Episode finished — Total reward: {total_reward:.1f}")
    print(f"  Treated: {info['treated_count']}  Dead: {info['dead_count']}")
    env.close()


def run_reinforce_model(env: CropDroneEnv):
    """Run the saved REINFORCE policy (custom PyTorch .pt weights)."""
    import torch
    import torch.nn as nn

    model_path = "models/pg/reinforce_best.pt"
    if not os.path.exists(model_path):
        print(f"  Model not found at {model_path}.")
        print(f"  Train it first:  python training/pg_training.py --algo reinforce")
        env.close()
        return

    # Rebuild the same PolicyNetwork architecture used in training
    obs_dim   = env.observation_space.shape[0]   # 72
    n_actions = env.action_space.n               # 8
    hidden    = 256                              # default hidden dim

    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),  nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )
        def forward(self, x):
            return torch.softmax(self.net(x), dim=-1)

    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    print(f"\n  Loading REINFORCE model from {model_path}")
    print("\n" + "="*60)
    print(f"  AfAlm — REINFORCE Agent")
    print("="*60)

    obs, info = env.reset()
    total_reward = 0.0
    done = False

    with torch.no_grad():
        while not done:
            obs_t  = torch.FloatTensor(obs).unsqueeze(0)
            probs  = policy(obs_t)
            action = probs.argmax(dim=-1).item()   # greedy at inference
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            print(
                f"  Step {info['step']:>3d} | "
                f"Action: {info['last_action']:<20s} | "
                f"Reward: {reward:>7.1f} | "
                f"Fuel: {info['fuel']:>3d} | "
                f"Infected: {info['infected_count']:>2d} | "
                f"{info['last_event']}"
            )

    print(f"\n  Episode finished — Total reward: {total_reward:.1f}")
    print(f"  Treated: {info['treated_count']}  Dead: {info['dead_count']}")
    env.close()


def find_best_model():
    """
    Scan models/ for saved checkpoints and return the algo name
    of whichever exists. Checks reward files to pick the best one
    if multiple are trained.
    """
    candidates = {
        "dqn":       ("models/dqn/best_model.zip",      "models/dqn/best_reward.txt"),
        "ppo":       ("models/pg/ppo_best.zip",          "models/pg/ppo_best_reward.txt"),
        "reinforce": ("models/pg/reinforce_best.pt",     "models/pg/reinforce_best_reward.txt"),
    }

    best_algo   = None
    best_reward = -np.inf

    for algo, (model_path, reward_path) in candidates.items():
        if not os.path.exists(model_path):
            continue
        reward = -np.inf
        if os.path.exists(reward_path):
            with open(reward_path) as f:
                try:
                    reward = float(f.read().strip())
                except ValueError:
                    pass
        if reward > best_reward:
            best_reward = reward
            best_algo   = algo

    return best_algo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AfAlm Drone RL Environment Runner")
    parser.add_argument(
        "--mode",
        choices=["random", "best", "dqn", "ppo", "reinforce"],
        default="random",
        help="Agent mode: random (no model), best (auto-detect), dqn, ppo, reinforce",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to run (random mode only)"
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable Pygame rendering (terminal output only)"
    )
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"

    env = CropDroneEnv(
        grid_size=8,
        max_steps=200,
        max_fuel=100,
        max_payload=3,
        spread_interval=10,
        spread_prob=0.20,
        n_infected=5,
        render_mode=render_mode,
    )

    if args.mode == "random":
        run_random(env, episodes=args.episodes)

    elif args.mode == "best":
        best = find_best_model()
        if best is None:
            print("  No trained models found — running random agent instead.")
            run_random(env, episodes=args.episodes)
        else:
            print(f"  Auto-detected best model: {best.upper()}")
            if best == "reinforce":
                run_reinforce_model(env)
            else:
                run_sb3_model(env, best)

    elif args.mode == "reinforce":
        run_reinforce_model(env)

    else:
        run_sb3_model(env, args.mode)
