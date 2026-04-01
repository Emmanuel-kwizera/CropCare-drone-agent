# AfAlm Drone Crop Disease Management — RL Summative Assignment

**Student:** Emmanuel Kwizera

## Description

A Deep Reinforcement Learning project where an autonomous drone agent learns to effectively patrol and manage agricultural diseases across an 8x8 farm grid. The agent must successfully navigate, **scan** for hidden *Fungal* or *Pest* infections, and strategically apply the correct treatment (Fungicide or Pesticide) while heavily constrained by a fuel limit and physical payloads.

This project simulates complex real-world resource management and diagnostic routing tasks under partial observability, solving for high-terminal penalties like drone crashes and plant death.

## Demo Video

[Watch the demo on YouTube (https://youtu.be/qlMvnwkym68)]

## Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Random Agent (No trained model needed)
```bash
python main.py --mode random
```
Demonstrates the high-fidelity isometric pygame visualization with an agent taking random actions to show the environment mechanics.

### Run Best Trained Model
```bash
python main.py --mode best
```
Automatically scans the `models/` directory, selects the policy network with the highest evaluation reward, and runs the simulation with pygame visualization. 

You can also explicitly specify and run specific versions of the trained agents (which maps to the exact phase 2 or phase 3 model) like this:
```bash
python main.py --mode ppo_v2
python main.py --mode ppo_v3
python main.py --mode dqn
```

### Train Models
```bash
python training/improved/v3/dqn_v3.py
python training/improved/v3/pg_v3.py
```
Each training script executes massive hyperparameter suites, leveraging Stable Baselines 3 (DQN, PPO) and Custom PyTorch scripts (REINFORCE), while logging directly to TensorBoard.

### Generate Plots
```bash
python generate_plots.py
```
Automatically reads the `logs/` directory TensorBoard events and generates visual comparison subplots saved to `plots/`.

## Environment

### Observation Space (72 continuous dimensions, normalized 0-1)
| Index | Feature | Description |
|-------|---------|-------------|
| 0-63 | Grid State | Flattened 8x8 array detailing the operational state of every cell (Healthy, At-Risk, Fungal, Pest, Treated, Dead, Base). |
| 64-65 | Drone Position | Normalized Row and Column (y,x) coordinates. |
| 66 | Fuel Level | Internal drone battery/fuel level. |
| 67-68 | Payload Counts | Remaining physical units of Fungicide and Pesticide. |
| 69 | Time Step | Episode step relative to Max Steps. |
| 70-71 | Infection Ratios | Percentage of known fungal and pest infections left on the grid. |

### Action Space (Discrete 8)
- `0-3`: Move (North, South, East, West)
- `4`: Apply Fungicide payload
- `5`: Apply Pesticide payload
- `6`: Scan (Deep diagnostic sensor to reveal hidden disease states on adjacent risk tiles)
- `7`: Return to Base (Jump to origin to instantly refill fuel/payload sequences at a high time cost)

### Rewards
- **Correct chemical applied:** +20.0
- **Scan uncovers hidden disease:** +5.0 per cell
- **Time step penalty:** -0.5 (efficiency pressure)
- **Wrong chemical applied:** -15.0
- **Applying chemicals to healthy crops:** -10.0
- **Cell dies (Untreated too long):** -30.0
- **Drone Crash (Fuel depleted away from base):** -50.0
- **Terminal Win (All clear):** +100.0

### Terminal Conditions
- **Victory**: All infections successfully scanned and treated.
- **Drone Crash**: Fuel drops to 0 while the drone is away from the base helipad.
- **Crop Failure**: Over 35% of the farm grid dies due to prolonged lack of treatment.
- **Timeout**: Maximum episode steps reached.

## Algorithms Implemented
1. **PPO** (Proximal Policy Optimization) — Stable Baselines 3 (`VecNormalize` augmented)
2. **DQN** (Deep Q-Network) — Stable Baselines 3 (Experience Replay, High Gradient Steps)
3. **REINFORCE** (Monte Carlo Policy Gradient) — Custom PyTorch implementation

## Project Structure
```text
CropCare-drone-agent/
├── design/                    # Interactive mockup explaining the intended 3D design
│   └── AfAlm_Drone_Treatment_3d.html
├── environment/
│   ├── custom_env.py          # Custom Gymnasium environment logic
│   └── rendering.py           # High-fidelity isometric Pygame visualization
├── training/
│   ├── dqn_training.py        # Standard run phase 1 algorithms
│   ├── pg_training.py         
│   └── improved/              # Phase 2 & 3 Advanced Reward Shaping configurations
├── models/                    # Directory for all trained agent versions
│   ├── dqn/                   # Phase 1: Baseline DQN experiments
│   ├── pg/                    # Phase 1: Baseline REINFORCE and PPO experiments
│   └── improved/              # Phase 2 & 3: Advanced architectures
│       ├── dqn/               # Phase 2 DQN (with reward shaping)
│       ├── ppo_v2/            # Phase 2 PPO (with reward shaping)
│       ├── reinforce_v2/      # Phase 2 REINFORCE
│       └── v3/                # Phase 3 Final Models (VecNormalize, best proxies)
│           ├── dqn/           # v3 High Gradient-rate DQN
│           ├── ppo/           # v3 Best Performing Model (-36 reward)
│           └── reinforce/     # v3 Deep Capacity network
├── logs/                      # Raw Tensorboard event tracking
├── plots/                     # Auto-generated hyperparameter line plots
├── main.py                    # Environment executable (play visually)
├── generate_plots.py          # Script extracting data for final report graphics
├── requirements.txt
└── README.md
```
