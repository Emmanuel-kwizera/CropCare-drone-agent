import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

# Cell state constants
HEALTHY  = 0
AT_RISK  = 1
FUNGAL   = 2
PEST     = 3
TREATED  = 4
DEAD     = 5
BASE     = 6

# Action constants
MOVE_NORTH   = 0
MOVE_SOUTH   = 1
MOVE_EAST    = 2
MOVE_WEST    = 3
APPLY_FUNGICIDE = 4
APPLY_PESTICIDE = 5
SCAN         = 6
RETURN_BASE  = 7

ACTION_NAMES = {
    0: "Move North",
    1: "Move South",
    2: "Move East",
    3: "Move West",
    4: "Apply Fungicide",
    5: "Apply Pesticide",
    6: "Scan",
    7: "Return to Base",
}

class CropDroneEnv(gym.Env):
    """
    AfAlm Drone Crop Disease Management Environment
    ------------------------------------------------
    An 8x8 farm grid where a drone agent patrols crops,
    detects disease (fungal / pest), and applies the correct
    chemical treatment before infections spread and kill plants.

    Observation space (72-dim float32 Box):
        - Grid state      : 64 values (flattened 8x8, normalised 0-1)
        - Drone position  : 2  values (row, col, normalised 0-1)
        - Fuel level      : 1  value  (normalised 0-1)
        - Payload         : 2  values (fungicide count, pesticide count, normalised 0-3)
        - Time step       : 1  value  (normalised 0-1)
        - Infection counts: 2  values (known fungal, known pest, normalised)

    Action space (Discrete 8):
        0-3 : Move N/S/E/W
        4   : Apply fungicide on current cell
        5   : Apply pesticide on current cell
        6   : Scan current + adjacent cells (reveals hidden disease type)
        7   : Return to base (recharge fuel + reload payload)

    Reward structure:
        +20  correct treatment applied
        +5   scan reveals a new infected cell
        +100 all infections cleared (terminal bonus)
        -0.5 per step (efficiency pressure)
        -15  wrong chemical applied
        -30  cell dies (infection untreated too long)
        -50  drone crashes (fuel = 0 away from base)
        -10  treating healthy / already-treated cell
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 200,
        max_fuel: int = 100,
        max_payload: int = 3,
        spread_interval: int = 10,
        spread_prob: float = 0.20,
        n_infected: int = 5,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.grid_size      = grid_size
        self.max_steps      = max_steps
        self.max_fuel       = max_fuel
        self.max_payload    = max_payload
        self.spread_interval = spread_interval
        self.spread_prob    = spread_prob
        self.n_infected     = n_infected
        self.render_mode    = render_mode

        # Observation: 64 grid + 2 pos + 1 fuel + 2 payload + 1 time + 2 inf_counts
        obs_dim = grid_size * grid_size + 2 + 1 + 2 + 1 + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)

        # Internal state (initialised in reset)
        self.grid             = None
        self.hidden_disease   = None   # true disease type hidden until scanned
        self.scanned          = None   # bool mask — has drone scanned this cell?
        self.drone_row        = 0
        self.drone_col        = 0
        self.fuel             = max_fuel
        self.fungicide        = max_payload
        self.pesticide        = max_payload
        self.step_count       = 0
        self.score            = 0.0
        self.cell_age         = None   # tracks how long each cell has been infected
        self.last_action      = None
        self.last_reward      = 0.0
        self.last_event       = ""

        # Renderer (lazy init)
        self._renderer = None

        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self._np_random = np.random.default_rng()

    # ──────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        rng = self.np_random

        # Build blank grid
        self.grid = np.full((self.grid_size, self.grid_size), HEALTHY, dtype=np.int32)
        self.grid[0, 0] = BASE

        # Hidden disease map (true disease, unknown to agent until scanned)
        self.hidden_disease = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.scanned        = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.cell_age       = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place infected cells at random non-base positions
        available = [(r, c) for r in range(self.grid_size)
                             for c in range(self.grid_size)
                             if not (r == 0 and c == 0)]
        rng.shuffle(available)

        at_risk_count  = min(self.n_infected + 2, len(available))
        infected_slots = available[:self.n_infected]
        at_risk_slots  = available[self.n_infected: at_risk_count]

        for r, c in infected_slots:
            disease = FUNGAL if rng.random() < 0.6 else PEST
            self.hidden_disease[r, c] = disease
            self.grid[r, c] = AT_RISK   # visible as at-risk (sensor flag)

        for r, c in at_risk_slots:
            self.grid[r, c] = AT_RISK

        # Drone starts at base
        self.drone_row  = 0
        self.drone_col  = 0
        self.fuel       = self.max_fuel
        self.fungicide  = self.max_payload
        self.pesticide  = self.max_payload
        self.step_count = 0
        self.score      = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.last_event  = "Episode started"

        return self._get_obs(), self._get_info()

    # ──────────────────────────────────────────────
    # STEP
    # ──────────────────────────────────────────────
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.grid is not None, "Call reset() before step()"
        self.last_action = action
        reward = -0.5   # per-step efficiency penalty
        terminated = False
        truncated  = False
        event = ""

        r, c = self.drone_row, self.drone_col

        # ── MOVEMENT ──────────────────────────────
        if action in (MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST):
            dr, dc = {
                MOVE_NORTH: (-1, 0),
                MOVE_SOUTH: ( 1, 0),
                MOVE_EAST:  ( 0, 1),
                MOVE_WEST:  ( 0,-1),
            }[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                self.drone_row, self.drone_col = nr, nc
                self.fuel -= 1
            else:
                reward -= 1.0  # hit boundary
                event = "Boundary hit"

        # ── APPLY FUNGICIDE ───────────────────────
        elif action == APPLY_FUNGICIDE:
            cell_state = self.grid[r, c]
            if self.fungicide <= 0:
                reward -= 5.0
                event = "No fungicide remaining!"
            elif cell_state in (HEALTHY, TREATED, BASE):
                reward -= 10.0
                event = "Wasted fungicide on healthy cell"
            elif cell_state == AT_RISK:
                # Check true disease
                true_disease = self.hidden_disease[r, c]
                if true_disease == FUNGAL:
                    self.grid[r, c] = TREATED
                    self.fungicide -= 1
                    self.fuel -= 2
                    reward += 20.0
                    event = "Fungicide applied correctly! +20"
                elif true_disease == PEST:
                    self.fungicide -= 1
                    self.fuel -= 2
                    reward -= 15.0
                    event = "Wrong chemical — pest needs pesticide! -15"
                else:
                    # False positive sensor flag — no real disease
                    self.grid[r, c] = HEALTHY
                    self.fungicide -= 1
                    reward -= 10.0
                    event = "No disease found — false sensor flag"
            elif cell_state in (FUNGAL, ):
                # Cell already confirmed fungal (post-scan)
                self.grid[r, c] = TREATED
                self.fungicide -= 1
                self.fuel -= 2
                reward += 20.0
                event = "Fungicide applied correctly! +20"
            elif cell_state == PEST:
                self.fungicide -= 1
                self.fuel -= 2
                reward -= 15.0
                event = "Wrong chemical — pest cell! -15"
            else:
                reward -= 10.0
                event = "Nothing to treat here"

        # ── APPLY PESTICIDE ───────────────────────
        elif action == APPLY_PESTICIDE:
            cell_state = self.grid[r, c]
            if self.pesticide <= 0:
                reward -= 5.0
                event = "No pesticide remaining!"
            elif cell_state in (HEALTHY, TREATED, BASE):
                reward -= 10.0
                event = "Wasted pesticide on healthy cell"
            elif cell_state == AT_RISK:
                true_disease = self.hidden_disease[r, c]
                if true_disease == PEST:
                    self.grid[r, c] = TREATED
                    self.pesticide -= 1
                    self.fuel -= 2
                    reward += 20.0
                    event = "Pesticide applied correctly! +20"
                elif true_disease == FUNGAL:
                    self.pesticide -= 1
                    self.fuel -= 2
                    reward -= 15.0
                    event = "Wrong chemical — fungal needs fungicide! -15"
                else:
                    self.grid[r, c] = HEALTHY
                    self.pesticide -= 1
                    reward -= 10.0
                    event = "No disease found — false sensor flag"
            elif cell_state == PEST:
                self.grid[r, c] = TREATED
                self.pesticide -= 1
                self.fuel -= 2
                reward += 20.0
                event = "Pesticide applied correctly! +20"
            elif cell_state == FUNGAL:
                self.pesticide -= 1
                self.fuel -= 2
                reward -= 15.0
                event = "Wrong chemical — fungal cell! -15"
            else:
                reward -= 10.0
                event = "Nothing to treat here"

        # ── SCAN ──────────────────────────────────
        elif action == SCAN:
            self.fuel -= 1
            newly_found = 0
            scan_targets = [(r, c)]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr2, nc2 = r+dr, c+dc
                if 0 <= nr2 < self.grid_size and 0 <= nc2 < self.grid_size:
                    scan_targets.append((nr2, nc2))
            for sr, sc in scan_targets:
                if not self.scanned[sr, sc]:
                    self.scanned[sr, sc] = True
                    if self.hidden_disease[sr, sc] in (FUNGAL, PEST):
                        # Reveal true disease on grid
                        self.grid[sr, sc] = self.hidden_disease[sr, sc]
                        newly_found += 1
            if newly_found > 0:
                reward += newly_found * 5.0
                event = f"Scan revealed {newly_found} infected cell(s)! +{newly_found*5}"
            else:
                event = "Scan: no new infections found"

        # ── RETURN TO BASE ────────────────────────
        elif action == RETURN_BASE:
            if r == 0 and c == 0:
                event = "Already at base"
            else:
                # Costs 5 steps worth of time but resets fuel + payload
                self.step_count += 4  # extra cost
                self.drone_row = 0
                self.drone_col = 0
                self.fuel      = self.max_fuel
                self.fungicide = self.max_payload
                self.pesticide = self.max_payload
                reward -= 5.0   # penalty for trip time
                event = "Returned to base — recharged and reloaded"

        # ── FUEL DEPLETION CHECK ──────────────────
        self.fuel = max(0, self.fuel)
        if self.fuel == 0 and not (self.drone_row == 0 and self.drone_col == 0):
            reward -= 50.0
            terminated = True
            event = "CRASH — fuel depleted away from base!"

        # ── DISEASE SPREAD ────────────────────────
        self.step_count += 1
        if self.step_count % self.spread_interval == 0:
            reward += self._spread_disease()

        # ── DEAD CELL CHECK ───────────────────────
        reward += self._age_infected_cells()

        # ── TERMINAL: ALL CLEAR ───────────────────
        infected_remaining = np.sum(
            (self.grid == AT_RISK) | (self.grid == FUNGAL) | (self.grid == PEST)
        )
        if infected_remaining == 0 and not terminated:
            reward += 100.0
            terminated = True
            event = "All infections cleared! Mission complete! +100"

        # ── TERMINAL: TOO MANY DEAD ───────────────
        dead_count = np.sum(self.grid == DEAD)
        if dead_count > int(0.35 * self.grid_size * self.grid_size):
            reward -= 100.0
            terminated = True
            event = "Too many crops dead — mission failed!"

        # ── TRUNCATION ────────────────────────────
        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            treated = np.sum(self.grid == TREATED)
            reward += treated * 2.0  # partial credit
            event = f"Time limit reached — {treated} cells treated"

        self.score += reward
        self.last_reward = reward
        self.last_event  = event

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ──────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────
    def _spread_disease(self) -> float:
        """Infected cells may spread to adjacent healthy/at-risk cells."""
        penalty = 0.0
        spreaders = list(zip(*np.where(
            (self.grid == FUNGAL) | (self.grid == PEST) | (self.grid == AT_RISK)
        )))
        rng = self.np_random
        for (r, c) in spreaders:
            if rng.random() < self.spread_prob:
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.grid[nr, nc] in (HEALTHY,):
                            neighbors.append((nr, nc))
                if neighbors:
                    nr, nc = neighbors[int(rng.random() * len(neighbors))]
                    disease = self.hidden_disease[r, c] if self.hidden_disease[r, c] != 0 else (
                        FUNGAL if rng.random() < 0.6 else PEST
                    )
                    self.hidden_disease[nr, nc] = disease
                    self.grid[nr, nc] = AT_RISK
                    penalty -= 2.0
        return penalty

    def _age_infected_cells(self) -> float:
        """Cells infected too long die and generate penalties."""
        penalty = 0.0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] in (AT_RISK, FUNGAL, PEST):
                    self.cell_age[r, c] += 1
                    if self.cell_age[r, c] >= 30:  # 30 steps untreated → dead
                        self.grid[r, c] = DEAD
                        penalty -= 30.0
                elif self.grid[r, c] == TREATED:
                    self.cell_age[r, c] = 0
        return penalty

    def _get_obs(self) -> np.ndarray:
        grid_flat = self.grid.flatten().astype(np.float32) / 6.0  # normalise 0-1
        drone_pos = np.array([
            self.drone_row / (self.grid_size - 1),
            self.drone_col / (self.grid_size - 1),
        ], dtype=np.float32)
        fuel_norm     = np.array([self.fuel / self.max_fuel], dtype=np.float32)
        payload_norm  = np.array([
            self.fungicide / self.max_payload,
            self.pesticide / self.max_payload,
        ], dtype=np.float32)
        time_norm = np.array([self.step_count / self.max_steps], dtype=np.float32)
        known_fungal = float(np.sum(self.grid == FUNGAL)) / (self.grid_size ** 2)
        known_pest   = float(np.sum(self.grid == PEST))   / (self.grid_size ** 2)
        inf_counts   = np.array([known_fungal, known_pest], dtype=np.float32)

        return np.concatenate([grid_flat, drone_pos, fuel_norm, payload_norm, time_norm, inf_counts])

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step":           self.step_count,
            "score":          self.score,
            "fuel":           self.fuel,
            "fungicide":      self.fungicide,
            "pesticide":      self.pesticide,
            "drone_pos":      (self.drone_row, self.drone_col),
            "infected_count": int(np.sum((self.grid == AT_RISK) | (self.grid == FUNGAL) | (self.grid == PEST))),
            "treated_count":  int(np.sum(self.grid == TREATED)),
            "dead_count":     int(np.sum(self.grid == DEAD)),
            "last_action":    ACTION_NAMES.get(self.last_action, "None"),
            "last_reward":    self.last_reward,
            "last_event":     self.last_event,
        }

    # ──────────────────────────────────────────────
    # RENDER
    # ──────────────────────────────────────────────
    def render(self):
        if self._renderer is None:
            from environment.rendering import FarmRenderer
            self._renderer = FarmRenderer(self.grid_size)
        return self._renderer.render(
            grid        = self.grid,
            drone_row   = self.drone_row,
            drone_col   = self.drone_col,
            fuel        = self.fuel,
            max_fuel    = self.max_fuel,
            fungicide   = self.fungicide,
            pesticide   = self.pesticide,
            step_count  = self.step_count,
            score       = self.score,
            last_event  = self.last_event,
            render_mode = self.render_mode,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
