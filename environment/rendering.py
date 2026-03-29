import pygame
import numpy as np
import math
from typing import Optional

# ── Cell state constants (mirrors custom_env.py) ──
HEALTHY  = 0
AT_RISK  = 1
FUNGAL   = 2
PEST     = 3
TREATED  = 4
DEAD     = 5
BASE     = 6

# ── Colour palette ────────────────────────────────
BG_COLOR       = (18,  18,  28)
GRID_LINE      = (35,  35,  50)

SOIL_COLORS    = [
    (74, 44, 14), (92, 56, 18), (62, 34,  8),
    (107,68, 24), (82, 48, 16),
]
SOIL_DARK      = (28, 16,  4)

CELL_COLORS = {
    HEALTHY:  (58, 140, 32),
    AT_RISK:  (184,168, 32),
    FUNGAL:   (160,  60,180),
    PEST:     (200,  50, 30),
    TREATED:  ( 42, 130, 80),
    DEAD:     ( 55,  45, 35),
    BASE:     ( 30,  50, 80),
}

HUD_BG         = (10,  14,  22, 220)
HUD_TEXT       = (200, 210, 220)
HUD_MUTED      = (110, 130, 150)
FUEL_GREEN     = ( 45, 170,  85)
FUEL_YELLOW    = (180, 180,  40)
FUEL_RED       = (200,  50,  40)
SCORE_GREEN    = ( 93, 232, 138)
FUNGAL_COL     = (200, 136, 255)
PEST_COL       = (255, 136,  68)
INFECTED_COL   = (238,  68,  68)
TRAIL_COL      = ( 68, 170, 255, 120)
WARNING_COL    = (255, 220,   0)


class FarmRenderer:
    """
    Pygame-based isometric renderer for the CropDroneEnv.
    Draws an 8×8 farm grid in isometric projection with:
      - Tilled soil tiles with furrow lines
      - Crop sprites that change appearance by disease state
      - Drone with spinning rotors, spray tank, LED indicators
      - Spray particle effects when treatment is applied
      - Drone trail showing recent patrol path
      - HUD panel: step, score, fuel bar, payload counts, event log
    """

    WINDOW_W   = 900
    WINDOW_H   = 660
    TILE_W     = 80      # isometric tile full width
    TILE_H     = 40      # isometric tile full height
    TILE_DEPTH = 14      # pixel depth of raised soil bed
    FPS        = 30

    def __init__(self, grid_size: int = 8):
        pygame.init()
        pygame.display.set_caption("AfAlm — Drone Crop Disease Management")

        self.grid_size  = grid_size
        self.screen     = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
        self.clock      = pygame.time.Clock()
        self.font_sm    = pygame.font.SysFont("monospace", 11)
        self.font_md    = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_lg    = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_hud   = pygame.font.SysFont("monospace", 12)
        self.tick       = 0

        # Isometric origin — centre of grid projection
        self.origin_x = self.WINDOW_W // 2
        self.origin_y = 80

        # Particle system
        self.particles  = []

        # Drone trail
        self.trail      = []

        # Event log lines: list of (text, colour, age)
        self.log_lines  = []

    # ──────────────────────────────────────────────
    # COORDINATE HELPERS
    # ──────────────────────────────────────────────
    def iso(self, col: int, row: int) -> tuple:
        """Convert grid (col, row) → screen (x, y) isometric centre."""
        x = self.origin_x + (col - row) * (self.TILE_W // 2)
        y = self.origin_y + (col + row) * (self.TILE_H // 2)
        return (x, y)

    # ──────────────────────────────────────────────
    # TILE DRAWING
    # ──────────────────────────────────────────────
    def _draw_tile(self, surface, col, row, state):
        x, y = self.iso(col, row)
        tw, th, td = self.TILE_W // 2, self.TILE_H // 2, self.TILE_DEPTH

        soil_idx = (row * 5 + col * 3) % len(SOIL_COLORS)
        top_col  = SOIL_COLORS[soil_idx]

        # Darken sides
        left_col  = tuple(max(0, v - 30) for v in top_col)
        right_col = tuple(max(0, v - 18) for v in top_col)

        # Top face (diamond)
        top_pts = [(x, y), (x+tw, y+th), (x, y+self.TILE_H), (x-tw, y+th)]
        pygame.draw.polygon(surface, top_col, top_pts)
        pygame.draw.polygon(surface, GRID_LINE, top_pts, 1)

        # Left face
        left_pts = [(x-tw, y+th), (x, y+self.TILE_H),
                    (x, y+self.TILE_H+td), (x-tw, y+th+td)]
        pygame.draw.polygon(surface, left_col, left_pts)
        pygame.draw.polygon(surface, GRID_LINE, left_pts, 1)

        # Right face
        right_pts = [(x, y+self.TILE_H), (x+tw, y+th),
                     (x+tw, y+th+td), (x, y+self.TILE_H+td)]
        pygame.draw.polygon(surface, right_col, right_pts)
        pygame.draw.polygon(surface, GRID_LINE, right_pts, 1)

        # Furrow lines on top face
        for i in range(3):
            fy = y + 8 + i * 10
            fxl = x - tw + 4
            fxr = x + tw - 4
            # Clip to diamond roughly
            pygame.draw.line(surface, SOIL_DARK,
                             (int(fxl + i*2), int(fy)),
                             (int(fxr - i*2), int(fy)), 1)

    # ──────────────────────────────────────────────
    # CROP DRAWING
    # ──────────────────────────────────────────────
    def _draw_crop(self, surface, cx, cy, state, offset_seed=0):
        t = self.tick
        sway = math.sin(t * 0.04 + offset_seed * 0.8) * 2.0

        if state == HEALTHY:
            stem_col  = (45, 110, 22)
            leaf_cols = [(58, 140, 32), (74, 170, 40), (45, 110, 22)]
            stem_h    = 28 + int(math.sin(offset_seed * 3.7) * 4)
            # Stem
            pygame.draw.line(surface, stem_col,
                             (cx, cy), (int(cx + sway), cy - stem_h), 2)
            # Leaves (3 pairs)
            leaf_data = [
                (cx - 12 + sway*0.4, cy - 10),
                (cx + 12 + sway*0.5, cy - 16),
                (cx - 8  + sway*0.3, cy - 22),
            ]
            for i, (lx, ly) in enumerate(leaf_data):
                lc = leaf_cols[i % len(leaf_cols)]
                pts = [
                    (int(cx + sway * 0.4), int(ly + 5)),
                    (int(lx), int(ly)),
                    (int(lx + (4 if i % 2 == 0 else -4)), int(ly - 6)),
                ]
                if len(pts) >= 3:
                    pygame.draw.polygon(surface, lc, pts)
                    pygame.draw.polygon(surface, (30, 80, 15), pts, 1)
            # Tassel
            for ti in range(4):
                ta = (ti / 4) * math.pi * 0.9 - 0.3
                tx2 = int(cx + sway + math.cos(ta) * 6)
                ty2 = int(cy - stem_h - math.sin(ta) * 7)
                pygame.draw.line(surface, (138, 170, 48),
                                 (int(cx + sway), cy - stem_h), (tx2, ty2), 1)
            # Corn ear
            ear_x = int(cx + 9 + sway * 0.5)
            ear_y = cy - int(stem_h * 0.55)
            pygame.draw.ellipse(surface, (212, 200, 64),
                                (ear_x - 4, ear_y - 8, 8, 16))
            pygame.draw.ellipse(surface, (90, 128, 16),
                                (ear_x - 2, ear_y - 10, 5, 12))

        elif state == AT_RISK:
            stem_col = (138, 128, 32)
            stem_h   = 22
            pygame.draw.line(surface, stem_col,
                             (cx, cy), (int(cx + sway), cy - stem_h), 2)
            for i, angle in enumerate([0.4, -0.5, 0.3]):
                lx = int(cx + math.cos(angle) * 14)
                ly = cy - 8 - i * 6
                pygame.draw.ellipse(surface, (184, 168, 32),
                                    (lx - 8, ly - 4, 16, 8))
            # Pulsing warning ring
            pulse = 0.5 + 0.5 * math.sin(t * 0.12)
            alpha = int(80 + pulse * 120)
            ring_surf = pygame.Surface((44, 22), pygame.SRCALPHA)
            pygame.draw.ellipse(ring_surf, (*WARNING_COL, alpha), (0, 0, 44, 22), 2)
            surface.blit(ring_surf, (cx - 22, cy - 2))

        elif state == FUNGAL:
            pygame.draw.line(surface, (90, 48, 106),
                             (cx, cy), (cx, cy - 20), 2)
            # Spore cluster
            for i in range(5):
                angle = (i / 5) * math.pi * 2 + t * 0.025
                sx2   = int(cx + math.cos(angle) * 11)
                sy2   = int(cy - 12 + math.sin(angle) * 5)
                radius = int(5 + math.sin(t * 0.07 + i) * 1.5)
                col_a  = int(160 + math.sin(t * 0.06 + i) * 40)
                pygame.draw.circle(surface, (col_a, 60, 200), (sx2, sy2), radius)
            # Ground glow
            glow_surf = pygame.Surface((38, 18), pygame.SRCALPHA)
            pygame.draw.ellipse(glow_surf, (150, 40, 180, 55), (0, 0, 38, 18))
            surface.blit(glow_surf, (cx - 19, cy - 4))

        elif state == PEST:
            pygame.draw.line(surface, (122, 48, 32),
                             (cx, cy), (cx, cy - 18), 2)
            for i, angle in enumerate([0.4, -0.5, 0.6]):
                lx = int(cx + math.cos(angle) * 12)
                ly = cy - 7 - i * 5
                pygame.draw.ellipse(surface, (139, 64, 48),
                                    (lx - 9, ly - 4, 18, 9))
            # Red lesion spots
            for i in range(4):
                sx2 = int(cx + (i - 2) * 5 + math.sin(offset_seed + i) * 3)
                sy2 = int(cy - 5 - i * 4)
                pygame.draw.circle(surface, (210, 50, 20), (sx2, sy2), 3)
            # Ground glow
            glow_surf = pygame.Surface((36, 16), pygame.SRCALPHA)
            pygame.draw.ellipse(glow_surf, (180, 30, 0, 48), (0, 0, 36, 16))
            surface.blit(glow_surf, (cx - 18, cy - 4))

        elif state == TREATED:
            stem_col  = (28, 108, 64)
            leaf_cols = [(42, 130, 78), (55, 155, 90)]
            stem_h    = 26
            pygame.draw.line(surface, stem_col,
                             (cx, cy), (int(cx + sway), cy - stem_h), 2)
            for i, angle in enumerate([0.5, -0.5, 0.4, -0.4]):
                lx = int(cx + math.cos(angle) * 12 + sway * 0.3)
                ly = cy - 8 - i * 5
                lc = leaf_cols[i % 2]
                pygame.draw.ellipse(surface, lc, (lx - 8, ly - 4, 16, 9))
            # Subtle green pulse ring
            pulse2 = 0.35 + 0.25 * math.sin(t * 0.08)
            ring_surf2 = pygame.Surface((44, 22), pygame.SRCALPHA)
            pygame.draw.ellipse(ring_surf2, (80, 200, 120, int(pulse2 * 180)),
                                (0, 0, 44, 22), 2)
            surface.blit(ring_surf2, (cx - 22, cy - 2))

        elif state == DEAD:
            # Wilted bare stalk
            pygame.draw.line(surface, (70, 55, 40),
                             (cx, cy), (int(cx - 4 + sway), cy - 14), 2)
            pygame.draw.line(surface, (60, 45, 30),
                             (cx, cy - 5), (int(cx - 9), cy - 12), 1)
            pygame.draw.line(surface, (60, 45, 30),
                             (cx, cy - 8), (int(cx + 7), cy - 14), 1)

        elif state == BASE:
            # Helipad
            pygame.draw.ellipse(surface, (38, 58, 80), (cx - 18, cy - 4, 36, 18))
            pygame.draw.ellipse(surface, (62, 90, 120), (cx - 18, cy - 4, 36, 18), 2)
            pygame.draw.ellipse(surface, (255, 200, 0), (cx - 12, cy - 2, 24, 12), 2)
            lbl = self.font_sm.render("BASE", True, (100, 160, 210))
            surface.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2 + 2))

    # ──────────────────────────────────────────────
    # DRONE DRAWING
    # ──────────────────────────────────────────────
    def _draw_drone(self, surface, cx, cy):
        t = self.tick

        # Shadow
        shadow_surf = pygame.Surface((40, 20), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 60), (0, 0, 40, 20))
        surface.blit(shadow_surf, (cx - 20, cy + 10))

        # Arms
        arm_ends = []
        for i, angle_deg in enumerate([45, 135, 225, 315]):
            angle = math.radians(angle_deg)
            ax = int(cx + math.cos(angle) * 20)
            ay = int(cy + math.sin(angle) * 11)
            arm_ends.append((ax, ay))
            pygame.draw.line(surface, (55, 55, 55), (cx, cy), (ax, ay), 3)
            # Motor housing
            pygame.draw.circle(surface, (75, 75, 75), (ax, ay), 5)
            # Spinning prop (blur effect with 2 semi-transparent lines)
            pa = t * 0.35 * (1 if i < 2 else -1)
            for pi in range(2):
                prop_a = pa + pi * math.pi
                px1 = int(ax + math.cos(prop_a) * 11)
                py1 = int(ay + math.sin(prop_a) * 5)
                px2 = int(ax - math.cos(prop_a) * 11)
                py2 = int(ay - math.sin(prop_a) * 5)
                prop_surf = pygame.Surface(
                    (surface.get_width(), surface.get_height()), pygame.SRCALPHA
                )
                pygame.draw.line(prop_surf, (200, 200, 200, 160), (px1, py1), (px2, py2), 2)
                surface.blit(prop_surf, (0, 0))

        # Body
        pygame.draw.ellipse(surface, (28, 28, 28), (cx - 11, cy - 6, 22, 12))
        pygame.draw.ellipse(surface, (200, 34, 34), (cx - 7, cy - 5, 14, 9))

        # Camera dome
        pygame.draw.circle(surface, (18, 18, 18), (cx, cy + 5), 5)
        pygame.draw.circle(surface, (0,  26, 68), (cx, cy + 6), 3)

        # Spray tank
        pygame.draw.rect(surface, (34, 85, 170),
                         (cx - 3, cy + 9, 6, 12), border_radius=2)
        pygame.draw.rect(surface, (50, 50, 50),
                         (cx - 2, cy + 20, 4, 4), border_radius=1)

        # LEDs
        blink_g = math.sin(t * 0.12) > 0
        led_g = (0, 255, 136) if blink_g else (0, 80, 40)
        blink_r = math.sin(t * 0.09 + math.pi) > 0
        led_r = (255, 34, 0) if blink_r else (80, 10, 0)
        pygame.draw.circle(surface, led_g,  (cx + 8, cy - 2), 2)
        pygame.draw.circle(surface, led_r,  (cx - 8, cy - 2), 2)

        # Direction arrow
        pygame.draw.polygon(surface, (220, 220, 220),
                            [(cx, cy - 8), (cx + 4, cy - 4), (cx - 4, cy - 4)])

    # ──────────────────────────────────────────────
    # PARTICLE SYSTEM
    # ──────────────────────────────────────────────
    def spawn_spray(self, sx, sy, ptype="fungal"):
        col = (200, 80, 255) if ptype == "fungal" else (255, 100, 30)
        for _ in range(24):
            angle  = np.random.uniform(0, math.pi * 2)
            speed  = np.random.uniform(1.5, 4.5)
            self.particles.append({
                "x": sx, "y": sy,
                "vx": math.cos(angle) * speed * 0.6,
                "vy": math.sin(angle) * speed * 0.35 - 1.5,
                "life": 1.0,
                "decay": np.random.uniform(0.022, 0.04),
                "r": np.random.uniform(2, 5),
                "col": col,
                "type": "drop",
            })
        # Expanding ring
        self.particles.append({
            "x": sx, "y": sy + 12, "scale": 1.0,
            "life": 1.0, "decay": 0.035,
            "col": col, "type": "ring",
        })

    def _update_particles(self, surface):
        alive = []
        for p in self.particles:
            if p["type"] == "drop":
                p["x"]  += p["vx"]
                p["y"]  += p["vy"]
                p["vy"] += 0.12   # gravity
                p["life"] -= p["decay"]
                if p["life"] > 0:
                    alpha = int(p["life"] * 220)
                    r     = int(p["r"] * p["life"])
                    if r > 0:
                        drop_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
                        pygame.draw.circle(drop_surf, (*p["col"], alpha),
                                           (r + 1, r + 1), r)
                        surface.blit(drop_surf, (int(p["x"]) - r - 1, int(p["y"]) - r - 1))
                    alive.append(p)
            elif p["type"] == "ring":
                p["life"]  -= p["decay"]
                p["scale"] += 0.12
                if p["life"] > 0:
                    w = int(60 * p["scale"])
                    h = int(30 * p["scale"])
                    ring_s = pygame.Surface((w + 4, h + 4), pygame.SRCALPHA)
                    alpha  = int(p["life"] * 160)
                    pygame.draw.ellipse(ring_s, (*p["col"], alpha),
                                        (2, 2, w, h), 2)
                    surface.blit(ring_s,
                                 (int(p["x"]) - w // 2 - 2, int(p["y"]) - h // 2 - 2))
                    alive.append(p)
        self.particles = alive

    # ──────────────────────────────────────────────
    # TRAIL
    # ──────────────────────────────────────────────
    def _draw_trail(self, surface):
        if len(self.trail) < 2:
            return
        for i in range(1, len(self.trail)):
            alpha = int(160 * (i / len(self.trail)))
            x1, y1 = self.trail[i - 1]
            x2, y2 = self.trail[i]
            trail_surf = pygame.Surface(
                (surface.get_width(), surface.get_height()), pygame.SRCALPHA
            )
            pygame.draw.line(trail_surf, (68, 170, 255, alpha),
                             (x1, y1), (x2, y2), 2)
            surface.blit(trail_surf, (0, 0))

    # ──────────────────────────────────────────────
    # HUD PANEL
    # ──────────────────────────────────────────────
    def _draw_hud(self, surface, fuel, max_fuel, fungicide, pesticide,
                  step_count, score, last_event):
        hud_h  = 58
        hud_y  = self.WINDOW_H - hud_h

        # Semi-transparent background
        hud_surf = pygame.Surface((self.WINDOW_W, hud_h), pygame.SRCALPHA)
        hud_surf.fill((10, 14, 22, 215))
        surface.blit(hud_surf, (0, hud_y))
        pygame.draw.line(surface, (80, 140, 200),
                         (0, hud_y), (self.WINDOW_W, hud_y), 1)

        # ── Sections ─────────────────────────────
        sections = [
            ("STEP",      f"{step_count:03d}",        HUD_TEXT),
            ("SCORE",     f"+{int(score)}",            SCORE_GREEN),
            ("FUNGICIDE", f"x{fungicide}",             FUNGAL_COL),
            ("PESTICIDE", f"x{pesticide}",             PEST_COL),
            ("INFECTED",  "—",                         INFECTED_COL),
        ]
        col_x = 14
        for label, val, col in sections:
            lbl_surf = self.font_hud.render(label, True, HUD_MUTED)
            val_surf = self.font_md.render(val, True, col)
            surface.blit(lbl_surf, (col_x, hud_y + 8))
            surface.blit(val_surf, (col_x, hud_y + 24))
            col_x += 100

        # ── Fuel bar ─────────────────────────────
        fuel_x = col_x
        lbl_f  = self.font_hud.render("FUEL", True, HUD_MUTED)
        surface.blit(lbl_f, (fuel_x, hud_y + 8))
        bar_w  = 100
        bar_h  = 10
        bar_y  = hud_y + 28
        pygame.draw.rect(surface, (40, 40, 55),
                         (fuel_x, bar_y, bar_w, bar_h), border_radius=4)
        pct    = fuel / max_fuel
        bar_col = FUEL_GREEN if pct > 0.4 else FUEL_YELLOW if pct > 0.2 else FUEL_RED
        fill_w  = int(bar_w * pct)
        if fill_w > 0:
            pygame.draw.rect(surface, bar_col,
                             (fuel_x, bar_y, fill_w, bar_h), border_radius=4)
        pct_lbl = self.font_sm.render(f"{int(pct*100)}%", True, HUD_MUTED)
        surface.blit(pct_lbl, (fuel_x + bar_w + 6, bar_y))

        # ── Event log ────────────────────────────
        if last_event:
            # Determine colour from content
            ecol = HUD_TEXT
            le_lower = last_event.lower()
            if "correctly" in le_lower or "complete" in le_lower or "+20" in le_lower:
                ecol = SCORE_GREEN
            elif "wrong" in le_lower or "crash" in le_lower or "failed" in le_lower:
                ecol = (255, 80, 80)
            elif "scan" in le_lower or "revealed" in le_lower:
                ecol = WARNING_COL
            elif "base" in le_lower:
                ecol = (100, 160, 220)
            ev_surf = self.font_hud.render(
                f">> {last_event[:68]}", True, ecol
            )
            surface.blit(ev_surf, (14, hud_y + 42))

        # ── Legend top-right ──────────────────────
        legend = [
            (CELL_COLORS[HEALTHY], "Healthy"),
            (CELL_COLORS[AT_RISK],  "At-risk"),
            (CELL_COLORS[FUNGAL],   "Fungal"),
            (CELL_COLORS[PEST],     "Pest"),
            (CELL_COLORS[TREATED],  "Treated"),
            (CELL_COLORS[DEAD],     "Dead"),
        ]
        lx = self.WINDOW_W - 310
        for i, (lc, lt) in enumerate(legend):
            pygame.draw.rect(surface, lc,
                             (lx + i * 50, hud_y + 10, 12, 12), border_radius=2)
            ls = self.font_sm.render(lt, True, HUD_MUTED)
            surface.blit(ls, (lx + i * 50, hud_y + 26))

    # ──────────────────────────────────────────────
    # MAIN RENDER
    # ──────────────────────────────────────────────
    def render(
        self,
        grid,
        drone_row,
        drone_col,
        fuel,
        max_fuel,
        fungicide,
        pesticide,
        step_count,
        score,
        last_event,
        render_mode="human",
    ) -> Optional[np.ndarray]:
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        self.tick += 1
        surface = self.screen

        # ── Background ───────────────────────────
        # Sky gradient
        for i in range(self.WINDOW_H // 2):
            t_val  = i / (self.WINDOW_H // 2)
            r_val  = int(26 + (90 - 26) * t_val)
            g_val  = int(42 + (154 - 42) * t_val)
            b_val  = int(58 + (191 - 58) * t_val)
            pygame.draw.line(surface, (r_val, g_val, b_val),
                             (0, i), (self.WINDOW_W, i))
        surface.fill(BG_COLOR, (0, self.WINDOW_H // 2,
                                self.WINDOW_W, self.WINDOW_H // 2))

        # ── Draw tiles back-to-front (painter's algorithm) ──
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self._draw_tile(surface, col, row, grid[row, col])

        # ── Draw crops back-to-front ─────────────
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = grid[row, col]
                x, y  = self.iso(col, row)
                # Offset two plants per cell
                for p_offset, (ox, oy) in enumerate([(-10, 4), (10, 8)]):
                    seed = col * 17 + row * 31 + p_offset * 7
                    self._draw_crop(surface, x + ox, y + oy, state, seed)

        # ── Trail ────────────────────────────────
        drone_screen = self.iso(drone_col, drone_row)
        drone_x = drone_screen[0]
        drone_y = drone_screen[1] - 18
        self.trail.append((drone_x, drone_y))
        if len(self.trail) > 55:
            self.trail.pop(0)
        self._draw_trail(surface)

        # ── Particles ────────────────────────────
        self._update_particles(surface)

        # ── Drone ────────────────────────────────
        self._draw_drone(surface, drone_x, drone_y)

        # ── HUD ──────────────────────────────────
        self._draw_hud(
            surface, fuel, max_fuel, fungicide, pesticide,
            step_count, score, last_event
        )

        if render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.FPS)
            return None
        else:
            return np.transpose(
                pygame.surfarray.array3d(surface), (1, 0, 2)
            )

    def close(self):
        pygame.quit()
