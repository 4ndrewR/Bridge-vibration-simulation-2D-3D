"""
bridge_simulator.py — interactive pygame bridge vibration simulator.

Native window with the same features as the HTML simulator: scenario picker
(cars, trucks, traffic, overload, earthquakes, wind, pedestrians, combined),
upgrade system (high-strength steel, deeper section, central pier, tuned
mass damper), live readouts, and dramatic collapse animation when the
moment demand exceeds the section capacity.

Run:
    pip install pygame
    python bridge_simulator.py
"""
from __future__ import annotations

import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import pygame
from pygame import Rect, Surface, Vector2

# =============================================================================
# Window + palette + fonts
# =============================================================================
WIN_W, WIN_H = 1480, 880
HEADER_H = 66
FOOTER_H = 64
LEFT_W   = 280
RIGHT_W  = 300
FPS      = 60

BG_0   = (7, 13, 23)
BG_1   = (12, 23, 38)
BG_2   = (20, 35, 54)
BG_3   = (30, 50, 71)
LINE   = (31, 53, 80)
LINE_D = (21, 38, 58)
INK    = (232, 240, 250)
INK_D  = (138, 160, 184)
INK_M  = (86, 112, 138)
CYAN   = (34, 211, 238)
CYAN_D = (14, 116, 144)
AMBER  = (251, 191, 36)
RED    = (248, 113, 113)
GREEN  = (52, 211, 153)
GRID   = (16, 32, 53)


# Global text-size multiplier. Bump this to make every label/readout larger
# without touching individual call sites — all fonts flow through _font().
UI_SCALE = 1.4


def _font(family_candidates: list[str], size: int, bold: bool = False,
          scale: bool = True) -> pygame.font.Font:
    if scale:
        size = max(1, int(round(size * UI_SCALE)))
    for name in family_candidates:
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except Exception:
            pass
    return pygame.font.Font(None, size)


# Lazy-initialised in main()
FONTS: dict[str, pygame.font.Font] = {}


# =============================================================================
# Physics constants
# =============================================================================
L         = 60.0        # bridge length [m]
SEG       = 60
STEEL_E   = 2.10e11
STEEL_SY  = 355e6
HS_SY     = 690e6
RHO       = 7850.0


VEHICLES: dict[str, dict] = {
    "car":   dict(mass=1_500,   length=4.5,  speed=22.0,  color=(125, 211, 252)),
    "van":   dict(mass=3_500,   length=5.5,  speed=22.0,  color=(196, 181, 253)),
    "bus":   dict(mass=18_000,  length=12.0, speed=16.7,  color=(252, 211, 77)),
    "truck": dict(mass=32_000,  length=11.0, speed=16.7,  color=(251, 146, 60)),
    "heavy": dict(mass=44_000,  length=16.0, speed=13.9,  color=(248, 113, 113)),
    "super": dict(mass=100_000, length=24.0, speed=11.1,  color=(239, 68, 68)),
}


# =============================================================================
# State
# =============================================================================
@dataclass
class Vehicle:
    kind: str
    x: float
    mass: float
    length: float
    speed: float
    color: tuple[int, int, int]


@dataclass
class Earthquake:
    active: bool = False
    intensity: float = 0.0
    dominant: float = 2.5
    t0: float = 0.0
    duration: float = 0.0


@dataclass
class Piece:
    x: float
    y: float
    vx: float
    vy: float
    omega: float
    theta: float
    length: float


@dataclass
class State:
    # upgrades
    ups: dict = field(default_factory=lambda: {
        "hs_steel": False, "reinforce": False, "central_pier": False, "tmd": False
    })
    depth_mul: float = 1.0

    # derived bridge properties
    EI: float = 1.0
    I: float = 1.0
    W: float = 1.0
    muL: float = 1.0
    depth: float = 1.5
    sig_y: float = STEEL_SY
    A: float = 0.1
    capacity_kNm: float = 1000.0
    f1: float = 1.5
    zeta: float = 0.02

    # dynamic
    vehicles: list[Vehicle] = field(default_factory=list)
    earthquake: Earthquake = field(default_factory=Earthquake)
    wind_active: bool = False
    wind_t0: float = 0.0
    peds_active: bool = False
    peds_t0: float = 0.0

    # SDOF mode-1 response
    amp: float = 0.0
    v_amp: float = 0.0

    static_defl: list[float] = field(default_factory=lambda: [0.0] * (SEG + 1))
    static_M_max: float = 0.0
    last_cap_frac: float = 0.0

    # visual
    t: float = 0.0
    paused: bool = False
    speed: float = 1.0
    shake_x: float = 0.0
    shake_y: float = 0.0
    collapsed: bool = False
    pieces: list[Piece] = field(default_factory=list)
    midspan_hist: deque = field(default_factory=lambda: deque(maxlen=400))
    wind_particles: list[list[float]] = field(default_factory=list)
    scenario_label: str = "READY"

    # Kanai-Tajimi internal state
    eq_z1: float = 0.0
    eq_z2: float = 0.0


# =============================================================================
# Bridge physics
# =============================================================================
def recompute_bridge(s: State) -> None:
    depth = 1.50 * s.depth_mul
    width = 3.0
    t_wall = 0.025

    I_out = (width * depth ** 3) / 12.0
    I_in  = ((width - 2 * t_wall) * (depth - 2 * t_wall) ** 3) / 12.0
    I     = I_out - I_in
    A     = (width * depth) - ((width - 2 * t_wall) * (depth - 2 * t_wall))
    c     = depth / 2.0
    W     = I / c

    sig_y = HS_SY if s.ups["hs_steel"] else STEEL_SY
    E     = STEEL_E

    f1 = (math.pi / (2 * L * L)) * math.sqrt(E * I / (RHO * A))
    if s.ups["central_pier"]:
        f1 *= 4.0

    s.zeta = 0.08 if s.ups["tmd"] else 0.02

    SERVICE_FACTOR = 0.45
    M_yield = sig_y * W
    M_serv  = M_yield * SERVICE_FACTOR
    cap_kNm = (M_serv / 1000.0) * (4.0 if s.ups["central_pier"] else 1.0)

    s.EI = E * I
    s.I = I
    s.W = W
    s.A = A
    s.muL = RHO * A
    s.depth = depth
    s.sig_y = sig_y
    s.f1 = f1
    s.capacity_kNm = cap_kNm


def static_deflection(s: State, P: float, a: float) -> list[float]:
    """Positive y = downward sag, per the engineering convention."""
    EI = s.EI
    out = [0.0] * (SEG + 1)
    if s.ups["central_pier"]:
        Lh = L / 2
        ah = a if a < Lh else a - Lh
        bh = Lh - ah
        for i in range(SEG + 1):
            x = (i / SEG) * L
            if (a < Lh and x < Lh) or (a >= Lh and x >= Lh):
                xh = x if x < Lh else x - Lh
                if xh <= ah:
                    y = P * bh * xh * (Lh * Lh - bh * bh - xh * xh) / (6 * Lh * EI)
                else:
                    y = P * ah * (Lh - xh) * (2 * Lh * xh - ah * ah - xh * xh) / (6 * Lh * EI)
            else:
                y = 0.0
            out[i] = y
    else:
        b = L - a
        for i in range(SEG + 1):
            x = (i / SEG) * L
            if x <= a:
                y = P * b * x * (L * L - b * b - x * x) / (6 * L * EI)
            else:
                y = P * a * (L - x) * (2 * L * x - a * a - x * x) / (6 * L * EI)
            out[i] = y
    return out


def max_static_moment(s: State, P: float, a: float) -> float:
    if s.ups["central_pier"]:
        Lh = L / 2
        ah = a if a < Lh else a - Lh
        bh = Lh - ah
        return P * ah * bh / Lh
    return P * a * (L - a) / L


def recompute_static_from_vehicles(s: State) -> None:
    for i in range(SEG + 1):
        s.static_defl[i] = 0.0
    M_max = 0.0
    for v in s.vehicles:
        if v.x < 0 or v.x > L:
            continue
        P = v.mass * 9.81
        y = static_deflection(s, P, v.x)
        for i in range(SEG + 1):
            s.static_defl[i] += y[i]
        M = max_static_moment(s, P, v.x)
        if M > M_max:
            M_max = M
    s.static_M_max = M_max


def eq_accel(s: State) -> float:
    """Kanai-Tajimi filter excited by white noise, with Jennings envelope."""
    wg = 2 * math.pi * s.earthquake.dominant
    zg = 0.6
    dt = 1.0 / 60.0
    w = (random.random() * 2 - 1) * 0.6
    s.eq_z1 += dt * s.eq_z2
    s.eq_z2 += dt * (-2 * zg * wg * s.eq_z2 - wg * wg * s.eq_z1 + w)
    a = -(2 * zg * wg * s.eq_z2 + wg * wg * s.eq_z1)

    tau = s.t - s.earthquake.t0
    T = s.earthquake.duration
    if tau < 0 or tau > T:
        return 0.0
    if tau < 2:
        env = (tau / 2) ** 2
    elif tau < T - 5:
        env = 1.0
    else:
        env = math.exp(-0.4 * (tau - (T - 5)))
    return a * env * s.earthquake.intensity * 9.81 * 2.0


def step_dynamic(s: State, dt: float) -> None:
    w1 = 2 * math.pi * s.f1
    k = w1 * w1
    c = 2 * s.zeta * w1

    F = 0.0

    for v in s.vehicles:
        if v.x < 0 or v.x > L:
            continue
        phi_x = math.sin(math.pi * v.x / L)
        Pn = v.mass * 9.81 * phi_x
        F += Pn / (s.muL * L * 0.5) * 0.06

    if s.earthquake.active:
        a_g = eq_accel(s)
        F += (-a_g) * (2 / math.pi)

    if s.wind_active:
        F += (random.random() * 2 - 1) * 0.4 * math.sin(2 * math.pi * s.f1 * s.t + random.random() * 0.5)

    if s.peds_active:
        F += 0.6 * math.sin(2 * math.pi * s.f1 * s.t) + 0.15 * math.sin(2 * math.pi * s.f1 * s.t * 2 + 0.3)

    a = F - c * s.v_amp - k * s.amp
    s.v_amp += a * dt
    s.amp   += s.v_amp * dt


def total_defl_at(s: State, i: int) -> float:
    x = (i / SEG) * L
    return s.static_defl[i] + s.amp * math.sin(math.pi * x / L)


def midspan_mm(s: State) -> float:
    return total_defl_at(s, SEG // 2) * 1000.0


def effective_moment_max(s: State) -> float:
    # Dynamic contribution: the pure A·EI·(π/L)² term is small for our
    # section, so we lump in dynamic amplification + low-cycle fatigue with
    # an empirical factor (treat this as an SLS check rather than instant
    # bending stress).
    M_dyn_pure = abs(s.amp) * s.EI * (math.pi / L) ** 2
    M_dyn_eff  = M_dyn_pure * 10.0
    return s.static_M_max + M_dyn_eff


# =============================================================================
# Scenarios
# =============================================================================
def spawn(s: State, kind: str) -> None:
    if s.collapsed:
        return
    v = VEHICLES[kind]
    s.vehicles.append(Vehicle(
        kind=kind,
        x=-v["length"] - 1,
        mass=v["mass"],
        length=v["length"],
        speed=v["speed"],
        color=v["color"],
    ))


def start_scenario(s: State, name: str, deferred: list[Callable]) -> None:
    if s.collapsed:
        reset_bridge(s)
    if name in ("car", "van", "bus", "truck", "heavy"):
        spawn(s, name)
        s.scenario_label = name.upper()
    elif name == "convoy":
        seq = ["car", "van", "bus", "truck", "heavy", "car"]
        t0 = s.t
        for i, k in enumerate(seq):
            deferred.append((t0 + i * 0.9, lambda kk=k: spawn(s, kk)))
        s.scenario_label = "MIXED TRAFFIC · 6 vehicles"
    elif name == "overload":
        t0 = s.t
        for k in range(5):
            deferred.append((t0 + k * 1.3, lambda: spawn(s, "super")))
        s.scenario_label = "OVERLOAD · 5 × 100 t"
    elif name == "earthquake":
        s.earthquake = Earthquake(active=True, intensity=0.30, dominant=2.5, t0=s.t, duration=25.0)
        s.eq_z1 = 0.0; s.eq_z2 = 0.0
        s.scenario_label = "EARTHQUAKE · PGA 0.30 g"
    elif name == "big_quake":
        s.earthquake = Earthquake(active=True, intensity=0.60, dominant=2.5, t0=s.t, duration=25.0)
        s.eq_z1 = 0.0; s.eq_z2 = 0.0
        s.scenario_label = "MAJOR QUAKE · PGA 0.60 g"
    elif name == "wind":
        s.wind_active = True
        s.wind_t0 = s.t
        s.scenario_label = "WIND BUFFETING · 25 m/s"
    elif name == "pedestrians":
        s.peds_active = True
        s.peds_t0 = s.t
        s.scenario_label = "PEDESTRIAN CROWD · 200"
    elif name == "combo":
        spawn(s, "heavy")
        deferred.append((s.t + 1.5, lambda: setattr(
            s, "earthquake",
            Earthquake(active=True, intensity=0.20, dominant=2.5, t0=s.t, duration=18.0),
        )))
        s.scenario_label = "COMBO · TRUCK + QUAKE"


def trigger_collapse(s: State, reason: str) -> None:
    if s.collapsed:
        return
    s.collapsed = True
    s.scenario_label = "COLLAPSED"
    s.fail_reason = reason
    s.pieces = []
    n = 14
    for k in range(n):
        x0 = (k       / n) * L
        x1 = ((k + 1) / n) * L
        xm = 0.5 * (x0 + x1)
        idx = int(round(xm / L * SEG))
        ym = s.static_defl[idx] + s.amp * math.sin(math.pi * xm / L)
        outward = xm - L / 2
        s.pieces.append(Piece(
            x=xm, y=ym,
            vx=outward * 0.08 + (random.random() - 0.5) * 1.5,
            vy=-1.5 - random.random() * 2.0,
            omega=(random.random() - 0.5) * 2.0,
            theta=0.0,
            length=(x1 - x0),
        ))


def reset_bridge(s: State) -> None:
    s.vehicles = []
    s.earthquake = Earthquake()
    s.wind_active = False
    s.peds_active = False
    s.amp = 0.0; s.v_amp = 0.0
    for i in range(SEG + 1):
        s.static_defl[i] = 0.0
    s.static_M_max = 0.0
    s.collapsed = False
    s.pieces = []
    s.shake_x = s.shake_y = 0.0
    s.midspan_hist.clear()
    s.scenario_label = "READY"
    s.fail_reason = ""


def clear_loads(s: State) -> None:
    s.vehicles = []
    s.earthquake.active = False
    s.wind_active = False
    s.peds_active = False
    for i in range(SEG + 1):
        s.static_defl[i] = 0.0
    s.static_M_max = 0.0
    s.scenario_label = "READY"


# =============================================================================
# Rendering helpers
# =============================================================================
def world_x(s: State, x_m: float, stage_rect: Rect) -> float:
    pad = 80
    return stage_rect.left + pad + (x_m / L) * (stage_rect.width - 2 * pad) + s.shake_x


def deck_base_y(s: State, stage_rect: Rect) -> float:
    return stage_rect.top + stage_rect.height * 0.50 + s.shake_y


def world_y(s: State, y_m: float, stage_rect: Rect) -> float:
    return deck_base_y(s, stage_rect) + y_m * 600.0


def draw_text(surf: Surface, text: str, font_key: str, pos, color, align="topleft") -> Rect:
    img = FONTS[font_key].render(text, True, color)
    r = img.get_rect(**{align: pos})
    surf.blit(img, r)
    return r


def draw_grid(surf: Surface, stage: Rect, s: State) -> None:
    surf.fill(BG_0, stage)
    step = 30
    for x in range(stage.left, stage.right, step):
        pygame.draw.line(surf, GRID, (x, stage.top), (x, stage.bottom), 1)
    for y in range(stage.top, stage.bottom, step):
        pygame.draw.line(surf, GRID, (stage.left, y), (stage.right, y), 1)
    # brighter every 5
    bright = (16, 50, 80)
    for x in range(stage.left, stage.right, step * 5):
        pygame.draw.line(surf, bright, (x, stage.top), (x, stage.bottom), 1)
    for y in range(stage.top, stage.bottom, step * 5):
        pygame.draw.line(surf, bright, (stage.left, y), (stage.right, y), 1)

    ground_y = deck_base_y(s, stage) + 130
    pygame.draw.line(surf, CYAN, (stage.left, ground_y), (stage.right, ground_y), 1)
    # earth ticks
    for x in range(stage.left + 10, stage.right, 22):
        pygame.draw.line(surf, (34, 90, 120),
                         (x, ground_y + 1), (x + 8, ground_y + 10), 1)


def draw_supports(surf: Surface, stage: Rect, s: State) -> None:
    base_y = deck_base_y(s, stage)
    ground_y = base_y + 130
    xL = world_x(s, 0, stage)
    xR = world_x(s, L, stage)

    # Left pin
    pygame.draw.polygon(surf, BG_3, [(xL, base_y + 8), (xL - 22, ground_y), (xL + 22, ground_y)])
    pygame.draw.polygon(surf, CYAN, [(xL, base_y + 8), (xL - 22, ground_y), (xL + 22, ground_y)], 1)
    pygame.draw.circle(surf, CYAN, (int(xL), int(base_y + 8)), 5)

    # Right roller
    pygame.draw.polygon(surf, BG_3, [(xR, base_y + 8), (xR - 22, ground_y - 6), (xR + 22, ground_y - 6)])
    pygame.draw.polygon(surf, CYAN, [(xR, base_y + 8), (xR - 22, ground_y - 6), (xR + 22, ground_y - 6)], 1)
    for k in range(-2, 3):
        pygame.draw.circle(surf, CYAN, (int(xR + k * 9), int(ground_y - 2)), 3)

    # Central pier
    if s.ups["central_pier"]:
        xC = world_x(s, L / 2, stage)
        pier_rect = Rect(int(xC - 14), int(base_y + 8), 28, int(ground_y - (base_y + 8) - 4))
        pygame.draw.rect(surf, BG_3, pier_rect)
        pygame.draw.rect(surf, CYAN, pier_rect, 1)
        pygame.draw.rect(surf, CYAN, Rect(int(xC - 18), int(base_y + 5), 36, 4))


def deflected_polygon(s: State, stage: Rect, t_pix: int):
    top = []
    bot = []
    for i in range(SEG + 1):
        x_m = (i / SEG) * L
        y_m = total_defl_at(s, i)
        sx = world_x(s, x_m, stage)
        sy = world_y(s, y_m, stage)
        top.append((sx, sy - t_pix / 2))
        bot.append((sx, sy + t_pix / 2))
    return top, bot


def draw_deck(surf: Surface, stage: Rect, s: State) -> None:
    thickness = int(18 * s.depth_mul)
    top, bot = deflected_polygon(s, stage, thickness)
    poly = top + list(reversed(bot))
    pygame.draw.polygon(surf, BG_2, [(int(x), int(y)) for x, y in poly])
    pygame.draw.polygon(surf, (34, 90, 120), [(int(x), int(y)) for x, y in poly], 1)

    # stress overlay strip
    cap_used = s.last_cap_frac
    if cap_used < 0.55:
        c = GREEN
    elif cap_used < 0.85:
        c = AMBER
    else:
        c = RED
    alpha = int(min(102, 50 + cap_used * 130))
    overlay = pygame.Surface((stage.width, stage.height), pygame.SRCALPHA)
    pts = []
    for i, (xt, yt) in enumerate(top):
        pts.append((xt - stage.left, yt - stage.top))
    for i in range(len(top) - 1, -1, -1):
        xt, yt = top[i]
        pts.append((xt - stage.left, yt - stage.top + 4))
    pygame.draw.polygon(overlay, (*c, alpha), [(int(p[0]), int(p[1])) for p in pts])
    surf.blit(overlay, stage.topleft)

    # 10 m tick labels
    for x_m in range(0, int(L) + 1, 10):
        i = int(round(x_m / L * SEG))
        sx = world_x(s, x_m, stage)
        sy = bot[i][1]
        pygame.draw.line(surf, (34, 90, 120), (sx, sy), (sx, sy + 8), 1)
        draw_text(surf, f"{x_m}m", "mono10", (sx, sy + 10), INK_M, align="midtop")


def draw_vehicles(surf: Surface, stage: Rect, s: State) -> None:
    px_per_m = world_x(s, 1, stage) - world_x(s, 0, stage)
    for v in s.vehicles:
        if v.x < -v.length * 2 or v.x > L + v.length * 2:
            continue
        i = max(0, min(SEG, int(round(v.x / L * SEG))))
        y_m = total_defl_at(s, i)
        sx = world_x(s, v.x, stage)
        sy = world_y(s, y_m, stage)
        draw_vehicle_shape(surf, sx, sy - 9, v, px_per_m)


def draw_vehicle_shape(surf: Surface, cx: float, base_y: float, v: Vehicle, px_per_m: float) -> None:
    w = max(20, v.length * px_per_m * 0.95)
    h = {"car": 10, "van": 12}.get(v.kind, 16)
    x0 = int(cx - w / 2)
    y0 = int(base_y - h)

    # shadow
    pygame.draw.rect(surf, (0, 0, 0, 90), Rect(x0 - 2, y0 + h + 1, int(w) + 4, 2))

    if v.kind in ("car", "van"):
        poly = [
            (x0,                          y0 + h),
            (x0,                          y0 + h * 0.7),
            (x0 + w * 0.20,               y0 + h * 0.7),
            (x0 + w * 0.30,               y0),
            (x0 + w * 0.75,               y0),
            (x0 + w * 0.85,               y0 + h * 0.5),
            (x0 + w,                      y0 + h * 0.5),
            (x0 + w,                      y0 + h),
        ]
        pygame.draw.polygon(surf, v.color, [(int(p[0]), int(p[1])) for p in poly])
        pygame.draw.polygon(surf, (10, 15, 25), [(int(p[0]), int(p[1])) for p in poly], 1)
        pygame.draw.rect(surf, (255, 255, 255, 160), Rect(int(x0 + w * 0.32), y0 + 2, int(w * 0.40), int(h * 0.55)))
    elif v.kind == "bus":
        pygame.draw.rect(surf, v.color, Rect(x0, y0, int(w), h))
        pygame.draw.rect(surf, (10, 15, 25), Rect(x0, y0, int(w), h), 1)
        for k in range(6):
            wx = x0 + 4 + k * (w / 6)
            pygame.draw.rect(surf, (220, 230, 240), Rect(int(wx), y0 + 3, int(w / 6 - 3), int(h * 0.45)))
    elif v.kind == "truck":
        cab = int(w * 0.30); box = int(w * 0.70)
        pygame.draw.rect(surf, v.color, Rect(x0 + box, y0, cab, h))
        pygame.draw.rect(surf, (10, 15, 25), Rect(x0 + box, y0, cab, h), 1)
        pygame.draw.rect(surf, (31, 41, 55), Rect(x0, int(y0 + h * 0.10), box - 2, int(h * 0.85)))
        pygame.draw.rect(surf, (10, 15, 25), Rect(x0, int(y0 + h * 0.10), box - 2, int(h * 0.85)), 1)
        pygame.draw.rect(surf, (220, 230, 240), Rect(x0 + box + 3, y0 + 3, cab - 6, int(h * 0.45)))
    elif v.kind in ("heavy", "super"):
        trac = int(w * 0.22); trail = int(w * 0.78)
        pygame.draw.rect(surf, v.color, Rect(x0 + trail, y0, trac - 1, h))
        pygame.draw.rect(surf, (10, 15, 25), Rect(x0 + trail, y0, trac - 1, h), 1)
        pygame.draw.rect(surf, (15, 23, 42), Rect(x0, int(y0 + h * 0.12), trail - 3, int(h * 0.85)))
        pygame.draw.rect(surf, (10, 15, 25), Rect(x0, int(y0 + h * 0.12), trail - 3, int(h * 0.85)), 1)
        pygame.draw.rect(surf, v.color, Rect(x0 + trail - 3, int(y0 + h * 0.15), 2, int(h * 0.78)))
        pygame.draw.rect(surf, (220, 230, 240), Rect(x0 + trail + 3, y0 + 3, trac - 7, int(h * 0.45)))

    # wheels
    if v.kind == "car":
        positions = [0.20 * w, 0.80 * w]
    elif v.kind in ("van", "bus"):
        positions = [0.15 * w, 0.85 * w]
    elif v.kind == "truck":
        positions = [0.10 * w, 0.70 * w, 0.90 * w]
    else:
        positions = [0.10 * w, 0.40 * w, 0.80 * w, 0.95 * w]
    for px in positions:
        pygame.draw.circle(surf, (15, 23, 42), (int(x0 + px), int(y0 + h + 1)), 3)
        pygame.draw.circle(surf, (71, 85, 105), (int(x0 + px), int(y0 + h + 1)), 3, 1)


def draw_pedestrians(surf: Surface, stage: Rect, s: State) -> None:
    if not s.peds_active:
        return
    n = 22
    for k in range(n):
        phase = (s.t * 1.4 + k * 0.5) % 1.0
        x_m = phase * L
        i = int(round(x_m / L * SEG))
        if i < 0 or i > SEG:
            continue
        sx = world_x(s, x_m, stage)
        sy = world_y(s, total_defl_at(s, i), stage) - 9
        bounce = math.sin(s.t * 2 * math.pi * s.f1 + k) * 1.5
        cy = int(sy + bounce)
        pygame.draw.circle(surf, INK_D, (int(sx), cy - 4), 2)
        pygame.draw.line(surf, INK_D, (int(sx), cy - 3), (int(sx), cy + 2), 1)
        pygame.draw.line(surf, INK_D, (int(sx), cy + 2), (int(sx - 2), cy + 5), 1)
        pygame.draw.line(surf, INK_D, (int(sx), cy + 2), (int(sx + 2), cy + 5), 1)


def draw_wind(surf: Surface, stage: Rect, s: State) -> None:
    if not s.wind_active:
        return
    while len(s.wind_particles) < 80:
        s.wind_particles.append([
            stage.left - random.random() * 100,
            stage.top + 40 + random.random() * (stage.height - 160),
            4 + random.random() * 5,
        ])
    for p in s.wind_particles:
        p[0] += p[2]
        pygame.draw.line(surf, (34, 130, 160),
                         (p[0], p[1]), (p[0] - 18 - p[2] * 0.8, p[1]), 1)
        if p[0] > stage.right + 40:
            p[0] = stage.left - random.random() * 100
            p[1] = stage.top + 40 + random.random() * (stage.height - 160)
            p[2] = 4 + random.random() * 5


def draw_quake_tint(surf: Surface, stage: Rect, s: State) -> None:
    if not s.earthquake.active:
        return
    tint = pygame.Surface((stage.width, stage.height), pygame.SRCALPHA)
    tint.fill((248, 113, 113, 14))
    surf.blit(tint, stage.topleft)


def draw_collapsed(surf: Surface, stage: Rect, s: State) -> None:
    # dust cloud
    cloud = pygame.Surface((stage.width, stage.height), pygame.SRCALPHA)
    for _ in range(3):
        cx = stage.width * 0.5 + (random.random() - 0.5) * 200
        cy = stage.height * 0.65 + (random.random() - 0.5) * 60
        pygame.draw.circle(cloud, (251, 191, 36, 10),
                           (int(cx), int(cy)), int(80 + random.random() * 60))
    surf.blit(cloud, stage.topleft)

    px_per_m = world_x(s, 1, stage) - world_x(s, 0, stage)
    for p in s.pieces:
        sx = world_x(s, p.x, stage)
        sy = world_y(s, p.y, stage)
        w = max(8, p.length * px_per_m * 0.92)
        h = max(6, 18 * s.depth_mul)
        # rotate via surface
        block = pygame.Surface((int(w) + 6, int(h) + 6), pygame.SRCALPHA)
        pygame.draw.rect(block, (42, 61, 85),
                         Rect(3, 3, int(w), int(h)))
        pygame.draw.rect(block, (248, 113, 113),
                         Rect(3, 3, int(w), int(h)), 1)
        # crack lines
        pygame.draw.line(block, (190, 80, 80, 200),
                         (3, 3 + h // 2), (3 + w, 3 + h // 2), 1)
        rot = pygame.transform.rotate(block, -math.degrees(p.theta))
        rr = rot.get_rect(center=(int(sx), int(sy)))
        surf.blit(rot, rr.topleft)


# =============================================================================
# Buttons / UI
# =============================================================================
@dataclass
class Button:
    rect: Rect
    label: str
    sublabel: str = ""
    icon: str = ""
    kind: str = "scenario"      # scenario | toggle | pill | speed
    payload: str = ""
    color: tuple = INK
    accent: tuple = CYAN
    active: bool = False
    border_color: tuple = LINE


def hit(b: Button, pos) -> bool:
    return b.rect.collidepoint(pos)


def draw_button(surf: Surface, b: Button, mouse_pos) -> None:
    is_hover = b.rect.collidepoint(mouse_pos)

    if b.kind == "scenario":
        bg = BG_3 if is_hover else BG_2
        border = b.accent if is_hover or b.active else b.border_color
        pygame.draw.rect(surf, bg, b.rect, border_radius=6)
        pygame.draw.rect(surf, border, b.rect, 1, border_radius=6)
        # icon
        if b.icon:
            draw_text(surf, b.icon, "icon", (b.rect.left + 16, b.rect.centery), b.accent, align="center")
        # label
        draw_text(surf, b.label, "ui_b",
                  (b.rect.left + 36, b.rect.top + 8), INK)
        if b.sublabel:
            draw_text(surf, b.sublabel, "mono10",
                      (b.rect.left + 36, b.rect.top + 26), INK_M)

    elif b.kind == "toggle":
        bg = (15, 35, 55) if b.active else (BG_3 if is_hover else BG_2)
        border = b.accent if b.active else (b.accent if is_hover else b.border_color)
        pygame.draw.rect(surf, bg, b.rect, border_radius=6)
        pygame.draw.rect(surf, border, b.rect, 1 + (1 if b.active else 0), border_radius=6)
        if b.icon:
            draw_text(surf, b.icon, "icon",
                      (b.rect.left + 16, b.rect.centery),
                      b.accent if b.active else INK_M, align="center")
        draw_text(surf, b.label, "ui_b",
                  (b.rect.left + 36, b.rect.top + 8), INK)
        if b.sublabel:
            draw_text(surf, b.sublabel, "mono10",
                      (b.rect.left + 36, b.rect.top + 26), INK_M)
        if b.active:
            badge = pygame.Rect(b.rect.right - 36, b.rect.top + 8, 28, 16)
            pygame.draw.rect(surf, (12, 60, 50), badge, border_radius=3)
            pygame.draw.rect(surf, GREEN, badge, 1, border_radius=3)
            draw_text(surf, "ON", "mono10",
                      badge.center, GREEN, align="center")

    elif b.kind == "pill":
        bg = b.color if not is_hover else tuple(min(255, c + 15) for c in b.color)
        pygame.draw.rect(surf, bg, b.rect, border_radius=99)
        if b.border_color:
            pygame.draw.rect(surf, b.border_color, b.rect, 1, border_radius=99)
        draw_text(surf, b.label, "ui_b", b.rect.center, b.accent, align="center")

    elif b.kind == "speed":
        bg = BG_3 if is_hover else BG_2
        pygame.draw.rect(surf, bg, b.rect, border_radius=4)
        pygame.draw.rect(surf, LINE, b.rect, 1, border_radius=4)
        draw_text(surf, b.label, "mono12", b.rect.center, INK, align="center")


# =============================================================================
# Panels and stats
# =============================================================================
def draw_header(surf: Surface, s: State) -> None:
    r = Rect(0, 0, WIN_W, HEADER_H)
    pygame.draw.rect(surf, BG_1, r)
    pygame.draw.line(surf, LINE, (0, HEADER_H), (WIN_W, HEADER_H), 1)

    # logo
    logo_y = (HEADER_H - 30) // 2
    pygame.draw.rect(surf, CYAN, Rect(22, logo_y, 30, 30), border_radius=6)
    pygame.draw.rect(surf, BG_0, Rect(30, logo_y + 8, 14, 14), border_radius=2)

    # title
    title_top = 6
    draw_text(surf, "BRIDGE·SIM", "title", (62, title_top), INK)
    draw_text(surf, "// PYGAME EDITION", "ui",
              (62 + FONTS["title"].size("BRIDGE·SIM")[0] + 10,
               title_top + FONTS["title"].get_height() - FONTS["ui"].get_height() - 2), INK_M)
    draw_text(surf, "FEM-BASED VIBRATION SIMULATOR · L=60m · STEEL S355",
              "mono10", (62, title_top + FONTS["title"].get_height() - 4), INK_M)

    # status tag (right side)
    cap = s.last_cap_frac
    if s.collapsed:
        col = RED; text = "FAILURE"
    elif cap < 0.55:
        col = CYAN; text = "NOMINAL"
    elif cap < 0.85:
        col = AMBER; text = "ELEVATED · WATCH STRESS"
    else:
        col = RED; text = "CRITICAL · NEAR YIELD"

    label_img = FONTS["mono11"].render(text, True, col)
    pad = 12
    tag_w = label_img.get_width() + 28
    tag_h = 26
    tag_rect = Rect(WIN_W - 22 - tag_w, (HEADER_H - tag_h) // 2, tag_w, tag_h)

    # background with alpha tint
    bg_surf = pygame.Surface((tag_w, tag_h), pygame.SRCALPHA)
    pygame.draw.rect(bg_surf, (*col, 28), Rect(0, 0, tag_w, tag_h), border_radius=99)
    pygame.draw.rect(bg_surf, (*col, 110), Rect(0, 0, tag_w, tag_h), 1, border_radius=99)
    surf.blit(bg_surf, tag_rect.topleft)

    pygame.draw.circle(surf, col, (tag_rect.left + 12, tag_rect.centery), 4)
    surf.blit(label_img, (tag_rect.left + 22, tag_rect.centery - label_img.get_height() // 2))


def draw_section_title(surf: Surface, text: str, pos) -> None:
    pygame.draw.line(surf, CYAN, (pos[0], pos[1] + 6), (pos[0] + 18, pos[1] + 6), 1)
    draw_text(surf, text, "mono10_upper", (pos[0] + 26, pos[1]), INK_M)


def draw_left_panel(surf: Surface, s: State, buttons: list[Button], mouse_pos) -> None:
    panel = Rect(0, HEADER_H, LEFT_W, WIN_H - HEADER_H - FOOTER_H)
    pygame.draw.rect(surf, BG_1, panel)
    pygame.draw.line(surf, LINE, (LEFT_W, HEADER_H),
                     (LEFT_W, WIN_H - FOOTER_H), 1)

    draw_section_title(surf, "EXCITATIONS", (16, HEADER_H + 14))
    draw_section_title(surf, "ENVIRONMENT", (16, HEADER_H + 14 + 7 * 50 + 16))
    draw_section_title(surf, "COMBINED",
                       (16, HEADER_H + 14 + 7 * 50 + 16 + 4 * 50 + 16))

    for b in buttons:
        if b.kind in ("scenario",):
            draw_button(surf, b, mouse_pos)


def draw_right_panel(surf: Surface, s: State, buttons: list[Button],
                     mouse_pos, spark_pts: deque) -> None:
    panel = Rect(WIN_W - RIGHT_W, HEADER_H, RIGHT_W, WIN_H - HEADER_H - FOOTER_H)
    pygame.draw.rect(surf, BG_1, panel)
    pygame.draw.line(surf, LINE, (panel.left, panel.top),
                     (panel.left, panel.bottom), 1)

    x0 = panel.left + 16
    y  = panel.top + 14
    draw_section_title(surf, "LIVE READOUTS", (x0, y))
    y += 24

    # CAPACITY USED gauge
    gauge = Rect(x0, y, panel.width - 32, 56)
    pygame.draw.rect(surf, BG_2, gauge, border_radius=6)
    pygame.draw.rect(surf, LINE, gauge, 1, border_radius=6)
    draw_text(surf, "CAPACITY USED", "mono10", (gauge.left + 12, gauge.top + 10), INK_M)
    pct = min(100, int(s.last_cap_frac * 100))
    draw_text(surf, f"{pct}", "huge", (gauge.right - 60, gauge.top + 6), INK, align="topright")
    draw_text(surf, "%", "mono11", (gauge.right - 14, gauge.top + 22), INK_M, align="topright")
    # bar
    bar = Rect(gauge.left + 12, gauge.bottom - 14, gauge.width - 24, 8)
    pygame.draw.rect(surf, BG_3, bar, border_radius=4)
    if pct > 0:
        fw = int(bar.width * min(1.0, pct / 100))
        # color band by value
        if pct < 55: bc = GREEN
        elif pct < 85: bc = AMBER
        else: bc = RED
        pygame.draw.rect(surf, bc, Rect(bar.left, bar.top, fw, bar.height), border_radius=4)
    y = gauge.bottom + 10

    # 4 mini-stats (2 cols)
    stats = [
        ("MID-SPAN DEFL.",   f"{midspan_mm(s):+.1f}", "mm"),
        ("MAX MOMENT",       f"{effective_moment_max(s)/1000:,.0f}", "kN·m"),
        ("PEAK STRESS",      f"{effective_moment_max(s)/s.W/1e6:.0f}", "MPa"),
        ("FREQUENCY f₁",     f"{s.f1:.2f}", "Hz"),
    ]
    cell_w = (panel.width - 38) // 2
    for k, (lbl, val, unit) in enumerate(stats):
        col = k % 2; row = k // 2
        cell = Rect(x0 + col * (cell_w + 6), y + row * 56, cell_w, 50)
        pygame.draw.rect(surf, BG_2, cell, border_radius=6)
        pygame.draw.rect(surf, LINE, cell, 1, border_radius=6)
        draw_text(surf, lbl, "mono9", (cell.left + 8, cell.top + 6), INK_M)
        draw_text(surf, val, "big", (cell.left + 8, cell.top + 22), INK)
        # unit
        v_img = FONTS["big"].render(val, True, INK)
        draw_text(surf, unit, "mono10",
                  (cell.left + 8 + v_img.get_width() + 4, cell.top + 32), INK_M)
    y += 2 * 56 + 6

    # spark
    sp = Rect(x0, y, panel.width - 32, 64)
    pygame.draw.rect(surf, BG_2, sp, border_radius=6)
    pygame.draw.rect(surf, LINE, sp, 1, border_radius=6)
    draw_text(surf, "MID-SPAN HISTORY · 20 s", "mono10", (sp.left + 8, sp.top + 6), INK_M)
    draw_spark(surf, sp, spark_pts)
    y = sp.bottom + 12

    # UPGRADES section
    draw_section_title(surf, "UPGRADES", (x0, y))
    y += 18
    for b in buttons:
        if b.kind == "toggle":
            draw_button(surf, b, mouse_pos)
    # already laid out by builder
    last_toggle = max((b.rect.bottom for b in buttons if b.kind == "toggle"), default=y)
    y = last_toggle + 10

    # CAPACITY BUDGET
    draw_section_title(surf, "CAPACITY BUDGET", (x0, y))
    y += 18
    g2 = Rect(x0, y, panel.width - 32, 70)
    pygame.draw.rect(surf, BG_2, g2, border_radius=6)
    pygame.draw.rect(surf, LINE, g2, 1, border_radius=6)
    draw_text(surf, "M_CAP", "mono10", (g2.left + 12, g2.top + 8), INK_M)
    draw_text(surf, f"{s.capacity_kNm:,.0f}", "big",
              (g2.right - 80, g2.top + 4), INK, align="topright")
    draw_text(surf, "kN·m", "mono10", (g2.right - 14, g2.top + 14), INK_M, align="topright")
    draw_text(surf, "DAF (DYN.)", "mono10", (g2.left + 12, g2.top + 40), INK_M)
    daf = 1.0 / (2 * s.zeta)
    draw_text(surf, f"{daf:.1f}", "big",
              (g2.right - 14, g2.top + 36), INK, align="topright")


def draw_spark(surf: Surface, rect: Rect, pts: deque) -> None:
    inner = Rect(rect.left + 6, rect.top + 22, rect.width - 12, rect.height - 28)
    pygame.draw.line(surf, (16, 60, 90),
                     (inner.left, inner.centery), (inner.right, inner.centery), 1)
    if len(pts) < 2:
        return
    max_abs = max(20.0, max(abs(v) for v in pts))
    last = None
    n = len(pts)
    for i, v in enumerate(pts):
        x = inner.left + (i / (n - 1)) * inner.width
        y = inner.centery - (v / max_abs) * (inner.height * 0.45)
        if last is not None:
            pygame.draw.line(surf, CYAN, last, (x, y), 1)
        last = (x, y)


def draw_footer(surf: Surface, buttons: list[Button], mouse_pos, s: State) -> None:
    r = Rect(0, WIN_H - FOOTER_H, WIN_W, FOOTER_H)
    pygame.draw.rect(surf, BG_1, r)
    pygame.draw.line(surf, LINE, (0, WIN_H - FOOTER_H), (WIN_W, WIN_H - FOOTER_H), 1)
    for b in buttons:
        if b.kind in ("pill", "speed"):
            draw_button(surf, b, mouse_pos)
    # speed label
    sx = WIN_W - 200
    draw_text(surf, "SPEED", "mono11", (sx, WIN_H - FOOTER_H // 2 - 6), INK_M)
    draw_text(surf, f"{s.speed:g}×", "mono12",
              (sx + 60, WIN_H - FOOTER_H // 2 - 6), CYAN)


def draw_overlay(surf: Surface, stage: Rect, s: State) -> None:
    draw_text(surf, f"t = {s.t:.2f} s", "mono10",
              (stage.left + 14, stage.top + 14), INK_M)
    draw_text(surf, s.scenario_label, "mono11",
              (stage.left + 14, stage.top + 28), CYAN)


def draw_fail_banner(surf: Surface, stage: Rect, s: State) -> None:
    if not s.collapsed:
        return
    text = "STRUCTURAL FAILURE"
    img = FONTS["banner"].render(text, True, RED)
    img_rect = img.get_rect(center=(stage.centerx, stage.top + stage.height // 3))
    # glow
    glow = pygame.Surface((img.get_width() + 20, img.get_height() + 20), pygame.SRCALPHA)
    pygame.draw.rect(glow, (248, 113, 113, 30),
                     glow.get_rect(), border_radius=12)
    surf.blit(glow, (img_rect.left - 10, img_rect.top - 10))
    surf.blit(img, img_rect)
    if getattr(s, "fail_reason", ""):
        draw_text(surf, s.fail_reason, "mono10",
                  (stage.centerx, img_rect.bottom + 8), INK_D, align="midtop")


# =============================================================================
# Layout builders
# =============================================================================
def build_buttons(s: State) -> list[Button]:
    buttons: list[Button] = []

    y = HEADER_H + 14 + 22
    excitations = [
        ("car",   "🚗", "Passenger car",   "1.5 t · 80 km/h", CYAN),
        ("van",   "🚚", "Van",             "3.5 t · 80 km/h", CYAN),
        ("bus",   "🚌", "Bus",             "18 t · 60 km/h",  CYAN),
        ("truck", "🛻", "Rigid truck",     "32 t · 60 km/h",  CYAN),
        ("heavy", "🚛", "Heavy lorry",     "44 t · 50 km/h",  AMBER),
        ("convoy", "🛣", "Mixed traffic",   "6 vehicles",     AMBER),
        ("overload", "⚠", "Overload test", "100 t convoy",   AMBER),
    ]
    for k, icon, lbl, sub, acc in excitations:
        buttons.append(Button(
            rect=Rect(16, y, LEFT_W - 32, 44),
            label=lbl, sublabel=sub, icon=icon,
            kind="scenario", payload=k, accent=acc,
        ))
        y += 50

    y += 16 + 22
    env = [
        ("earthquake", "🌋", "Earthquake",      "PGA 0.30 g · Kanai-Tajimi", RED),
        ("big_quake",  "💥", "Major quake",     "PGA 0.60 g · destructive",  RED),
        ("wind",       "💨", "Turbulent wind",  "V̄ = 25 m/s · I = 12 %",    CYAN),
        ("pedestrians","🚶", "Pedestrian crowd","200 walkers · ≈ f₁",       CYAN),
    ]
    for k, icon, lbl, sub, acc in env:
        buttons.append(Button(
            rect=Rect(16, y, LEFT_W - 32, 44),
            label=lbl, sublabel=sub, icon=icon,
            kind="scenario", payload=k, accent=acc,
        ))
        y += 50

    y += 16 + 22
    buttons.append(Button(
        rect=Rect(16, y, LEFT_W - 32, 44),
        label="Truck + Quake", sublabel="Worst-case combo", icon="🚛",
        kind="scenario", payload="combo", accent=RED,
    ))

    # Right panel upgrades (positioned after readouts; use generous y offset)
    panel_left = WIN_W - RIGHT_W
    y_right = HEADER_H + 14 + 24 + 56 + 10 + 2*56 + 6 + 64 + 12 + 18
    upgrades = [
        ("hs_steel",     "⚙", "High-strength steel","σ_y → 690 MPa"),
        ("reinforce",    "📏", "Deeper section",     "+50 % depth · ×1.5 W"),
        ("central_pier", "🏛", "Central pier",       "2-span continuous"),
        ("tmd",          "⚖", "Tuned-mass damper",  "ζ_eff 2 % → 8 %"),
    ]
    for k, icon, lbl, sub in upgrades:
        buttons.append(Button(
            rect=Rect(panel_left + 16, y_right, RIGHT_W - 32, 44),
            label=lbl, sublabel=sub, icon=icon,
            kind="toggle", payload=k, accent=CYAN,
        ))
        y_right += 50

    # Footer pill controls
    fy = WIN_H - FOOTER_H + (FOOTER_H - 36) // 2
    buttons.append(Button(
        rect=Rect(22, fy, 110, 36),
        label="⏸ Pause", kind="pill", payload="play",
        color=CYAN, accent=BG_0, border_color=CYAN,
    ))
    buttons.append(Button(
        rect=Rect(140, fy, 130, 36),
        label="↺ Reset bridge", kind="pill", payload="reset",
        color=BG_2, accent=INK, border_color=LINE,
    ))
    buttons.append(Button(
        rect=Rect(278, fy, 130, 36),
        label="✕ Clear loads", kind="pill", payload="clear",
        color=(40, 18, 22), accent=RED, border_color=(120, 60, 60),
    ))
    # speed - / +
    buttons.append(Button(
        rect=Rect(WIN_W - 260, fy + 2, 30, 32),
        label="−", kind="speed", payload="speed_down",
    ))
    buttons.append(Button(
        rect=Rect(WIN_W - 70, fy + 2, 30, 32),
        label="+", kind="speed", payload="speed_up",
    ))
    return buttons


# =============================================================================
# Main loop
# =============================================================================
def main():
    pygame.init()
    pygame.display.set_caption("Bridge Vibration Simulator · Pygame Edition")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    # Build fonts
    SANS = ["Plus Jakarta Sans", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
    DISP = ["Big Shoulders Display", "Bebas Neue", "Impact", "Arial Black", "DejaVu Sans"]
    MONO = ["JetBrains Mono", "Consolas", "Menlo", "DejaVu Sans Mono"]
    EMO  = ["Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji",
            "Symbola", "DejaVu Sans", "Arial"]

    FONTS.update({
        "title":      _font(DISP, 22, bold=True),
        "huge":       _font(DISP, 32, bold=True),
        "banner":     _font(DISP, 72, bold=True, scale=False),
        "big":        _font(DISP, 22, bold=True),
        "ui":         _font(SANS, 11),
        "ui_b":       _font(SANS, 13, bold=True),
        "mono9":      _font(MONO, 9),
        "mono10":     _font(MONO, 10),
        "mono10_upper": _font(MONO, 10),
        "mono11":     _font(MONO, 11),
        "mono12":     _font(MONO, 12),
        "icon":       _font(EMO, 18),
    })

    s = State()
    recompute_bridge(s)
    buttons = build_buttons(s)
    deferred: list[tuple] = []          # (when_t, callable)
    spark_pts: deque = deque(maxlen=400)

    running = True
    while running:
        real_dt = clock.tick(FPS) / 1000.0
        mouse_pos = pygame.mouse.get_pos()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for b in buttons:
                    if not hit(b, ev.pos):
                        continue
                    if b.kind == "scenario":
                        start_scenario(s, b.payload, deferred)
                    elif b.kind == "toggle":
                        s.ups[b.payload] = not s.ups[b.payload]
                        b.active = s.ups[b.payload]
                        if b.payload == "reinforce":
                            s.depth_mul = 1.5 if s.ups["reinforce"] else 1.0
                        recompute_bridge(s)
                    elif b.kind == "pill":
                        if b.payload == "play":
                            s.paused = not s.paused
                            b.label = "▶ Play" if s.paused else "⏸ Pause"
                        elif b.payload == "reset":
                            reset_bridge(s)
                        elif b.payload == "clear":
                            clear_loads(s)
                    elif b.kind == "speed":
                        if b.payload == "speed_up":
                            s.speed = min(8.0, s.speed * 2)
                        else:
                            s.speed = max(0.25, s.speed / 2)
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    s.paused = not s.paused
                    for b in buttons:
                        if b.payload == "play":
                            b.label = "▶ Play" if s.paused else "⏸ Pause"

        # advance simulation
        if not s.paused:
            sim_dt = real_dt * s.speed
            sub_steps = max(1, int(math.ceil(sim_dt / 0.005)))
            dt = sim_dt / sub_steps
            for _ in range(sub_steps):
                # deferred scenario events
                still = []
                for when, fn in deferred:
                    if s.t >= when:
                        fn()
                    else:
                        still.append((when, fn))
                deferred[:] = still

                if not s.collapsed:
                    for v in s.vehicles:
                        v.x += v.speed * dt
                    s.vehicles = [v for v in s.vehicles
                                  if v.x < L + v.length * 2 + 5]
                    recompute_static_from_vehicles(s)
                    step_dynamic(s, dt)

                    if s.earthquake.active:
                        ag = eq_accel(s)
                        s.shake_x = ag * 0.6
                        s.shake_y = ag * 0.3
                    else:
                        s.shake_x *= 0.8
                        s.shake_y *= 0.8

                    if s.wind_active and s.t - s.wind_t0 > 30:
                        s.wind_active = False
                    if s.peds_active and s.t - s.peds_t0 > 30:
                        s.peds_active = False
                    if s.earthquake.active and s.t - s.earthquake.t0 > s.earthquake.duration:
                        s.earthquake.active = False

                    Mt = effective_moment_max(s)
                    cap_Nm = s.capacity_kNm * 1000
                    if Mt > cap_Nm:
                        reason = (
                            f"Yield exceeded under seismic excitation (M = {Mt/1000:.0f} "
                            f"kN·m > M_cap = {s.capacity_kNm:.0f} kN·m)"
                            if s.earthquake.active
                            else
                            f"Yield exceeded under load (M = {Mt/1000:.0f} kN·m > "
                            f"M_cap = {s.capacity_kNm:.0f} kN·m)"
                        )
                        trigger_collapse(s, reason)
                else:
                    ground_wy = 0.22
                    for p in s.pieces:
                        p.vy += 9.81 * dt * 0.30
                        p.x += p.vx * dt
                        p.y += p.vy * dt
                        p.theta += p.omega * dt
                        if p.y > ground_wy:
                            p.y = ground_wy
                            p.vy *= -0.25
                            p.vx *= 0.7
                            p.omega *= 0.5
                            if abs(p.vy) < 0.3:
                                p.vy = 0
                s.t += dt

        # spark history
        spark_pts.append(midspan_mm(s))
        # capacity fraction
        cap_Nm = s.capacity_kNm * 1000
        s.last_cap_frac = effective_moment_max(s) / cap_Nm if cap_Nm > 0 else 0.0

        # --- draw ---
        screen.fill(BG_0)

        # toggle button states reflect upgrades
        for b in buttons:
            if b.kind == "toggle":
                b.active = s.ups[b.payload]

        stage = Rect(LEFT_W, HEADER_H, WIN_W - LEFT_W - RIGHT_W, WIN_H - HEADER_H - FOOTER_H)
        draw_grid(screen, stage, s)
        draw_wind(screen, stage, s)
        draw_supports(screen, stage, s)
        if s.collapsed:
            draw_collapsed(screen, stage, s)
        else:
            draw_deck(screen, stage, s)
            draw_vehicles(screen, stage, s)
            draw_pedestrians(screen, stage, s)
        draw_quake_tint(screen, stage, s)
        draw_overlay(screen, stage, s)
        draw_fail_banner(screen, stage, s)

        draw_left_panel(screen, s, buttons, mouse_pos)
        draw_right_panel(screen, s, buttons, mouse_pos, spark_pts)
        draw_header(screen, s)
        draw_footer(screen, buttons, mouse_pos, s)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
