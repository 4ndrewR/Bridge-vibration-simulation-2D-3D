# === BUILD_TAG v7 · 2026-06-04 · vehicle ride-height + 3D wind ===
"""
bridge_simulator_3d.py — 3D bridge vibration simulator built on Ursina/Panda3D.

A real-time 3D view of the bridge with deforming deck mesh, 3D vehicles
driving across, screen shake during earthquakes, and a rigid-body collapse
when capacity is exceeded. Same scenarios, upgrade system, live readouts
and physics as the 2D pygame version.

Run:
    pip install ursina
    python bridge_simulator_3d.py

Camera:
    right-mouse drag  -> orbit / look around
    scroll wheel      -> zoom
    WASD              -> pan target

Keys:
    space             -> pause / resume
    r                 -> reset bridge
    c                 -> clear loads
    esc / q           -> quit
"""
from __future__ import annotations

import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field

from panda3d.core import loadPrcFileData
loadPrcFileData("", "framebuffer-srgb #t")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "win-size 1480 880")
loadPrcFileData("", "window-title Bridge Vibration Simulator · 3D Edition")

from ursina import (
    Ursina, Entity, Mesh, Text, Button, color, mouse, held_keys, time,
    camera, application, window, scene, Vec2, Vec3, lerp, invoke, destroy,
    EditorCamera, DirectionalLight, AmbientLight, Sky,
)


# =============================================================================
# Palette + constants (mirrors the 2D simulator)
# =============================================================================
COL_BG_0   = color.rgb(7/255, 13/255, 23/255)
COL_BG_1   = color.rgb(12/255, 23/255, 38/255)
COL_BG_2   = color.rgb(20/255, 35/255, 54/255)
COL_BG_3   = color.rgb(30/255, 50/255, 71/255)
COL_LINE   = color.rgb(31/255, 53/255, 80/255)
COL_INK    = color.rgb(232/255, 240/255, 250/255)
COL_INK_D  = color.rgb(138/255, 160/255, 184/255)
COL_INK_M  = color.rgb(86/255, 112/255, 138/255)
COL_CYAN   = color.rgb(34/255, 211/255, 238/255)
COL_AMBER  = color.rgb(251/255, 191/255, 36/255)
COL_RED    = color.rgb(248/255, 113/255, 113/255)
COL_GREEN  = color.rgb(52/255, 211/255, 153/255)

# Physics
L = 60.0
SEG = 60
STEEL_E   = 2.10e11
STEEL_SY  = 355e6
HS_SY     = 690e6
RHO       = 7850.0

VEHICLES: dict[str, dict] = {
    "car":   dict(mass=1500,   length=4.5,  speed=22.0,   color=color.rgb(125/255, 211/255, 252/255)),
    "van":   dict(mass=3500,   length=5.5,  speed=22.0,   color=color.rgb(196/255, 181/255, 253/255)),
    "bus":   dict(mass=18000,  length=12.0, speed=16.7,   color=color.rgb(252/255, 211/255, 77/255)),
    "truck": dict(mass=32000,  length=11.0, speed=16.7,   color=color.rgb(251/255, 146/255, 60/255)),
    "heavy": dict(mass=44000,  length=16.0, speed=13.9,   color=color.rgb(248/255, 113/255, 113/255)),
    "super": dict(mass=100000, length=24.0, speed=11.1,   color=color.rgb(239/255, 68/255, 68/255)),
}

# Visual amplification factor on deflection (deflections are mm-scale; if
# rendered at true scale the deck would barely move on screen).
DEFL_VIS_SCALE = 25.0


# =============================================================================
# State
# =============================================================================
@dataclass
class VehicleState:
    kind: str
    x: float
    mass: float
    length: float
    speed: float
    color_rgb: tuple
    entity: object | None = None


@dataclass
class Earthquake:
    active: bool = False
    intensity: float = 0.0
    dominant: float = 2.5
    t0: float = 0.0
    duration: float = 0.0


@dataclass
class Piece:
    """Falling debris chunk after collapse."""
    pos: Vec3
    vel: Vec3
    omega: Vec3
    entity: object | None = None


@dataclass
class State:
    ups: dict = field(default_factory=lambda: {
        "hs_steel": False, "reinforce": False, "central_pier": False, "tmd": False
    })
    depth_mul: float = 1.0

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

    vehicles: list[VehicleState] = field(default_factory=list)
    earthquake: Earthquake = field(default_factory=Earthquake)
    wind_active: bool = False; wind_t0: float = 0.0
    peds_active: bool = False; peds_t0: float = 0.0

    amp: float = 0.0
    v_amp: float = 0.0
    static_defl: list[float] = field(default_factory=lambda: [0.0] * (SEG + 1))
    static_M_max: float = 0.0
    last_cap_frac: float = 0.0

    t: float = 0.0
    paused: bool = False
    sim_speed: float = 1.0
    shake_x: float = 0.0
    shake_y: float = 0.0
    collapsed: bool = False
    pieces: list[Piece] = field(default_factory=list)
    midspan_hist: deque = field(default_factory=lambda: deque(maxlen=400))
    scenario_label: str = "READY"
    fail_reason: str = ""

    eq_z1: float = 0.0
    eq_z2: float = 0.0


s = State()
deferred_actions: list[tuple] = []   # (when_t, callable)


# =============================================================================
# Physics (identical to 2D pygame version)
# =============================================================================
def recompute_bridge() -> None:
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
    s.capacity_kNm = (M_serv / 1000.0) * (4.0 if s.ups["central_pier"] else 1.0)

    s.EI = E * I; s.I = I; s.W = W; s.A = A
    s.muL = RHO * A; s.depth = depth; s.sig_y = sig_y; s.f1 = f1


def static_deflection(P: float, a: float) -> list[float]:
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


def max_static_moment(P: float, a: float) -> float:
    if s.ups["central_pier"]:
        Lh = L / 2
        ah = a if a < Lh else a - Lh
        bh = Lh - ah
        return P * ah * bh / Lh
    return P * a * (L - a) / L


def recompute_static_from_vehicles() -> None:
    for i in range(SEG + 1):
        s.static_defl[i] = 0.0
    M_max = 0.0
    for v in s.vehicles:
        if v.x < 0 or v.x > L:
            continue
        P = v.mass * 9.81
        y = static_deflection(P, v.x)
        for i in range(SEG + 1):
            s.static_defl[i] += y[i]
        M = max_static_moment(P, v.x)
        if M > M_max:
            M_max = M
    s.static_M_max = M_max


def eq_accel() -> float:
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


def step_dynamic(dt: float) -> None:
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
        a_g = eq_accel()
        F += (-a_g) * (2 / math.pi)

    if s.wind_active:
        F += (random.random() * 2 - 1) * 0.4 * math.sin(2 * math.pi * s.f1 * s.t + random.random() * 0.5)

    if s.peds_active:
        F += 0.10 * math.sin(2 * math.pi * s.f1 * s.t) + 0.03 * math.sin(2 * math.pi * s.f1 * s.t * 2 + 0.3)

    a = F - c * s.v_amp - k * s.amp
    s.v_amp += a * dt
    s.amp   += s.v_amp * dt


def total_defl_at(i: int) -> float:
    x = (i / SEG) * L
    return s.static_defl[i] + s.amp * math.sin(math.pi * x / L)


def midspan_mm() -> float:
    return total_defl_at(SEG // 2) * 1000.0


def effective_moment_max() -> float:
    M_dyn_pure = abs(s.amp) * s.EI * (math.pi / L) ** 2
    return s.static_M_max + M_dyn_pure * 15.0


# =============================================================================
# Boot Ursina
# =============================================================================
app = Ursina(borderless=False, fullscreen=False, vsync=True)
window.title = "Bridge Vibration Simulator · 3D Edition"
window.color = COL_BG_0
window.fps_counter.enabled = False
window.exit_button.visible = False
window.entity_counter.enabled = False
window.collider_counter.enabled = False


# Sky / lighting
Sky(color=color.rgb(8/255, 16/255, 28/255))
sun = DirectionalLight()
sun.position = (40, 60, -40)
sun.look_at(Vec3(0, 0, 0))
sun.color = color.rgb(1.0, 0.95, 0.85)
AmbientLight(color=color.rgba(0.4, 0.5, 0.6, 1))


# =============================================================================
# 3D scene
# =============================================================================
# Ground plane (with grid)
ground = Entity(
    model="plane", scale=(220, 1, 60),
    position=(0, -8, 0),
    color=color.rgb(10/255, 18/255, 30/255),
    texture="white_cube",
    texture_scale=(110, 30),
)
# A second darker overlay to look like noise / asphalt
Entity(model="plane", scale=(220, 1, 60),
       position=(0, -7.99, 0),
       color=color.rgba(34/255, 211/255, 238/255, 0.04))

# Coordinate-system note:
#  +x = along the bridge (0 to L is from x=-L/2 to +L/2 centred)
#  +y = up
#  +z = "into the screen" (we'll keep deck centred at z=0)

DECK_WIDTH = 5.0    # visual width
DECK_DEPTH = 1.5    # visual depth (height) of box section


def make_deck_mesh(depth):
    """Procedural mesh for the deck — N segments along length, 8 vertices per slice
    (4 top + 4 bottom). Vertices ordered: for each slice i=0..SEG:
      base+0 = top  -z (front-top-left when looking +x)
      base+1 = top  +z (front-top-right)
      base+2 = bot  +z
      base+3 = bot  -z
    Triangles connect consecutive slices.
    """
    verts: list[Vec3] = []
    tris: list[int] = []
    half_w = DECK_WIDTH / 2
    half_h = depth / 2
    for i in range(SEG + 1):
        x = (i / SEG) * L - L / 2
        verts.append(Vec3(x,  half_h, -half_w))   # 0
        verts.append(Vec3(x,  half_h,  half_w))   # 1
        verts.append(Vec3(x, -half_h,  half_w))   # 2
        verts.append(Vec3(x, -half_h, -half_w))   # 3
    for i in range(SEG):
        b = i * 4
        n = (i + 1) * 4
        # top:   b+0, n+0, n+1   ;  b+0, n+1, b+1
        tris += [b+0, n+0, n+1, b+0, n+1, b+1]
        # bottom (reverse winding): b+2, n+2, n+3 ; b+2, n+3, b+3
        tris += [b+2, n+2, n+3, b+2, n+3, b+3]
        # side +z: b+1, n+1, n+2 ; b+1, n+2, b+2
        tris += [b+1, n+1, n+2, b+1, n+2, b+2]
        # side -z: b+3, n+3, n+0 ; b+3, n+0, b+0
        tris += [b+3, n+3, n+0, b+3, n+0, b+0]

    # store the raw rest positions on the side for fast update
    rest = [Vec3(v.x, v.y, v.z) for v in verts]
    return verts, tris, rest


deck_verts, deck_tris, deck_rest = make_deck_mesh(DECK_DEPTH)
deck_mesh = Mesh(vertices=deck_verts, triangles=deck_tris, mode="triangle")
deck = Entity(model=deck_mesh, color=color.rgb(48/255, 70/255, 95/255))
deck.collider = None

# Stress-overlay strip running along the top of the deck.
strip_verts: list[Vec3] = []
strip_tris: list[int] = []
half_w = DECK_WIDTH / 2 - 0.15
for i in range(SEG + 1):
    x = (i / SEG) * L - L / 2
    strip_verts.append(Vec3(x, DECK_DEPTH / 2 + 0.10, -half_w))
    strip_verts.append(Vec3(x, DECK_DEPTH / 2 + 0.10,  half_w))
for i in range(SEG):
    b = i * 2
    n = b + 2
    strip_tris += [b, n, n + 1, b, n + 1, b + 1]
stress_strip = Entity(
    model=Mesh(vertices=strip_verts, triangles=strip_tris),
    color=color.rgba(52/255, 211/255, 153/255, 0.55),
)
stress_rest = [Vec3(v.x, v.y, v.z) for v in strip_verts]


# Supports (pin left, roller right)
def make_pin_support(x_pos):
    return Entity(
        model="cube",
        position=(x_pos, -3.5, 0),
        scale=(1.5, 7, DECK_WIDTH + 1),
        color=color.rgb(30/255, 50/255, 71/255),
    )

def make_pin_cap(x_pos):
    return Entity(
        model="cube",
        position=(x_pos, 0.05, 0),
        scale=(2.0, 0.4, DECK_WIDTH + 1.2),
        color=COL_CYAN,
    )

pin_left  = make_pin_support(-L/2)
pin_right = make_pin_support( L/2)
cap_left  = make_pin_cap(-L/2)
cap_right = make_pin_cap( L/2)
# decorative pin pivot at left
pivot_left = Entity(model="sphere", position=(-L/2, 0.4, 0), scale=0.5, color=COL_CYAN)
# roller wheels at right under cap
roller_wheels = []
for k in range(-2, 3):
    w_e = Entity(model="sphere", position=(L/2, -0.30, k*1.1), scale=0.35, color=COL_CYAN)
    roller_wheels.append(w_e)

# Central pier (created/destroyed depending on upgrade)
central_pier_ent: Entity | None = None
central_pier_cap: Entity | None = None
def update_central_pier():
    global central_pier_ent, central_pier_cap
    if s.ups["central_pier"]:
        if central_pier_ent is None:
            central_pier_ent = Entity(
                model="cube",
                position=(0, -3.5, 0),
                scale=(1.8, 7, DECK_WIDTH + 0.6),
                color=color.rgb(30/255, 50/255, 71/255),
            )
            central_pier_cap = Entity(
                model="cube",
                position=(0, 0.05, 0),
                scale=(2.2, 0.4, DECK_WIDTH + 1.0),
                color=COL_CYAN,
            )
    else:
        if central_pier_ent is not None:
            destroy(central_pier_ent); central_pier_ent = None
        if central_pier_cap is not None:
            destroy(central_pier_cap); central_pier_cap = None

# =============================================================================
# Approach ramps — sloped surfaces on each side of the bridge
# =============================================================================
RAMP_H_LEN   = 26.0   # horizontal extent of each ramp (m)
RAMP_DEPTH   = 0.6    # vertical thickness of ramp slab (m)
RAMP_GROUND  = -7.0   # y-coordinate where the ramp meets the ground

def _build_ramp(side: str) -> Entity:
    """side = 'left' or 'right'. Build a triangular-prism mesh ramp that
    rises from RAMP_GROUND at the far end up to the deck-top at the bridge end.
    """
    deck_top_y = DECK_DEPTH / 2.0 * s.depth_mul
    dw_half = DECK_WIDTH / 2.0
    if side == "left":
        x_bridge = -L / 2          # end touching the bridge
        x_far    = -L / 2 - RAMP_H_LEN
    else:
        x_bridge =  L / 2
        x_far    =  L / 2 + RAMP_H_LEN

    # Six vertices forming a triangular prism viewed from +Z (a sloped slab).
    # Top edge of the bridge-end is at deck_top_y; the far end sits at ground.
    A = Vec3(x_bridge, deck_top_y,            +dw_half)   # 0 near-top  front
    B = Vec3(x_bridge, deck_top_y,            -dw_half)   # 1 near-top  back
    C = Vec3(x_bridge, RAMP_GROUND,           +dw_half)   # 2 near-bot  front
    D = Vec3(x_bridge, RAMP_GROUND,           -dw_half)   # 3 near-bot  back
    E = Vec3(x_far,    RAMP_GROUND,           +dw_half)   # 4 far-bot   front
    F = Vec3(x_far,    RAMP_GROUND,           -dw_half)   # 5 far-bot   back

    verts = [A, B, C, D, E, F]
    # Triangles — wound CCW from outside the volume.
    tris = [
        # slanted top (the road): A B F E
        0, 1, 5,  0, 5, 4,
        # bottom (on ground): C E F D
        2, 4, 5,  2, 5, 3,
        # near vertical face (between bridge end and slope): A C D B
        0, 2, 3,  0, 3, 1,
        # front side (z=+w/2):  A E C   (triangle)
        0, 4, 2,
        # back side (z=-w/2):   B D F   (triangle)
        1, 3, 5,
    ]
    mesh = Mesh(vertices=verts, triangles=tris, mode="triangle")
    ramp = Entity(model=mesh, color=color.rgb(38/255, 56/255, 78/255))
    return ramp


ramp_left  = _build_ramp("left")
ramp_right = _build_ramp("right")



# Distance markers on ground every 10 m
for x_m in range(0, int(L) + 1, 10):
    x_pos = x_m - L/2
    Entity(model="cube", position=(x_pos, -7.94, DECK_WIDTH/2 + 1.5),
           scale=(0.06, 0.02, 0.6), color=color.rgba(34/255, 211/255, 238/255, 0.4))
    Text(text=f"{x_m}m",
         position=(x_pos / 35, -0.15, 0),     # screen space below the deck won't work; this is world
         scale=10, parent=scene,
         world_parent=scene, color=COL_INK_M,
         x=x_pos, y=-7.85, z=DECK_WIDTH/2 + 2.0,
         origin=(0, 0), billboard=True)


# =============================================================================
# Vehicles
# =============================================================================
def make_vehicle_entity(kind: str, col) -> Entity:
    """Create a vehicle as a group of unparented-scale parts under a single
    root Entity. The root has scale (1,1,1) so children render at their
    declared absolute size in metres."""
    info = VEHICLES[kind]
    length = info["length"]
    # Root is a transform-only Entity, scale 1 -> no multiplication.
    root = Entity()
    deep = color.rgb(15/255, 23/255, 42/255)
    grey = color.rgb(31/255, 41/255, 55/255)

    if kind == "car":
        h = 1.4
        # Lower box (chassis)
        Entity(parent=root, model="cube", color=col,
               scale=(length, h * 0.65, DECK_WIDTH * 0.55),
               position=(0, h * 0.15, 0))
        # Cabin (smaller box on top, slightly toward back)
        Entity(parent=root, model="cube",
               color=color.rgba(0.95, 0.95, 1.0, 0.9),
               scale=(length * 0.55, h * 0.45, DECK_WIDTH * 0.50),
               position=(-length * 0.05, h * 0.75, 0))
        wheels = [(-length * 0.30, -DECK_WIDTH * 0.30),
                  (-length * 0.30,  DECK_WIDTH * 0.30),
                  ( length * 0.30, -DECK_WIDTH * 0.30),
                  ( length * 0.30,  DECK_WIDTH * 0.30)]
        wheel_r = 0.40

    elif kind == "van":
        h = 1.9
        Entity(parent=root, model="cube", color=col,
               scale=(length, h, DECK_WIDTH * 0.55),
               position=(0, h * 0.5, 0))
        Entity(parent=root, model="cube",
               color=color.rgba(0.95, 0.95, 1.0, 0.9),
               scale=(length * 0.30, h * 0.5, DECK_WIDTH * 0.50),
               position=(length * 0.30, h * 0.75, 0))
        wheels = [(-length * 0.32, -DECK_WIDTH * 0.30),
                  (-length * 0.32,  DECK_WIDTH * 0.30),
                  ( length * 0.32, -DECK_WIDTH * 0.30),
                  ( length * 0.32,  DECK_WIDTH * 0.30)]
        wheel_r = 0.45

    elif kind == "bus":
        h = 2.6
        Entity(parent=root, model="cube", color=col,
               scale=(length, h, DECK_WIDTH * 0.55),
               position=(0, h * 0.5, 0))
        # window stripe
        Entity(parent=root, model="cube",
               color=color.rgba(0.95, 0.95, 1.0, 0.8),
               scale=(length * 0.92, h * 0.30, DECK_WIDTH * 0.56),
               position=(0, h * 0.65, 0))
        wheels = [(-length * 0.38, -DECK_WIDTH * 0.30),
                  (-length * 0.38,  DECK_WIDTH * 0.30),
                  ( length * 0.38, -DECK_WIDTH * 0.30),
                  ( length * 0.38,  DECK_WIDTH * 0.30)]
        wheel_r = 0.55

    elif kind == "truck":
        h = 2.8
        # cargo box (rear)
        Entity(parent=root, model="cube", color=grey,
               scale=(length * 0.68, h, DECK_WIDTH * 0.55),
               position=(-length * 0.16, h * 0.5, 0))
        # cab (front)
        Entity(parent=root, model="cube", color=col,
               scale=(length * 0.30, h * 0.9, DECK_WIDTH * 0.55),
               position=(length * 0.35, h * 0.45, 0))
        # windshield
        Entity(parent=root, model="cube",
               color=color.rgba(0.95, 0.95, 1.0, 0.8),
               scale=(length * 0.25, h * 0.35, DECK_WIDTH * 0.56),
               position=(length * 0.35, h * 0.70, 0))
        wheels = [(-length * 0.38, -DECK_WIDTH * 0.30),
                  (-length * 0.38,  DECK_WIDTH * 0.30),
                  ( length * 0.10, -DECK_WIDTH * 0.30),
                  ( length * 0.10,  DECK_WIDTH * 0.30),
                  ( length * 0.42, -DECK_WIDTH * 0.30),
                  ( length * 0.42,  DECK_WIDTH * 0.30)]
        wheel_r = 0.55

    elif kind in ("heavy", "super"):
        h = 3.0
        # trailer
        Entity(parent=root, model="cube", color=deep,
               scale=(length * 0.74, h, DECK_WIDTH * 0.55),
               position=(-length * 0.13, h * 0.5, 0))
        # logo stripe on trailer side (kept relatively small)
        Entity(parent=root, model="cube", color=col,
               scale=(length * 0.50, h * 0.30, DECK_WIDTH * 0.56),
               position=(-length * 0.12, h * 0.5, 0))
        # tractor cab (right end)
        Entity(parent=root, model="cube", color=col,
               scale=(length * 0.22, h * 0.92, DECK_WIDTH * 0.55),
               position=(length * 0.38, h * 0.46, 0))
        # windshield
        Entity(parent=root, model="cube",
               color=color.rgba(0.95, 0.95, 1.0, 0.8),
               scale=(length * 0.18, h * 0.35, DECK_WIDTH * 0.56),
               position=(length * 0.38, h * 0.70, 0))
        wheels = [(-length * 0.40, -DECK_WIDTH * 0.30),
                  (-length * 0.40,  DECK_WIDTH * 0.30),
                  (-length * 0.15, -DECK_WIDTH * 0.30),
                  (-length * 0.15,  DECK_WIDTH * 0.30),
                  ( length * 0.20, -DECK_WIDTH * 0.30),
                  ( length * 0.20,  DECK_WIDTH * 0.30),
                  ( length * 0.42, -DECK_WIDTH * 0.30),
                  ( length * 0.42,  DECK_WIDTH * 0.30)]
        wheel_r = 0.60

    else:
        Entity(parent=root, model="cube", color=col,
               scale=(length, 1.5, DECK_WIDTH * 0.55),
               position=(0, 0.75, 0))
        wheels = [(-length * 0.3, 0), (length * 0.3, 0)]
        wheel_r = 0.45

    for wx, wz in wheels:
        Entity(parent=root, model="sphere", color=deep,
               scale=(wheel_r, wheel_r, wheel_r),
               position=(wx, 0, wz))
    # Record the ride height so the simulator can seat the wheels on the
    # surface (the wheel centres sit at local y=0, so their bottoms are one
    # radius below — lift the whole vehicle by that to avoid a buried look).
    root.ride_height = wheel_r
    return root


def spawn(kind: str) -> None:
    if s.collapsed:
        return
    info = VEHICLES[kind]
    v = VehicleState(
        kind=kind,
        x=-RAMP_H_LEN - info["length"] - 1,
        mass=info["mass"],
        length=info["length"],
        speed=info["speed"],
        color_rgb=info["color"],
    )
    v.entity = make_vehicle_entity(kind, info["color"])
    s.vehicles.append(v)


def destroy_vehicle(v: VehicleState):
    if v.entity is not None:
        destroy(v.entity)
        v.entity = None

# =============================================================================
# Pedestrian crowd visuals
# =============================================================================
ped_entities: list[Entity] = []
ped_state: list[dict] = []  # per-pedestrian phase, lane, direction

def _make_pedestrian():
    """Tiny humanoid figure: thin body + small head, no parent-scale multiplication."""
    root = Entity()
    body_color = color.rgb(0.65 + random.random()*0.15,
                           0.65 + random.random()*0.15,
                           0.80)
    head_color = color.rgb(0.95, 0.85, 0.75)
    Entity(parent=root, model="cube", color=body_color,
           scale=(0.35, 1.4, 0.35), position=(0, 0.7, 0))
    Entity(parent=root, model="sphere", color=head_color,
           scale=(0.32, 0.32, 0.32), position=(0, 1.55, 0))
    return root


def spawn_pedestrian_visuals(n=28):
    global ped_entities, ped_state
    clear_pedestrian_visuals()
    for _ in range(n):
        e = _make_pedestrian()
        ped_entities.append(e)
        ped_state.append({
            "phase": random.random(),
            "speed": 0.012 + random.random() * 0.015,    # phase / sim-second
            "dir":   random.choice([-1, 1]),
            "lane":  (random.random() - 0.5) * (DECK_WIDTH - 0.8),
            "bob":   random.random() * math.pi * 2,
        })


def clear_pedestrian_visuals():
    global ped_entities, ped_state
    for e in ped_entities:
        try: destroy(e)
        except Exception: pass
    ped_entities = []
    ped_state = []


def update_pedestrian_visuals(dt):
    if not ped_entities:
        return
    deck_top_y = DECK_DEPTH * s.depth_mul / 2.0
    for ent, st in zip(ped_entities, ped_state):
        st["phase"] = (st["phase"] + st["speed"] * st["dir"]) % 1.0
        x_m = st["phase"] * L
        i = max(0, min(SEG, int(round(x_m / L * SEG))))
        dy = total_defl_at(i) * DEFL_VIS_SCALE
        # gait bounce (small) — also coupled to bridge f1 (Millennium effect)
        bob = 0.04 * math.sin(s.t * 2 * math.pi * 1.8 + st["bob"])
        ent.position = (x_m - L / 2,
                        deck_top_y + bob + s.shake_y * 0.3,
                        st["lane"])


# =============================================================================
# Turbulent-wind visuals (cyan streaks blowing across the span)
# =============================================================================
wind_particles_3d: list[Entity] = []

def _reset_wind_particle(pcl):
    pcl.x = -L * 0.75 - random.random() * L * 0.6
    pcl.y = -3.0 + random.random() * 15.0
    pcl.z = (random.random() - 0.5) * (DECK_WIDTH * 4.0)
    pcl.speed = 28.0 + random.random() * 22.0

def make_wind_particle():
    pcl = Entity(model="cube",
                 color=color.rgba(34/255, 211/255, 238/255, 0.45),
                 scale=(3.2, 0.06, 0.06))
    _reset_wind_particle(pcl)
    return pcl

def spawn_wind_visuals(n=45):
    global wind_particles_3d
    clear_wind_visuals()
    for _ in range(n):
        wind_particles_3d.append(make_wind_particle())

def clear_wind_visuals():
    global wind_particles_3d
    for pcl in wind_particles_3d:
        try: destroy(pcl)
        except Exception: pass
    wind_particles_3d = []

def update_wind_visuals(dt):
    if not wind_particles_3d:
        return
    for pcl in wind_particles_3d:
        pcl.x += pcl.speed * dt
        # slight vertical sway so the field looks turbulent, not laminar
        pcl.y += math.sin(s.t * 3 + pcl.z) * 6 * dt
        if pcl.x > L * 0.9:
            _reset_wind_particle(pcl)




# =============================================================================
# Scenarios + collapse
# =============================================================================
def start_scenario(name: str) -> None:
    global deferred_actions
    if s.collapsed:
        reset_bridge()
    if name in ("car", "van", "bus", "truck", "heavy"):
        spawn(name)
        s.scenario_label = name.upper()
    elif name == "convoy":
        seq = ["car", "van", "bus", "truck", "heavy", "car"]
        t0 = s.t
        for i, k in enumerate(seq):
            deferred_actions.append((t0 + i * 1.8, lambda kk=k: spawn(kk)))
        s.scenario_label = "MIXED TRAFFIC · 6 vehicles"
    elif name == "overload":
        t0 = s.t
        for k in range(5):
            deferred_actions.append((t0 + k * 3.0, lambda: spawn("super")))
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
        s.wind_active = True; s.wind_t0 = s.t
        spawn_wind_visuals()
        s.scenario_label = "WIND BUFFETING · 25 m/s"
    elif name == "pedestrians":
        s.peds_active = True; s.peds_t0 = s.t
        spawn_pedestrian_visuals()
        s.scenario_label = "PEDESTRIAN CROWD · 200"
    elif name == "combo":
        spawn("heavy")
        deferred_actions.append((s.t + 1.5, lambda: setattr(
            s, "earthquake",
            Earthquake(active=True, intensity=0.20, dominant=2.5, t0=s.t, duration=18.0),
        )))
        s.scenario_label = "COMBO · TRUCK + QUAKE"


def trigger_collapse(reason: str) -> None:
    if s.collapsed:
        return
    s.collapsed = True
    s.scenario_label = "COLLAPSED"
    s.fail_reason = reason
    clear_pedestrian_visuals()

    # break the deck into chunks
    deck.enabled = False
    stress_strip.enabled = False
    n = 12
    for k in range(n):
        x0 = (k       / n) * L - L/2
        x1 = ((k + 1) / n) * L - L/2
        xm = 0.5 * (x0 + x1)
        idx = int(round((xm + L/2) / L * SEG))
        ym = (s.static_defl[idx] if 0 <= idx <= SEG else 0) \
             + s.amp * math.sin(math.pi * (idx / SEG))
        outward = (xm) * 0.6
        piece_ent = Entity(
            model="cube",
            color=color.rgb(48/255, 70/255, 95/255),
            scale=(L / n * 0.92, DECK_DEPTH * s.depth_mul, DECK_WIDTH),
            position=(xm, -ym * DEFL_VIS_SCALE, 0),
        )
        # red emissive accent (slightly larger so it shows around the cube)
        Entity(model="cube",
               color=color.rgba(248/255, 113/255, 113/255, 0.35),
               scale=(1.04, 1.04, 1.04),
               parent=piece_ent)
        p = Piece(
            pos=Vec3(xm, -ym * DEFL_VIS_SCALE, 0),
            vel=Vec3(outward * 0.5 + (random.random() - 0.5) * 4,
                     2 + random.random() * 4,
                     (random.random() - 0.5) * 4),
            omega=Vec3((random.random() - 0.5) * 3,
                       (random.random() - 0.5) * 3,
                       (random.random() - 0.5) * 3),
            entity=piece_ent,
        )
        s.pieces.append(p)


def reset_bridge() -> None:
    for v in s.vehicles:
        destroy_vehicle(v)
    s.vehicles.clear()
    for p in s.pieces:
        if p.entity is not None:
            destroy(p.entity)
    s.pieces.clear()
    s.earthquake = Earthquake()
    s.wind_active = False; s.peds_active = False
    clear_pedestrian_visuals()
    clear_wind_visuals()
    s.amp = 0.0; s.v_amp = 0.0
    for i in range(SEG + 1):
        s.static_defl[i] = 0.0
    s.static_M_max = 0.0
    s.collapsed = False
    s.shake_x = s.shake_y = 0.0
    s.midspan_hist.clear()
    s.scenario_label = "READY"
    s.fail_reason = ""
    deck.enabled = True
    stress_strip.enabled = True


def clear_loads() -> None:
    for v in s.vehicles:
        destroy_vehicle(v)
    s.vehicles.clear()
    s.earthquake.active = False
    s.wind_active = False
    s.peds_active = False
    clear_pedestrian_visuals()
    clear_wind_visuals()
    for i in range(SEG + 1):
        s.static_defl[i] = 0.0
    s.static_M_max = 0.0
    s.scenario_label = "READY"


# =============================================================================
# UI overlay
# =============================================================================
# Helpers — put everything directly on camera.ui at absolute positions so
# Ursina's scale inheritance doesn't shrink the text inside small Quads.

aspect = window.aspect_ratio
HUD_LEFT_X   = -0.5 * aspect + 0.025
HUD_RIGHT_X  = +0.5 * aspect - 0.025
HUD_TOP_Y    = 0.498
HUD_BOT_Y    = -0.498
PANEL_W      = 0.22


def ui_text(text, pos, color=COL_INK, scale=1.0, origin=(0, 0)):
    # Force the label to render on top of any UI quad it overlaps. The
    # Panda3D depth test was discarding text that sat at the same Z as the
    # button background; disabling depth + moving to the 'fixed' bin with a
    # high sort value pushes the glyphs in front of everything else in the
    # UI layer.
    t = Text(parent=camera.ui, text=text, position=pos,
             color=color, scale=scale, origin=origin)
    t.set_depth_test(False)
    t.set_depth_write(False)
    t.set_bin("fixed", 100)
    return t


def ui_quad(pos, scale, color, collider=None):
    return Entity(parent=camera.ui, model="quad",
                  position=pos, scale=scale, color=color, collider=collider)


# Header bar
header_bg = ui_quad((0, HUD_TOP_Y - 0.0225), (aspect, 0.045), COL_BG_1)
ui_text("BRIDGE·SIM 3D",
        (HUD_LEFT_X + 0.025, HUD_TOP_Y - 0.020),
        color=COL_INK, scale=0.95, origin=(-0.5, 0))
ui_text("L = 60 m  ·  STEEL S355  ·  Ursina/Panda3D",
        (HUD_LEFT_X + 0.250, HUD_TOP_Y - 0.020),
        color=COL_INK_M, scale=0.55, origin=(-0.5, 0))

status_tag_bg = ui_quad((HUD_RIGHT_X - 0.075, HUD_TOP_Y - 0.022),
                        (0.115, 0.034),
                        color.rgba(34/255, 211/255, 238/255, 0.10))
status_tag = ui_text("NOMINAL",
                     (HUD_RIGHT_X - 0.075, HUD_TOP_Y - 0.020),
                     color=COL_CYAN, scale=0.85, origin=(0, 0))

# Footer bar
footer_bg = ui_quad((0, HUD_BOT_Y + 0.0225), (aspect, 0.045), COL_BG_1)


# --- Scenarios panel (left) ---
left_panel_centre_x = HUD_LEFT_X + PANEL_W / 2 + 0.008
left_panel_bg = ui_quad((left_panel_centre_x, 0),
                        (PANEL_W, 0.86),
                        color.rgba(12/255, 23/255, 38/255, 0.94))


scenario_btns = []


def add_section_title(text, x, y):
    # text is left-aligned, with a short cyan accent stub immediately to its left
    # Calculate the panel's left edge using PANEL_W (passed-in x is the panel centre)
    panel_left_edge = x - PANEL_W / 2 + 0.014
    Entity(parent=camera.ui, model="quad", color=COL_CYAN,
           position=(panel_left_edge, y), scale=(0.018, 0.002))
    ui_text(text, (panel_left_edge + 0.022, y),
            color=COL_INK_M, scale=0.7, origin=(-0.5, 0))


def add_scenario_button(label, sub, x, y, payload, accent):
    btn_w = PANEL_W - 0.020
    btn_h = 0.048
    # background (click target) — text is NOT parented here
    bg = ui_quad((x, y), (btn_w, btn_h), COL_BG_2, collider="box")
    bg.kind = "scenario"
    bg.payload = payload
    bg.hover_color = COL_BG_3
    bg.normal_color = COL_BG_2
    # accent bar on left
    Entity(parent=camera.ui, model="quad", color=accent,
           position=(x - btn_w / 2 + 0.004, y), scale=(0.005, btn_h * 0.85))
    # label + sublabel
    ui_text(label, (x - btn_w / 2 + 0.020, y + 0.010),
            color=COL_INK, scale=0.78, origin=(-0.5, 0))
    ui_text(sub, (x - btn_w / 2 + 0.020, y - 0.010),
            color=COL_INK_M, scale=0.62, origin=(-0.5, 0))
    scenario_btns.append(bg)
    return bg


y_cursor = HUD_TOP_Y - 0.080
add_section_title("EXCITATIONS", left_panel_centre_x, y_cursor)
y_cursor -= 0.030

scenarios = [
    ("car",      "Passenger car", "1.5 t · 80 km/h",  COL_CYAN),
    ("van",      "Van",           "3.5 t · 80 km/h",  COL_CYAN),
    ("bus",      "Bus",           "18 t · 60 km/h",   COL_CYAN),
    ("truck",    "Rigid truck",   "32 t · 60 km/h",   COL_CYAN),
    ("heavy",    "Heavy lorry",   "44 t · 50 km/h",   COL_AMBER),
    ("convoy",   "Mixed traffic", "6 vehicles",       COL_AMBER),
    ("overload", "Overload test", "100 t convoy",     COL_AMBER),
]
for k, lbl, sub, acc in scenarios:
    add_scenario_button(lbl, sub, left_panel_centre_x, y_cursor, k, acc)
    y_cursor -= 0.054

y_cursor -= 0.014
add_section_title("ENVIRONMENT", left_panel_centre_x, y_cursor)
y_cursor -= 0.030
env_scen = [
    ("earthquake",  "Earthquake",        "PGA 0.30 g · KT", COL_RED),
    ("big_quake",   "Major quake",       "PGA 0.60 g",      COL_RED),
    ("wind",        "Turbulent wind",    "V = 25 m/s",      COL_CYAN),
    ("pedestrians", "Pedestrian crowd",  "200 walkers",     COL_CYAN),
]
for k, lbl, sub, acc in env_scen:
    add_scenario_button(lbl, sub, left_panel_centre_x, y_cursor, k, acc)
    y_cursor -= 0.054

y_cursor -= 0.014
add_section_title("COMBINED", left_panel_centre_x, y_cursor)
y_cursor -= 0.030
add_scenario_button("Truck + Quake", "Worst-case combo",
                    left_panel_centre_x, y_cursor, "combo", COL_RED)


# --- Readouts panel (right) ---
right_panel_centre_x = HUD_RIGHT_X - PANEL_W / 2 - 0.008
right_panel_bg = ui_quad((right_panel_centre_x, 0),
                         (PANEL_W, 0.86),
                         color.rgba(12/255, 23/255, 38/255, 0.94))


y_r = HUD_TOP_Y - 0.080
add_section_title("LIVE READOUTS", right_panel_centre_x, y_r)
y_r -= 0.030

# Capacity used (big gauge)
gauge_h = 0.075
ui_quad((right_panel_centre_x, y_r - gauge_h / 2),
        (PANEL_W - 0.014, gauge_h), COL_BG_2)
ui_text("CAPACITY USED",
        (right_panel_centre_x - (PANEL_W - 0.020) / 2, y_r - 0.013),
        color=COL_INK_M, scale=0.62, origin=(-0.5, 0))
cap_pct_text = ui_text("0",
                       (right_panel_centre_x + (PANEL_W - 0.022) / 2 - 0.020,
                        y_r - 0.025),
                       color=COL_INK, scale=1.8, origin=(0.5, 0))
ui_text("%",
        (right_panel_centre_x + (PANEL_W - 0.022) / 2 - 0.005, y_r - 0.025),
        color=COL_INK_M, scale=0.8, origin=(-0.5, 0))
# bar
bar_y = y_r - 0.060
bar_w = PANEL_W - 0.030
ui_quad((right_panel_centre_x, bar_y), (bar_w, 0.010), COL_BG_3)
cap_bar = Entity(parent=camera.ui, model="quad", color=COL_GREEN,
                 position=(right_panel_centre_x - bar_w / 2, bar_y),
                 scale=(0.0, 0.010),
                 origin=(-0.5, 0))
y_r -= gauge_h + 0.010


# 2×2 mini stats
def add_mini_stat(label, x, y):
    cell_w = (PANEL_W - 0.020) / 2 - 0.003
    cell_h = 0.058
    ui_quad((x, y - cell_h / 2), (cell_w, cell_h), COL_BG_2)
    ui_text(label, (x - cell_w / 2 + 0.006, y - 0.010),
            color=COL_INK_M, scale=0.5, origin=(-0.5, 0))
    val = ui_text("0", (x - cell_w / 2 + 0.006, y - 0.038),
                  color=COL_INK, scale=1.1, origin=(-0.5, 0))
    return val

cell_dx = (PANEL_W - 0.020) / 4 + 0.0015
midspan_val_t = add_mini_stat("MID-SPAN (mm)",
                              right_panel_centre_x - cell_dx, y_r)
moment_val_t = add_mini_stat("MOMENT (kN·m)",
                             right_panel_centre_x + cell_dx, y_r)
y_r -= 0.068
stress_val_t = add_mini_stat("STRESS (MPa)",
                             right_panel_centre_x - cell_dx, y_r)
freq_val_t = add_mini_stat("FREQ f1 (Hz)",
                           right_panel_centre_x + cell_dx, y_r)
y_r -= 0.075


# Upgrades section
add_section_title("UPGRADES", right_panel_centre_x, y_r)
y_r -= 0.030

upgrade_btns = []
upgrade_badges = {}

def add_upgrade_toggle(label, sub, payload, y):
    btn_w = PANEL_W - 0.014
    btn_h = 0.054
    bg = ui_quad((right_panel_centre_x, y), (btn_w, btn_h), COL_BG_2,
                 collider="box")
    bg.kind = "toggle"
    bg.payload = payload
    bg.normal_color = COL_BG_2
    bg.active_color = color.rgba(15/255, 35/255, 55/255, 1)
    ui_text(label,
            (right_panel_centre_x - btn_w / 2 + 0.012, y + 0.010),
            color=COL_INK, scale=0.78, origin=(-0.5, 0))
    ui_text(sub,
            (right_panel_centre_x - btn_w / 2 + 0.012, y - 0.011),
            color=COL_INK_M, scale=0.60, origin=(-0.5, 0))
    badge = ui_text("ON",
                    (right_panel_centre_x + btn_w / 2 - 0.014, y),
                    color=COL_GREEN, scale=0.72, origin=(0.5, 0))
    badge.visible = False
    upgrade_btns.append(bg)
    upgrade_badges[payload] = badge
    return bg


ups_defs = [
    ("hs_steel",     "High-strength steel", "sigma_y: 355 -> 690 MPa"),
    ("reinforce",    "Deeper section",      "+50 % depth · ×1.5 W"),
    ("central_pier", "Central pier",        "2-span continuous"),
    ("tmd",          "Tuned-mass damper",   "zeta_eff: 2% -> 8%"),
]
for k, lbl, sub in ups_defs:
    add_upgrade_toggle(lbl, sub, k, y_r)
    y_r -= 0.060


# Capacity budget at bottom of right panel
y_r -= 0.005
add_section_title("CAPACITY BUDGET", right_panel_centre_x, y_r)
y_r -= 0.030
ui_quad((right_panel_centre_x, y_r - 0.040),
        (PANEL_W - 0.014, 0.080), COL_BG_2)
ui_text("M_CAP [kN·m]",
        (right_panel_centre_x - (PANEL_W - 0.020) / 2, y_r - 0.013),
        color=COL_INK_M, scale=0.60, origin=(-0.5, 0))
m_cap_t = ui_text("—",
                  (right_panel_centre_x + (PANEL_W - 0.020) / 2,
                   y_r - 0.025),
                  color=COL_INK, scale=1.2, origin=(0.5, 0))
ui_text("DAF (DYN.)",
        (right_panel_centre_x - (PANEL_W - 0.020) / 2, y_r - 0.052),
        color=COL_INK_M, scale=0.60, origin=(-0.5, 0))
daf_t = ui_text("25.0",
                (right_panel_centre_x + (PANEL_W - 0.020) / 2,
                 y_r - 0.062),
                color=COL_INK, scale=1.0, origin=(0.5, 0))


# Status overlay (mid stage)
scenario_t = ui_text("READY", (0, HUD_TOP_Y - 0.075),
                     color=COL_CYAN, scale=0.95, origin=(0, 0))
time_t = ui_text("t = 0.00 s", (0, HUD_TOP_Y - 0.100),
                 color=COL_INK_M, scale=0.7, origin=(0, 0))

# Camera-controls hint (bottom-centre, above footer)
ui_text("drag right-mouse or arrow keys to orbit  ·  scroll or +/- to zoom",
        (0, HUD_BOT_Y + 0.060), color=COL_INK_M, scale=0.6, origin=(0, 0))


# Failure banner
fail_banner_bg = ui_quad((0, 0.10), (0.70, 0.10),
                         color.rgba(248/255, 113/255, 113/255, 0.18))
fail_banner_bg.enabled = False
fail_banner_t = ui_text("STRUCTURAL FAILURE", (0, 0.115),
                        color=COL_RED, scale=2.4, origin=(0, 0))
fail_banner_t.enabled = False
fail_reason_t = ui_text("", (0, 0.080),
                        color=COL_INK_D, scale=0.7, origin=(0, 0))
fail_reason_t.enabled = False


# Footer controls — pill buttons, text not parented to backgrounds
btn_play = ui_quad((HUD_LEFT_X + PANEL_W + 0.06, HUD_BOT_Y + 0.022),
                   (0.10, 0.034), COL_CYAN, collider="box")
btn_play.kind = "play"
play_label_t = ui_text("|| Pause",
                      (HUD_LEFT_X + PANEL_W + 0.06, HUD_BOT_Y + 0.022),
                      color=COL_BG_0, scale=0.80, origin=(0, 0))

btn_reset = ui_quad((HUD_LEFT_X + PANEL_W + 0.18, HUD_BOT_Y + 0.022),
                    (0.10, 0.034), COL_BG_2, collider="box")
btn_reset.kind = "reset"
ui_text("<- Reset bridge",
        (HUD_LEFT_X + PANEL_W + 0.18, HUD_BOT_Y + 0.022),
        color=COL_INK, scale=0.78, origin=(0, 0))

btn_clear = ui_quad((HUD_LEFT_X + PANEL_W + 0.30, HUD_BOT_Y + 0.022),
                    (0.10, 0.034),
                    color.rgba(60/255, 18/255, 22/255, 1), collider="box")
btn_clear.kind = "clear"
ui_text("X  Clear loads",
        (HUD_LEFT_X + PANEL_W + 0.30, HUD_BOT_Y + 0.022),
        color=COL_RED, scale=0.78, origin=(0, 0))

# speed
speed_label_t = ui_text("SPEED",
                        (HUD_RIGHT_X - 0.08, HUD_BOT_Y + 0.022),
                        color=COL_INK_M, scale=0.7, origin=(0, 0))
speed_value_t = ui_text("1×",
                        (HUD_RIGHT_X - 0.030, HUD_BOT_Y + 0.022),
                        color=COL_CYAN, scale=0.9, origin=(0, 0))



# =============================================================================
# Camera (orbit controller around the bridge mid-span)
# =============================================================================
camera.parent = scene
camera_target = Vec3(0, 0, 0)
camera_yaw   = 25.0
camera_pitch = 18.0
camera_dist  = 140.0
camera.fov = 55


def update_camera():
    yaw_r   = math.radians(camera_yaw)
    pitch_r = math.radians(camera_pitch)
    cx = camera_target.x + camera_dist * math.cos(pitch_r) * math.sin(yaw_r)
    cy = camera_target.y + camera_dist * math.sin(pitch_r)
    cz = camera_target.z - camera_dist * math.cos(pitch_r) * math.cos(yaw_r)
    camera.position = (cx, cy, cz)
    camera.look_at(camera_target)


update_camera()


# =============================================================================
# Per-frame update
# =============================================================================
def update_deck_mesh():
    if s.collapsed:
        return
    # write new positions back into the deck mesh vertices
    half_h = DECK_DEPTH / 2 * s.depth_mul
    for i in range(SEG + 1):
        # world dy is in metres; apply visual scale
        dy = total_defl_at(i) * DEFL_VIS_SCALE
        b = i * 4
        rx = deck_rest[b].x
        deck_verts[b + 0] = Vec3(rx,  half_h - dy, deck_rest[b + 0].z)
        deck_verts[b + 1] = Vec3(rx,  half_h - dy, deck_rest[b + 1].z)
        deck_verts[b + 2] = Vec3(rx, -half_h - dy, deck_rest[b + 2].z)
        deck_verts[b + 3] = Vec3(rx, -half_h - dy, deck_rest[b + 3].z)

    deck.model.vertices = deck_verts
    deck.model.generate()

    # stress strip on top: same dy, fixed offset above deck
    for i in range(SEG + 1):
        dy = total_defl_at(i) * DEFL_VIS_SCALE
        b = i * 2
        strip_verts[b + 0] = Vec3(stress_rest[b + 0].x,
                                  half_h + 0.10 - dy,
                                  stress_rest[b + 0].z)
        strip_verts[b + 1] = Vec3(stress_rest[b + 1].x,
                                  half_h + 0.10 - dy,
                                  stress_rest[b + 1].z)
    stress_strip.model.vertices = strip_verts
    stress_strip.model.generate()
    # color by stress
    f = s.last_cap_frac
    if f < 0.55:
        c = color.rgba(52/255, 211/255, 153/255, 0.50)
    elif f < 0.85:
        c = color.rgba(251/255, 191/255, 36/255, 0.60)
    else:
        c = color.rgba(248/255, 113/255, 113/255, 0.65)
    stress_strip.color = c


def update_vehicle_visuals(real_dt: float):
    """Place each vehicle along the full left-ramp -> deck -> right-ramp path
    and pitch its body to match the local surface slope (ramp incline on the
    approaches, deflected-deck slope on the span) so it stays seated on the road."""
    deck_top = DECK_DEPTH * s.depth_mul / 2.0
    rise = deck_top - RAMP_GROUND

    def surf(vx: float) -> float:
        """Road-surface height at bridge-coordinate vx (ramps at vx<0 / vx>L)."""
        if vx < 0:
            frac = max(0.0, min(1.0, (vx + RAMP_H_LEN) / RAMP_H_LEN))
            return RAMP_GROUND + frac * rise
        if vx > L:
            frac = max(0.0, min(1.0, (vx - L) / RAMP_H_LEN))
            return deck_top - frac * rise
        # deck: linearly interpolate the deflection between nodes for a smooth slope
        t = vx / L * SEG
        i0 = max(0, min(SEG, int(math.floor(t))))
        i1 = min(SEG, i0 + 1)
        f = t - i0
        defl = total_defl_at(i0) * (1.0 - f) + total_defl_at(i1) * f
        return deck_top - defl * DEFL_VIS_SCALE

    eps = 0.6  # m, for the finite-difference slope of the surface
    for v in s.vehicles:
        if v.entity is None:
            continue
        world_x = v.x - L / 2.0
        surface_y = surf(v.x)
        # Pitch = negative of the surface angle. In this scene a positive
        # rotation_z drops the vehicle's front (+X), so to align the body with a
        # road that rises toward +X (front up) we apply a negative rotation.
        slope = (surf(v.x + eps) - surf(v.x - eps)) / (2.0 * eps)
        tilt = -math.degrees(math.atan(slope))
        lift = getattr(v.entity, "ride_height", 0.4)
        v.entity.position = (world_x, surface_y + lift + s.shake_y * 0.3, 0)
        v.entity.rotation = (0, 0, tilt)


def update_pieces(dt: float):
    """Collapse animation — falling deck chunks."""
    ground_y = -7.0
    for p in s.pieces:
        p.vel.y -= 9.81 * dt * 0.30
        p.pos += p.vel * dt
        p.entity.rotation = (p.entity.rotation[0] + p.omega.x * dt * 60,
                             p.entity.rotation[1] + p.omega.y * dt * 60,
                             p.entity.rotation[2] + p.omega.z * dt * 60)
        if p.pos.y < ground_y:
            p.pos.y = ground_y
            p.vel.y *= -0.25
            p.vel.x *= 0.7
            p.vel.z *= 0.7
            p.omega *= 0.5
            if abs(p.vel.y) < 0.5:
                p.vel.y = 0
                p.omega = Vec3(0, 0, 0)
        p.entity.position = p.pos


def update_ui():
    cap_used = s.last_cap_frac
    pct = max(0, min(100, int(cap_used * 100)))
    cap_pct_text.text = str(pct)
    # cap bar — width is in absolute UI units (scale_x of a non-parented Quad)
    bar_w_total = PANEL_W - 0.030
    cap_bar.scale_x = bar_w_total * min(1.0, cap_used)
    if pct < 55:   cap_bar.color = COL_GREEN
    elif pct < 85: cap_bar.color = COL_AMBER
    else:          cap_bar.color = COL_RED

    midspan_val_t.text = f"{midspan_mm():+.1f}"
    M = effective_moment_max() / 1000
    moment_val_t.text = f"{M:,.0f}"
    sigma = (M * 1000) / s.W / 1e6
    stress_val_t.text = f"{sigma:.0f}"
    freq_val_t.text = f"{s.f1:.2f}"
    m_cap_t.text = f"{s.capacity_kNm:,.0f}"
    daf_t.text = f"{1 / (2 * s.zeta):.1f}"

    if s.collapsed:
        status_tag.text = "FAILURE"; status_tag.color = COL_RED
    elif cap_used < 0.55:
        status_tag.text = "NOMINAL"; status_tag.color = COL_CYAN
    elif cap_used < 0.85:
        status_tag.text = "ELEVATED"; status_tag.color = COL_AMBER
    else:
        status_tag.text = "CRITICAL"; status_tag.color = COL_RED

    scenario_t.text = s.scenario_label
    time_t.text     = f"t = {s.t:.2f} s"
    speed_value_t.text = f"{s.sim_speed:g}×"

    # toggle button visuals
    for b in upgrade_btns:
        active = s.ups[b.payload]
        upgrade_badges[b.payload].visible = active
        b.color = b.active_color if active else b.normal_color

    fail_banner_bg.enabled = s.collapsed
    fail_banner_t.enabled  = s.collapsed
    fail_reason_t.enabled  = s.collapsed
    if s.collapsed:
        fail_reason_t.text = s.fail_reason


def update():
    """Ursina calls this every frame."""
    real_dt = time.dt

    if not s.paused:
        sim_dt = real_dt * s.sim_speed
        sub_steps = max(1, math.ceil(sim_dt / 0.005))
        dt = sim_dt / sub_steps
        for _ in range(sub_steps):
            # deferred scenario events
            still = []
            for when, fn in deferred_actions:
                if s.t >= when:
                    fn()
                else:
                    still.append((when, fn))
            deferred_actions[:] = still

            if not s.collapsed:
                for v in s.vehicles:
                    v.x += v.speed * dt
                # drop off-screen vehicles
                survivors = []
                for v in s.vehicles:
                    if v.x < L + RAMP_H_LEN + v.length * 2 + 5:
                        survivors.append(v)
                    else:
                        destroy_vehicle(v)
                s.vehicles = survivors

                recompute_static_from_vehicles()
                step_dynamic(dt)

                if s.earthquake.active:
                    ag = eq_accel()
                    s.shake_x = ag * 0.4
                    s.shake_y = ag * 0.25
                else:
                    s.shake_x *= 0.85
                    s.shake_y *= 0.85

                if s.wind_active and s.t - s.wind_t0 > 30:
                    s.wind_active = False
                    clear_wind_visuals()
                if s.peds_active and s.t - s.peds_t0 > 30:
                    s.peds_active = False
                    clear_pedestrian_visuals()
                if s.earthquake.active and s.t - s.earthquake.t0 > s.earthquake.duration:
                    s.earthquake.active = False

                Mt = effective_moment_max()
                cap_Nm = s.capacity_kNm * 1000
                if Mt > cap_Nm:
                    reason = (
                        f"Yield exceeded under seismic excitation "
                        f"(M = {Mt/1000:.0f} kN·m > M_cap = {s.capacity_kNm:.0f} kN·m)"
                        if s.earthquake.active else
                        f"Yield exceeded under load "
                        f"(M = {Mt/1000:.0f} kN·m > M_cap = {s.capacity_kNm:.0f} kN·m)"
                    )
                    trigger_collapse(reason)
            else:
                update_pieces(dt)
            s.t += dt

        s.midspan_hist.append(midspan_mm())
        cap_Nm = s.capacity_kNm * 1000
        s.last_cap_frac = effective_moment_max() / cap_Nm if cap_Nm > 0 else 0.0

    # Always update visuals
    if not s.collapsed:
        update_deck_mesh()
        update_vehicle_visuals(real_dt)
        update_pedestrian_visuals(real_dt)
    update_wind_visuals(real_dt)

    update_ui()

    # earthquake camera shake
    camera_target.x = s.shake_x * 0.3
    camera_target.y = -2 + s.shake_y * 0.3
    update_camera()

    # camera orbit via right-mouse drag
    global camera_yaw, camera_pitch, camera_dist
    moved = False
    if mouse.right:
        camera_yaw   += mouse.velocity[0] * 200
        camera_pitch += mouse.velocity[1] * 200
        moved = True
    # Arrow keys orbit; +/- (or =/-) zoom. held_keys gives smooth continuous motion.
    rot_speed = 60 * time.dt          # deg per second
    zoom_speed = 40 * time.dt
    if held_keys["left arrow"]:  camera_yaw   -= rot_speed; moved = True
    if held_keys["right arrow"]: camera_yaw   += rot_speed; moved = True
    if held_keys["up arrow"]:    camera_pitch += rot_speed; moved = True
    if held_keys["down arrow"]:  camera_pitch -= rot_speed; moved = True
    if held_keys["+"] or held_keys["="]:
        camera_dist = max(25, camera_dist - zoom_speed); moved = True
    if held_keys["-"] or held_keys["_"]:
        camera_dist = min(240, camera_dist + zoom_speed); moved = True
    if moved:
        camera_pitch = max(5, min(80, camera_pitch))
        update_camera()


# =============================================================================
# Input handling
# =============================================================================
def input(key):
    global camera_dist
    if key == "escape" or key == "q":
        application.quit()
    elif key == "space":
        s.paused = not s.paused
        play_label_t.text = ">  Play" if s.paused else "|| Pause"
    elif key == "r":
        reset_bridge()
    elif key == "c":
        clear_loads()
    elif key == "scroll up":
        camera_dist = max(25, camera_dist * 0.92)
        update_camera()
    elif key == "scroll down":
        camera_dist = min(240, camera_dist * 1.08)
        update_camera()
    elif key == "left mouse down":
        if not mouse.hovered_entity:
            return
        e = mouse.hovered_entity
        if getattr(e, "kind", None) == "scenario":
            start_scenario(e.payload)
        elif getattr(e, "kind", None) == "toggle":
            s.ups[e.payload] = not s.ups[e.payload]
            if e.payload == "reinforce":
                s.depth_mul = 1.5 if s.ups["reinforce"] else 1.0
            recompute_bridge()
            update_central_pier()
        elif getattr(e, "kind", None) == "play":
            s.paused = not s.paused
            play_label_t.text = ">  Play" if s.paused else "|| Pause"
        elif getattr(e, "kind", None) == "reset":
            reset_bridge()
        elif getattr(e, "kind", None) == "clear":
            clear_loads()


# =============================================================================
# Initial bridge setup + run
# =============================================================================
recompute_bridge()
update_central_pier()

app.run()
