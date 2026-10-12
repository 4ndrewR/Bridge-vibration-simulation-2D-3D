"""
Time-dependent loading scenarios for the bridge vibration simulation.

Each generator returns a callable F(t) -> ndarray of length n_dof, the global
external force vector consistent with the FE model.

Available loadings
------------------
- MovingLoad        : single moving point load (car, truck...)
- TrafficStream     : multiple vehicles travelling at varied speeds
- SprungMassVehicle : quarter-car model coupling with the bridge (VBI light)
- HarmonicLoad      : fixed sinusoidal point load (resonance scans)
- PedestrianCrowd   : superposition of pedestrians walking at ~2 Hz
- WindBuffeting     : turbulent distributed wind load (random + drag)
- Earthquake        : base acceleration -> equivalent inertial load
- ImpulseLoad       : short triangular pulse (impact / hammer test)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .model import Bridge


# -----------------------------------------------------------------------------
# Vehicle / traffic loads
# -----------------------------------------------------------------------------
@dataclass
class Vehicle:
    """A simple constant-weight vehicle moving along the bridge at a given speed."""
    mass_kg: float        # gross mass [kg]
    speed_mps: float      # forward speed [m/s]
    entry_time_s: float = 0.0
    label: str = "vehicle"

    @property
    def weight_N(self) -> float:
        return self.mass_kg * 9.81


# Convenient presets (gross weights are realistic for Lebanese roads / EN 1991-2)
CAR        = Vehicle(mass_kg=1_500,  speed_mps=22.0, label="passenger car (1.5 t @ 80 km/h)")
VAN        = Vehicle(mass_kg=3_500,  speed_mps=22.0, label="van (3.5 t @ 80 km/h)")
BUS        = Vehicle(mass_kg=18_000, speed_mps=16.7, label="bus (18 t @ 60 km/h)")
TRUCK      = Vehicle(mass_kg=32_000, speed_mps=16.7, label="rigid truck (32 t @ 60 km/h)")
HEAVY_TRUCK = Vehicle(mass_kg=44_000, speed_mps=13.9, label="articulated lorry (44 t @ 50 km/h)")


def moving_load(bridge: Bridge, vehicle: Vehicle) -> Callable[[float], np.ndarray]:
    """Single moving point load. Returns F(t)."""
    L = bridge.length
    W = vehicle.weight_N

    def F(t: float) -> np.ndarray:
        x = vehicle.speed_mps * (t - vehicle.entry_time_s)
        if x < 0.0 or x > L:
            return np.zeros(bridge.n_dof)
        return bridge.consistent_load_vector(x, -W)   # downward (negative)

    return F


def traffic_stream(bridge: Bridge, vehicles: Sequence[Vehicle]) -> Callable[[float], np.ndarray]:
    """Superposition of several moving vehicles."""
    funcs = [moving_load(bridge, v) for v in vehicles]

    def F(t: float) -> np.ndarray:
        out = np.zeros(bridge.n_dof)
        for f in funcs:
            out += f(t)
        return out

    return F


# -----------------------------------------------------------------------------
# Harmonic point load (resonance scans)
# -----------------------------------------------------------------------------
@dataclass
class HarmonicLoad:
    """F(t) = F0 * sin(2*pi*frequency*t) applied at x_position."""
    bridge: Bridge
    F0: float
    frequency_hz: float
    x_position: float
    phase_rad: float = 0.0

    def __call__(self, t: float) -> np.ndarray:
        omega = 2 * np.pi * self.frequency_hz
        return self.bridge.consistent_load_vector(
            self.x_position, self.F0 * np.sin(omega * t + self.phase_rad),
        )


# -----------------------------------------------------------------------------
# Pedestrian crowd (millennium-bridge-style synchronised walking)
# -----------------------------------------------------------------------------
@dataclass
class PedestrianCrowd:
    """
    Superposition of individual pedestrians walking on the bridge.

    Each pedestrian applies a near-harmonic vertical force at their current
    position. The walking frequency for each pedestrian is drawn from a
    Gaussian centred on `mean_frequency_hz` with `freq_std_hz` standard
    deviation, capturing partial synchronisation.
    """
    bridge: Bridge
    n_pedestrians: int = 200
    mean_frequency_hz: float = 2.0
    freq_std_hz: float = 0.15
    weight_per_person_N: float = 700.0
    dynamic_load_factor: float = 0.4   # DLF for vertical walking
    walking_speed_mps: float = 1.4
    seed: int = 7

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.freqs = rng.normal(self.mean_frequency_hz, self.freq_std_hz, self.n_pedestrians)
        self.phases = rng.uniform(0.0, 2 * np.pi, self.n_pedestrians)
        self.entry_times = rng.uniform(0.0, self.bridge.length / self.walking_speed_mps,
                                       self.n_pedestrians)

    def __call__(self, t: float) -> np.ndarray:
        F_total = np.zeros(self.bridge.n_dof)
        for k in range(self.n_pedestrians):
            x = self.walking_speed_mps * (t - self.entry_times[k])
            if x < 0 or x > self.bridge.length:
                continue
            # Walking force model: static + harmonic
            omega = 2 * np.pi * self.freqs[k]
            F = self.weight_per_person_N * (1 + self.dynamic_load_factor
                                            * np.sin(omega * t + self.phases[k]))
            F_total += self.bridge.consistent_load_vector(x, -F)
        return F_total


# -----------------------------------------------------------------------------
# Wind buffeting (turbulent random distributed load)
# -----------------------------------------------------------------------------
@dataclass
class WindBuffeting:
    """
    Simplified turbulent wind buffeting load applied at every node.

    The longitudinal wind speed at each node is generated once as a band-
    limited Gaussian process around a mean wind speed; the corresponding
    vertical lift / drag fluctuation per unit length is converted into a
    nodal force using the tributary length.
    """
    bridge: Bridge
    mean_speed_mps: float = 25.0    # 90 km/h, design wind
    turbulence_intensity: float = 0.12
    air_density: float = 1.225
    drag_coeff: float = 1.2
    deck_height_m: float = 2.5
    dt_sample: float = 0.01
    duration_s: float = 30.0
    seed: int = 11

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        n_t = int(self.duration_s / self.dt_sample) + 1
        self.t_grid = np.linspace(0.0, self.duration_s, n_t)
        nn = self.bridge.n_nodes
        # Cross-correlated turbulent fluctuations: low-pass filtered noise
        white = rng.standard_normal((n_t, nn))
        # cheap first-order low pass
        alpha = 0.05
        filt = np.zeros_like(white)
        filt[0] = white[0]
        for i in range(1, n_t):
            filt[i] = (1 - alpha) * filt[i - 1] + alpha * white[i]
        u_rms = self.turbulence_intensity * self.mean_speed_mps
        self.u_field = filt * u_rms

        # Tributary length per node (lumped)
        L_el = self.bridge.length / self.bridge.n_elements
        self.tributary_len = np.full(nn, L_el)
        self.tributary_len[0] *= 0.5
        self.tributary_len[-1] *= 0.5

    def __call__(self, t: float) -> np.ndarray:
        if t <= 0:
            idx = 0
        elif t >= self.duration_s:
            idx = -1
        else:
            idx = int(t / self.dt_sample)
        u_t = self.u_field[idx]                                 # turb fluctuation at every node
        v_tot = self.mean_speed_mps + u_t
        q = 0.5 * self.air_density * v_tot**2                   # dynamic pressure
        f_per_m = self.drag_coeff * self.deck_height_m * q       # N/m on each node
        # vertical component (assume cross wind => lifting): take 30% of drag as lift
        f_v_per_m = 0.30 * f_per_m
        F_nodes = f_v_per_m * self.tributary_len                 # nodal vertical force [N]
        F = np.zeros(self.bridge.n_dof)
        F[self.bridge.nodal_vertical_dofs()] = -F_nodes          # downward positive when v>0
        return F


# -----------------------------------------------------------------------------
# Earthquake ground motion -> equivalent inertial load
# -----------------------------------------------------------------------------
def _kanai_tajimi_ground_motion(
    duration_s: float, dt: float,
    pga_g: float = 0.30,
    omega_g_hz: float = 2.5, zeta_g: float = 0.6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic ground acceleration from a Kanai-Tajimi filter excited by white
    noise. The signal is windowed and scaled so its peak equals pga_g * g.
    """
    rng = np.random.default_rng(seed)
    n_t = int(duration_s / dt) + 1
    t = np.linspace(0.0, duration_s, n_t)
    w = rng.standard_normal(n_t)

    omega_g = 2 * np.pi * omega_g_hz
    # Discrete-time KT filter (2nd order) - integrate with Newmark-style sub-step
    a = np.zeros(n_t)
    v = np.zeros(n_t)
    x = np.zeros(n_t)
    for i in range(1, n_t):
        a[i] = -2 * zeta_g * omega_g * v[i - 1] - omega_g**2 * x[i - 1] + w[i]
        v[i] = v[i - 1] + dt * a[i]
        x[i] = x[i - 1] + dt * v[i]
    # The filtered acceleration is the response a + 2*zeta_g*omega_g*v + omega_g^2*x ~ w
    # but using x directly gives the band-limited signal we want.
    sig = -(2 * zeta_g * omega_g * v + omega_g**2 * x)

    # Modulating envelope (Jennings) to make it transient
    t1, t2 = 2.0, duration_s - 5.0
    env = np.ones_like(t)
    env[t < t1] = (t[t < t1] / t1) ** 2
    decay = np.exp(-0.18 * (t - t2))
    env[t > t2] = decay[t > t2]
    sig *= env

    peak = np.max(np.abs(sig))
    if peak > 0:
        sig *= (pga_g * 9.81) / peak
    return t, sig


@dataclass
class Earthquake:
    """
    Ground-motion excitation. Equivalent inertial load on the bridge is

        F_eq(t) = -M * r * a_g(t)

    where r is the influence vector (1 on every vertical DOF, 0 on rotations).
    """
    bridge: Bridge
    duration_s: float = 30.0
    dt: float = 0.01
    pga_g: float = 0.30
    dominant_frequency_hz: float = 2.5
    damping_ground: float = 0.6
    seed: int = 42
    # populated below
    t_grid: np.ndarray = field(init=False, default=None)
    a_g: np.ndarray = field(init=False, default=None)
    r: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.t_grid, self.a_g = _kanai_tajimi_ground_motion(
            self.duration_s, self.dt, self.pga_g,
            self.dominant_frequency_hz, self.damping_ground, self.seed,
        )
        r = np.zeros(self.bridge.n_dof)
        r[self.bridge.nodal_vertical_dofs()] = 1.0
        self.r = r
        self._M_r = self.bridge.M @ self.r

    def acceleration(self, t: float) -> float:
        if t <= 0:
            return float(self.a_g[0])
        if t >= self.duration_s:
            return float(self.a_g[-1])
        idx = int(t / self.dt)
        # linear interpolation between samples
        t0 = idx * self.dt
        frac = (t - t0) / self.dt
        return float(self.a_g[idx] + frac * (self.a_g[idx + 1] - self.a_g[idx]))

    def __call__(self, t: float) -> np.ndarray:
        return -self._M_r * self.acceleration(t)


# -----------------------------------------------------------------------------
# Impulse (hammer test)
# -----------------------------------------------------------------------------
@dataclass
class ImpulseLoad:
    """Half-sine impulse of amplitude F0 and duration T_imp at x_position."""
    bridge: Bridge
    F0: float
    duration_s: float
    x_position: float
    start_time_s: float = 0.5

    def __call__(self, t: float) -> np.ndarray:
        dt = t - self.start_time_s
        if dt < 0.0 or dt > self.duration_s:
            return np.zeros(self.bridge.n_dof)
        amp = self.F0 * np.sin(np.pi * dt / self.duration_s)
        return self.bridge.consistent_load_vector(self.x_position, -amp)


# -----------------------------------------------------------------------------
# Combined load helper
# -----------------------------------------------------------------------------
def combine(*loads: Callable[[float], np.ndarray]) -> Callable[[float], np.ndarray]:
    """Sum any number of load callables into a single F(t)."""
    def F(t: float) -> np.ndarray:
        return sum((ld(t) for ld in loads), np.zeros(loads[0](0.0).shape))
    return F
