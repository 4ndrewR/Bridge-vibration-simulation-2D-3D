"""
Interactive bridge-vibration viewer.

Opens a Matplotlib window with:
  - top panel : animated deflected bridge shape (deformation scaled ×500)
  - middle    : scrolling mid-span time history
  - right     : buttons to switch between simulation scenarios
                ('Truck', 'Earthquake', 'Wind', 'Pedestrians', 'Traffic',
                 'Truck + Earthquake')
  - bottom    : play / pause / restart controls

Each scenario is precomputed with Newmark-β when first selected, then the
animation plays back the stored time history in real time (with a small
speed factor for visibility).

Run with:
    python examples/visual_simulation.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge_sim import (
    Bridge, HEAVY_TRUCK, TRUCK, BUS, VAN, CAR, Vehicle,
    moving_load, traffic_stream,
    Earthquake, WindBuffeting, PedestrianCrowd,
    combine, newmark_beta, modal_analysis,
)


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    name: str
    bridge: Bridge
    t: np.ndarray
    deflection_xt: np.ndarray   # (n_nodes, n_t) in mm
    midspan: np.ndarray         # (n_t,) in mm
    info: str


def build_bridge() -> Bridge:
    return Bridge(length=60.0, n_elements=40)


def simulate(name: str) -> SimResult:
    bridge = build_bridge()
    vdofs = bridge.nodal_vertical_dofs()

    if name == "Truck (44 t)":
        load = moving_load(bridge, HEAVY_TRUCK)
        duration = bridge.length / HEAVY_TRUCK.speed_mps + 5.0
        info = "Poids-lourd articulé de 44 t à 50 km/h"

    elif name == "Earthquake":
        eq = Earthquake(bridge, duration_s=20.0, dt=0.01, pga_g=0.30,
                        dominant_frequency_hz=2.5)
        load = eq
        duration = 20.0
        info = "Séisme synthétique Kanai-Tajimi  ·  PGA = 0.30 g"

    elif name == "Wind":
        wind = WindBuffeting(bridge, mean_speed_mps=25.0,
                             turbulence_intensity=0.12,
                             duration_s=20.0, dt_sample=0.01)
        load = wind
        duration = 20.0
        info = "Vent turbulent  ·  V̄ = 25 m/s   I_u = 12 %"

    elif name == "Pedestrians":
        modes = modal_analysis(bridge, n_modes=2)
        crowd = PedestrianCrowd(bridge, n_pedestrians=200,
                                mean_frequency_hz=modes.natural_frequencies_hz[0])
        load = crowd
        duration = 20.0
        info = "Foule synchronisée 200 piétons près de f₁"

    elif name == "Traffic":
        fleet = [
            Vehicle(CAR.mass_kg,         CAR.speed_mps,         entry_time_s=0.0),
            Vehicle(VAN.mass_kg,         VAN.speed_mps,         entry_time_s=0.6),
            Vehicle(BUS.mass_kg,         BUS.speed_mps,         entry_time_s=1.4),
            Vehicle(TRUCK.mass_kg,       TRUCK.speed_mps,       entry_time_s=2.6),
            Vehicle(HEAVY_TRUCK.mass_kg, HEAVY_TRUCK.speed_mps, entry_time_s=4.0),
            Vehicle(CAR.mass_kg,         CAR.speed_mps,         entry_time_s=4.6),
        ]
        load = traffic_stream(bridge, fleet)
        duration = 12.0
        info = "Flux mixte : 2 voitures · 1 fourgon · 1 bus · 2 PL"

    elif name == "Truck + Earthquake":
        truck = moving_load(bridge, HEAVY_TRUCK)
        eq = Earthquake(bridge, duration_s=15.0, dt=0.01, pga_g=0.20,
                        dominant_frequency_hz=2.5)
        load = combine(truck, eq)
        duration = 15.0
        info = "Scénario combiné : poids-lourd + séisme (0.20 g)"

    else:
        raise ValueError(f"Unknown scenario {name!r}")

    print(f"Computing scenario {name!r}  ({duration:.1f} s)...")
    th = newmark_beta(bridge, load, duration_s=duration, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    defl = th.u[vdofs, :] * 1000.0
    return SimResult(
        name=name,
        bridge=bridge,
        t=th.t,
        deflection_xt=defl,
        midspan=defl[bridge.midspan_node(), :],
        info=info,
    )


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
class BridgeViewer:
    SCENARIOS = [
        "Truck (44 t)", "Earthquake", "Wind",
        "Pedestrians", "Traffic", "Truck + Earthquake",
    ]

    def __init__(self):
        self.cache: dict[str, SimResult] = {}
        self.current: SimResult | None = None
        self.frame = 0
        self.paused = False
        self.speed = 4         # play 1 of every `speed` precomputed frames
        self._build_figure()
        self.load_scenario(self.SCENARIOS[0])
        self.anim = FuncAnimation(self.fig, self._update, interval=33,
                                  blit=False, cache_frame_data=False)

    # ---- figure layout ------------------------------------------------
    def _build_figure(self):
        plt.rcParams.update({
            "figure.dpi": 100,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#34495E",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7E9",
        })
        self.fig = plt.figure(figsize=(13, 7))
        self.fig.canvas.manager.set_window_title("Bridge Vibration Simulator")

        gs = self.fig.add_gridspec(
            nrows=2, ncols=2,
            width_ratios=[4, 1], height_ratios=[3, 2],
            left=0.06, right=0.78, bottom=0.13, top=0.92,
            hspace=0.35, wspace=0.20,
        )
        self.ax_bridge = self.fig.add_subplot(gs[0, 0])
        self.ax_th     = self.fig.add_subplot(gs[1, 0])

        # Info text gets its own axes at the top of the right column;
        # buttons go below in a clearly separated area.
        self.ax_side = self.fig.add_axes([0.80, 0.55, 0.18, 0.37])
        self.ax_side.axis("off")

        self.ax_bridge.set_xlabel("Position x [m]")
        self.ax_bridge.set_ylabel("Déformation verticale [mm × 500]")
        self.ax_bridge.set_xlim(0, 60)

        self.ax_th.set_xlabel("Temps t [s]")
        self.ax_th.set_ylabel("Déplacement mi-travée [mm]")

        # Placeholder artists (real data filled in load_scenario)
        self.line_bridge, = self.ax_bridge.plot([], [], color="#1E2761", lw=3)
        self.fill_bridge = None
        self.line_th,     = self.ax_th.plot([], [], color="#C0392B", lw=1.5)
        self.marker_t,    = self.ax_th.plot([], [], "o", color="#1E2761", ms=7)

        self.txt_info = self.ax_side.text(
            0.02, 0.97, "", transform=self.ax_side.transAxes,
            fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4F6F7",
                      edgecolor="#34495E"),
        )

        # Scenario buttons (right column, below the info text)
        self.buttons: list[Button] = []
        for i, name in enumerate(self.SCENARIOS):
            y = 0.46 - i * 0.065
            ax_btn = self.fig.add_axes([0.80, y, 0.18, 0.05])
            btn = Button(ax_btn, name, color="#CADCFC", hovercolor="#5BA3D0")
            btn.on_clicked(lambda evt, n=name: self.load_scenario(n))
            self.buttons.append(btn)

        # Play / pause / restart buttons (bottom)
        ax_pp = self.fig.add_axes([0.30, 0.04, 0.10, 0.05])
        self.btn_pp = Button(ax_pp, "Pause", color="#FAD7A0", hovercolor="#E67E22")
        self.btn_pp.on_clicked(self._toggle_pause)

        ax_rst = self.fig.add_axes([0.42, 0.04, 0.10, 0.05])
        self.btn_rst = Button(ax_rst, "Restart", color="#A9DFBF", hovercolor="#1E8449")
        self.btn_rst.on_clicked(self._restart)

        ax_slow = self.fig.add_axes([0.54, 0.04, 0.07, 0.05])
        self.btn_slow = Button(ax_slow, "← slow", color="#E5E7E9")
        self.btn_slow.on_clicked(lambda e: self._change_speed(-1))
        ax_fast = self.fig.add_axes([0.63, 0.04, 0.07, 0.05])
        self.btn_fast = Button(ax_fast, "fast →", color="#E5E7E9")
        self.btn_fast.on_clicked(lambda e: self._change_speed(+1))

    # ---- buttons -------------------------------------------------------
    def _toggle_pause(self, _evt):
        self.paused = not self.paused
        self.btn_pp.label.set_text("Play" if self.paused else "Pause")

    def _restart(self, _evt):
        self.frame = 0

    def _change_speed(self, delta: int):
        self.speed = int(np.clip(self.speed + delta, 1, 12))

    # ---- scenario loading ---------------------------------------------
    def load_scenario(self, name: str):
        if name not in self.cache:
            self.cache[name] = simulate(name)
        self.current = self.cache[name]
        self.frame = 0
        sim = self.current

        x = sim.bridge.node_x
        ymax = max(abs(sim.deflection_xt.min()), abs(sim.deflection_xt.max())) * 1.10
        ymax = max(ymax, 1.0) * 500  # display amplification factor
        self.ax_bridge.set_ylim(-ymax, ymax)
        self.ax_bridge.set_title(f"Simulation : {sim.name}")
        self.line_bridge.set_data(x, sim.deflection_xt[:, 0] * 500)
        if self.fill_bridge is not None:
            self.fill_bridge.remove()
        self.fill_bridge = self.ax_bridge.fill_between(
            x, 0, sim.deflection_xt[:, 0] * 500, color="#CADCFC", alpha=0.5,
        )

        self.ax_th.set_xlim(0, sim.t[-1])
        m_min = sim.midspan.min() * 1.10
        m_max = sim.midspan.max() * 1.10
        pad = max(abs(m_min), abs(m_max)) * 0.1 + 1e-9
        self.ax_th.set_ylim(m_min - pad, m_max + pad)
        self.line_th.set_data(sim.t, sim.midspan)

        peak = float(np.max(np.abs(sim.midspan)))
        modes = modal_analysis(sim.bridge, n_modes=3)
        self.txt_info.set_text(
            f"Scénario\n {sim.info}\n\n"
            f"Pont 60 m, acier S355\n"
            f"40 éléments FEM (82 DOF)\n\n"
            f"Fréquences propres\n"
            f"  f₁ = {modes.natural_frequencies_hz[0]:.2f} Hz\n"
            f"  f₂ = {modes.natural_frequencies_hz[1]:.2f} Hz\n"
            f"  f₃ = {modes.natural_frequencies_hz[2]:.2f} Hz\n\n"
            f"Réponse simulée\n"
            f"  pic mi-travée : {peak:.2f} mm\n"
            f"  durée : {sim.t[-1]:.1f} s\n"
            f"  ζ = 2 %  (Rayleigh)"
        )

    # ---- frame update -------------------------------------------------
    def _update(self, _frame_idx):
        if self.paused or self.current is None:
            return
        sim = self.current
        n_t = len(sim.t)
        self.frame = (self.frame + self.speed) % n_t

        x = sim.bridge.node_x
        y_now = sim.deflection_xt[:, self.frame] * 500
        self.line_bridge.set_data(x, y_now)
        if self.fill_bridge is not None:
            self.fill_bridge.remove()
        self.fill_bridge = self.ax_bridge.fill_between(
            x, 0, y_now, color="#CADCFC", alpha=0.6,
        )
        self.marker_t.set_data([sim.t[self.frame]], [sim.midspan[self.frame]])
        self.ax_bridge.set_xlabel(
            f"Position x [m]    ·    t = {sim.t[self.frame]:.2f} s    "
            f"·    speed ×{self.speed}"
        )


def main():
    print("Starting interactive bridge-vibration viewer...")
    print("Scenarios are computed on-demand; the first click may take a few seconds.")
    viewer = BridgeViewer()
    plt.show()


if __name__ == "__main__":
    main()
