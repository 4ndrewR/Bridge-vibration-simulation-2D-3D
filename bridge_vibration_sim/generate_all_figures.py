"""
Generate every figure embedded in the project report and presentation.

Outputs (under figures/):
    - figure1_schematic_modes.png   -> docx Figure 1 + pptx slide 4
    - figure2_mode_shapes.png       -> docx Figure 2 + pptx slide 6
    - extra figures (truck, earthquake, wind, damage, traffic, FRF, animation)
      used in extra docx sections / appendix
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge_sim import (
    Bridge, HEAVY_TRUCK, TRUCK, BUS, VAN, CAR, Vehicle,
    moving_load, traffic_stream,
    Earthquake, WindBuffeting, PedestrianCrowd, HarmonicLoad,
    newmark_beta, modal_analysis, frequency_response,
    combine,
)
from bridge_sim.damage import damage_sweep
from bridge_sim.post import (
    plot_bridge_schematic_with_modes, plot_mode_shapes,
    plot_time_history, plot_spectrum, plot_frf, plot_damage_sweep,
    plot_envelope_and_moving_position, plot_3d_waterfall, animate_bridge,
    midspan_deflection, fft_spectrum, PALETTE, use_report_style,
)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def fp(name: str) -> str:
    return os.path.join(FIG_DIR, name)


def build_bridge() -> Bridge:
    return Bridge(length=60.0, n_elements=40)


# ----------------------------------------------------------------------------
# Figure 1 — schematic with mode overlay  (used in docx intro + pptx slide 4)
# ----------------------------------------------------------------------------
def figure1():
    bridge = build_bridge()
    modes = modal_analysis(bridge, n_modes=6)
    plot_bridge_schematic_with_modes(bridge, modes, fp("figure1_schematic_modes.png"),
                                     n_modes=4)
    print("[OK] figure1_schematic_modes.png")


# ----------------------------------------------------------------------------
# Figure 2 — three first mode shapes  (docx Figure 2 + pptx slide 6)
# ----------------------------------------------------------------------------
def figure2():
    bridge = build_bridge()
    modes = modal_analysis(bridge, n_modes=4)
    plot_mode_shapes(bridge, modes, fp("figure2_mode_shapes.png"), n_modes=3)
    print("[OK] figure2_mode_shapes.png")


# ----------------------------------------------------------------------------
# Figure 3 — truck crossing midspan deflection
# ----------------------------------------------------------------------------
def figure3_truck():
    bridge = build_bridge()
    load = moving_load(bridge, HEAVY_TRUCK)
    cross_t = bridge.length / HEAVY_TRUCK.speed_mps
    duration = cross_t + 6.0
    th = newmark_beta(bridge, load, duration_s=duration, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    w = midspan_deflection(bridge, th) * 1000
    plot_time_history(
        th.t, w, fp("figure3_truck_midspan.png"),
        title="Déflexion mi-travée — passage d'un poids-lourd articulé de 44 t (50 km/h)",
    )
    f, A = fft_spectrum(th.t, w)
    m = modal_analysis(bridge, n_modes=4)
    plot_spectrum(f, A, fp("figure3_truck_spectrum.png"), f_max_hz=20.0,
                  title="Spectre d'amplitude — passage poids-lourd",
                  natural_freqs_hz=m.natural_frequencies_hz[:4])
    print("[OK] figure3_truck_midspan.png  +  figure3_truck_spectrum.png")
    return float(np.max(np.abs(w)))


# ----------------------------------------------------------------------------
# Figure 4 — Earthquake response
# ----------------------------------------------------------------------------
def figure4_earthquake():
    import matplotlib.pyplot as plt
    bridge = build_bridge()
    eq = Earthquake(bridge, duration_s=30.0, dt=0.01, pga_g=0.30,
                    dominant_frequency_hz=2.5, damping_ground=0.6)
    th = newmark_beta(bridge, eq, duration_s=30.0, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    w = midspan_deflection(bridge, th) * 1000

    use_report_style()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True)
    axes[0].plot(eq.t_grid, eq.a_g / 9.81, color=PALETTE["red"])
    axes[0].set_ylabel("Accélération du sol [g]")
    axes[0].set_title("Accélérogramme synthétique de Kanai-Tajimi  —  PGA = 0.30 g, f_dom = 2.5 Hz")
    axes[0].axhline(0, color="#212121", lw=0.5, alpha=0.5)
    axes[1].plot(th.t, w, color=PALETTE["navy"])
    axes[1].set_xlabel("Temps t [s]")
    axes[1].set_ylabel("Déplacement mi-travée [mm]")
    axes[1].set_title("Réponse du pont au séisme")
    axes[1].axhline(0, color="#212121", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(fp("figure4_earthquake.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot_3d_waterfall(bridge, th, fp("figure4_earthquake_waterfall.png"),
                      n_snapshots=40,
                      title="Réponse spatio-temporelle du pont sous séisme")
    print("[OK] figure4_earthquake.png  +  figure4_earthquake_waterfall.png")
    return float(np.max(np.abs(w)))


# ----------------------------------------------------------------------------
# Figure 5 — Mixed traffic
# ----------------------------------------------------------------------------
def figure5_traffic():
    bridge = build_bridge()
    fleet = [
        Vehicle(CAR.mass_kg,         CAR.speed_mps,         entry_time_s=0.0,  label="car 1"),
        Vehicle(VAN.mass_kg,         VAN.speed_mps,         entry_time_s=0.6,  label="van"),
        Vehicle(BUS.mass_kg,         BUS.speed_mps,         entry_time_s=1.4,  label="bus"),
        Vehicle(TRUCK.mass_kg,       TRUCK.speed_mps,       entry_time_s=2.6,  label="truck"),
        Vehicle(HEAVY_TRUCK.mass_kg, HEAVY_TRUCK.speed_mps, entry_time_s=4.0,  label="44-t lorry"),
        Vehicle(CAR.mass_kg,         CAR.speed_mps,         entry_time_s=4.6,  label="car 2"),
    ]
    load = traffic_stream(bridge, fleet)
    th = newmark_beta(bridge, load, duration_s=12.0, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    w = midspan_deflection(bridge, th) * 1000
    plot_time_history(th.t, w, fp("figure5_traffic_midspan.png"),
                      title="Déplacement mi-travée — flux mixte de 6 véhicules")
    plot_envelope_and_moving_position(
        bridge, th, fp("figure5_traffic_envelope.png"),
        title="Enveloppe min/max — flux de trafic mixte",
    )
    print("[OK] figure5_traffic_midspan.png  +  figure5_traffic_envelope.png")
    return float(np.max(np.abs(w)))


# ----------------------------------------------------------------------------
# Figure 6 — Wind buffeting
# ----------------------------------------------------------------------------
def figure6_wind():
    bridge = build_bridge()
    wind = WindBuffeting(bridge, mean_speed_mps=25.0, turbulence_intensity=0.12,
                         duration_s=30.0, dt_sample=0.01)
    th = newmark_beta(bridge, wind, duration_s=30.0, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    w = midspan_deflection(bridge, th) * 1000
    plot_time_history(th.t, w, fp("figure6_wind_midspan.png"),
                      title="Déplacement mi-travée — vent turbulent (V = 25 m/s, I = 12 %)")
    f, A = fft_spectrum(th.t, w)
    m = modal_analysis(bridge, n_modes=4)
    plot_spectrum(f, A, fp("figure6_wind_spectrum.png"), f_max_hz=15.0,
                  title="Spectre de la réponse sous vent turbulent",
                  natural_freqs_hz=m.natural_frequencies_hz[:4])
    print("[OK] figure6_wind_midspan.png  +  figure6_wind_spectrum.png")
    return float(np.max(np.abs(w)))


# ----------------------------------------------------------------------------
# Figure 7 — Damage detection
# ----------------------------------------------------------------------------
def figure7_damage():
    bridge = build_bridge()
    mid_el = bridge.n_elements // 2
    losses = np.linspace(0.0, 0.6, 13)
    losses, freqs = damage_sweep(bridge, mid_el, losses, n_modes=4)
    plot_damage_sweep(losses, freqs, fp("figure7_damage_sensitivity.png"))
    print("[OK] figure7_damage_sensitivity.png")
    return losses, freqs


# ----------------------------------------------------------------------------
# Figure 8 — FRF
# ----------------------------------------------------------------------------
def figure8_frf():
    bridge = build_bridge()
    mid_dof = 2 * bridge.midspan_node()
    f_hz = np.linspace(0.1, 30.0, 800)
    H = frequency_response(bridge, mid_dof, mid_dof, f_hz, n_modes=8)
    plot_frf(f_hz, H, fp("figure8_frf.png"),
             title="Fonction de réponse en fréquence H(f) — mi-travée")
    print("[OK] figure8_frf.png")


# ----------------------------------------------------------------------------
# Figure 9 — Combined scenario animation
# ----------------------------------------------------------------------------
def figure9_animation():
    bridge = build_bridge()
    truck = moving_load(bridge, HEAVY_TRUCK)
    eq = Earthquake(bridge, duration_s=12.0, dt=0.01, pga_g=0.20,
                    dominant_frequency_hz=2.5)
    F = combine(truck, eq)
    th = newmark_beta(bridge, F, duration_s=12.0, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    animate_bridge(bridge, th, fp("figure9_animation.gif"),
                   fps=20, downsample=4,
                   title="Animation — poids-lourd + séisme")
    # also a still-frame plot for the docx
    w = midspan_deflection(bridge, th) * 1000
    plot_time_history(th.t, w, fp("figure9_combined_midspan.png"),
                      title="Combiné : poids-lourd + séisme (PGA = 0.20 g)")
    print("[OK] figure9_animation.gif  +  figure9_combined_midspan.png")


# ----------------------------------------------------------------------------
def main():
    figure1()
    figure2()
    w_truck = figure3_truck()
    w_eq    = figure4_earthquake()
    w_traf  = figure5_traffic()
    w_wind  = figure6_wind()
    losses, freqs = figure7_damage()
    figure8_frf()
    figure9_animation()

    print("\nSummary of peak mid-span deflections:")
    print(f"  truck (44 t)        : {w_truck:6.2f} mm")
    print(f"  mixed traffic       : {w_traf:6.2f} mm")
    print(f"  earthquake (0.30 g) : {w_eq:6.2f} mm")
    print(f"  wind  (25 m/s)      : {w_wind:6.2f} mm")
    print(f"  damage 60 % at mid  : f1 drops from {freqs[0,0]:.3f} -> {freqs[-1,0]:.3f} Hz")


if __name__ == "__main__":
    main()
