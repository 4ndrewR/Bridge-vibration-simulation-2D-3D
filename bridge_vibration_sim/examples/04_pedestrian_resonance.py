"""
Example 4 — Pedestrian crowd and harmonic resonance scan.

Two demonstrations:
    (a) 200 pedestrians walking at a near-synchronised pace (Millennium-bridge
        type problem) on a flexible footbridge.
    (b) Harmonic resonance scan: a unit harmonic load is swept across
        frequency; the mid-span amplitude is plotted to reveal the FRF peaks.
"""
import numpy as np

from bridge_sim import Bridge, PedestrianCrowd, newmark_beta, frequency_response
from bridge_sim.solver import modal_analysis
from bridge_sim.post import (
    plot_time_history, plot_frf, midspan_deflection, use_report_style, PALETTE,
)


def main():
    # Lighter, longer pedestrian bridge to put f1 close to ~2 Hz
    bridge = Bridge(length=80.0, n_elements=48)
    modes = modal_analysis(bridge, n_modes=4)
    print("Footbridge frequencies (Hz):", [f"{f:.2f}" for f in modes.natural_frequencies_hz[:4]])

    crowd = PedestrianCrowd(
        bridge, n_pedestrians=200,
        mean_frequency_hz=modes.natural_frequencies_hz[0],  # near resonance
        freq_std_hz=0.10,
    )
    th = newmark_beta(bridge, crowd, duration_s=40.0, dt=0.01,
                      zeta_1=0.008, zeta_2=0.008)
    w_mid = midspan_deflection(bridge, th) * 1000
    plot_time_history(
        th.t, w_mid, "figures/04_pedestrian_resonance.png",
        title="Foule piétonne synchronisée — résonance proche du 1er mode",
    )

    # FRF scan at mid-span
    mid_dof = 2 * bridge.midspan_node()
    f_hz = np.linspace(0.1, 25.0, 600)
    H = frequency_response(bridge, mid_dof, mid_dof, f_hz,
                           zeta_1=0.015, zeta_2=0.015, n_modes=8)
    plot_frf(f_hz, H, "figures/04_frf_midspan.png",
             title="Fonction de réponse en fréquence — mi-travée (excitation/réponse colocalisées)")

    print(f"Max midspan amplitude under crowd: {np.max(np.abs(w_mid)):.2f} mm")


if __name__ == "__main__":
    main()
