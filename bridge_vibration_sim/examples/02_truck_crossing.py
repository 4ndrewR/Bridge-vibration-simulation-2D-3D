"""
Example 2 — Heavy truck crossing the bridge.

Simulates a 44-tonne articulated lorry crossing a 60-m simply-supported
steel bridge at 50 km/h. Plots the mid-span vertical deflection history
and the corresponding amplitude spectrum.
"""
import numpy as np

from bridge_sim import Bridge, HEAVY_TRUCK, moving_load, newmark_beta
from bridge_sim.post import (
    plot_time_history, plot_spectrum, midspan_deflection, fft_spectrum,
    plot_envelope_and_moving_position,
)
from bridge_sim.solver import modal_analysis


def main():
    bridge = Bridge(length=60.0, n_elements=40)
    print(bridge.summary())

    load = moving_load(bridge, HEAVY_TRUCK)
    # crossing time + free vibration tail
    cross_time = bridge.length / HEAVY_TRUCK.speed_mps
    duration = cross_time + 6.0
    print(f"Crossing time: {cross_time:.2f} s | total duration: {duration:.2f} s")

    th = newmark_beta(bridge, load, duration_s=duration, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    w_mid = midspan_deflection(bridge, th) * 1000  # mm

    f, A = fft_spectrum(th.t, w_mid)
    modes = modal_analysis(bridge, n_modes=4)

    plot_time_history(
        th.t, w_mid, "figures/02_truck_midspan.png",
        title=f"Déflexion au mi-travée — passage d'un poids-lourd de "
              f"{HEAVY_TRUCK.mass_kg/1000:.0f} t",
    )
    plot_spectrum(
        f, A, "figures/02_truck_spectrum.png", f_max_hz=20.0,
        title="Spectre d'amplitude — passage d'un poids-lourd",
        natural_freqs_hz=modes.natural_frequencies_hz[:4],
    )
    plot_envelope_and_moving_position(
        bridge, th, "figures/02_truck_envelope.png",
        title="Enveloppe min/max des déformations — passage d'un poids-lourd",
    )

    print(f"Max midspan deflection: {np.max(np.abs(w_mid)):.3f} mm")


if __name__ == "__main__":
    main()
