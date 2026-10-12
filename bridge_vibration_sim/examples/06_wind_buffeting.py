"""
Example 6 — Wind buffeting.

A simplified turbulent wind load (mean 25 m/s, 12 % turbulence) excites the
bridge over 30 s. The mid-span vertical displacement and its spectrum show
the broad-band character and the peaks coinciding with the first flexural
modes.
"""
import numpy as np

from bridge_sim import Bridge, WindBuffeting, newmark_beta
from bridge_sim.solver import modal_analysis
from bridge_sim.post import (
    plot_time_history, plot_spectrum, midspan_deflection, fft_spectrum,
)


def main():
    bridge = Bridge(length=60.0, n_elements=40)
    wind = WindBuffeting(bridge, mean_speed_mps=25.0, turbulence_intensity=0.12,
                         duration_s=30.0, dt_sample=0.01)
    th = newmark_beta(bridge, wind, duration_s=30.0, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    w_mid = midspan_deflection(bridge, th) * 1000
    f, A = fft_spectrum(th.t, w_mid)
    modes = modal_analysis(bridge, n_modes=4)

    plot_time_history(
        th.t, w_mid, "figures/06_wind_midspan.png",
        title="Déplacement mi-travée — vent turbulent (V = 25 m/s, I = 12 %)",
    )
    plot_spectrum(
        f, A, "figures/06_wind_spectrum.png", f_max_hz=15.0,
        title="Spectre de la réponse sous vent turbulent",
        natural_freqs_hz=modes.natural_frequencies_hz[:4],
    )
    print(f"Max midspan deflection under wind: {np.max(np.abs(w_mid)):.2f} mm")


if __name__ == "__main__":
    main()
