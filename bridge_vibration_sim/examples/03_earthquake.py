"""
Example 3 — Earthquake response.

Applies a synthetic Kanai-Tajimi ground motion (PGA = 0.30 g, dominant
frequency 2.5 Hz, 30 s duration) to the simply-supported steel bridge and
reports mid-span displacement, acceleration history and spectrum.
"""
import numpy as np

from bridge_sim import Bridge, Earthquake, newmark_beta
from bridge_sim.solver import modal_analysis
from bridge_sim.post import (
    plot_time_history, plot_spectrum, midspan_deflection, fft_spectrum,
    plot_3d_waterfall, PALETTE,
)


def main():
    bridge = Bridge(length=60.0, n_elements=40)
    eq = Earthquake(bridge, duration_s=30.0, dt=0.01, pga_g=0.30,
                    dominant_frequency_hz=2.5, damping_ground=0.6)

    th = newmark_beta(bridge, eq, duration_s=30.0, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    w_mid = midspan_deflection(bridge, th) * 1000      # mm
    a_mid = th.a[2 * bridge.midspan_node(), :]          # m/s^2

    f, A = fft_spectrum(th.t, w_mid)
    modes = modal_analysis(bridge, n_modes=4)

    # Plot ground motion + response together
    from bridge_sim.post import use_report_style
    import matplotlib.pyplot as plt
    use_report_style()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True)
    axes[0].plot(eq.t_grid, eq.a_g / 9.81, color=PALETTE["red"])
    axes[0].set_ylabel("Accélération sol [g]")
    axes[0].set_title("Excitation sismique — Kanai-Tajimi (PGA = 0.30 g)")
    axes[1].plot(th.t, w_mid, color=PALETTE["navy"])
    axes[1].set_xlabel("Temps t [s]")
    axes[1].set_ylabel("Déplacement mi-travée [mm]")
    axes[1].set_title("Réponse du pont au séisme")
    fig.tight_layout()
    fig.savefig("figures/03_earthquake_response.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot_spectrum(
        f, A, "figures/03_earthquake_spectrum.png", f_max_hz=15.0,
        title="Spectre de la réponse sismique",
        natural_freqs_hz=modes.natural_frequencies_hz[:4],
    )

    plot_3d_waterfall(
        bridge, th, "figures/03_earthquake_waterfall.png",
        n_snapshots=40,
        title="Réponse spatio-temporelle du pont sous séisme",
    )

    print(f"Max midspan deflection: {np.max(np.abs(w_mid)):.2f} mm")
    print(f"Max midspan acceleration: {np.max(np.abs(a_mid)):.2f} m/s²")


if __name__ == "__main__":
    main()
