"""
Example 8 — Combined scenario.

The bridge experiences a heavy truck crossing while a moderate earthquake
strikes at t = 4 s. This is a stress test for the integrator: two strongly
time-varying loadings simultaneously.

Also produces an animated GIF of the deflected shape.
"""
from bridge_sim import (
    Bridge, HEAVY_TRUCK, moving_load, Earthquake, combine, newmark_beta,
)
from bridge_sim.post import (
    plot_time_history, midspan_deflection, animate_bridge,
)


def main():
    bridge = Bridge(length=60.0, n_elements=40)
    truck = moving_load(bridge, HEAVY_TRUCK)
    eq = Earthquake(bridge, duration_s=12.0, dt=0.01, pga_g=0.20,
                    dominant_frequency_hz=2.5)
    F = combine(truck, eq)

    th = newmark_beta(bridge, F, duration_s=12.0, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    w_mid = midspan_deflection(bridge, th) * 1000
    plot_time_history(
        th.t, w_mid, "figures/08_combined_midspan.png",
        title="Combiné : poids-lourd + séisme (PGA = 0.20 g)",
    )

    animate_bridge(bridge, th, "figures/08_combined_animation.gif",
                   fps=20, downsample=4,
                   title="Animation — poids-lourd + séisme")


if __name__ == "__main__":
    main()
