"""
Example 1 — Modal analysis of a simply-supported steel box-girder bridge.

Computes natural frequencies and plots the first three flexural mode shapes.
"""
from bridge_sim import Bridge, modal_analysis
from bridge_sim.post import plot_mode_shapes


def main():
    bridge = Bridge(length=60.0, n_elements=40)
    modes = modal_analysis(bridge, n_modes=6)

    print(bridge.summary())
    print("\nFréquences propres (Hz) :")
    for k, f in enumerate(modes.natural_frequencies_hz):
        print(f"  Mode {k+1:>2d} : {f:7.3f} Hz")

    plot_mode_shapes(bridge, modes, "figures/01_mode_shapes.png", n_modes=3)
    print("\n[OK] figures/01_mode_shapes.png")


if __name__ == "__main__":
    main()
