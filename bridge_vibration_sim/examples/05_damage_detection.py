"""
Example 5 — Damage detection by frequency shift.

Local damage (loss of stiffness in one element) is progressively introduced
in a mid-span element. The first four natural frequencies are tracked and
plotted as a function of the local stiffness loss.
"""
import numpy as np

from bridge_sim import Bridge
from bridge_sim.damage import damage_sweep
from bridge_sim.post import plot_damage_sweep


def main():
    bridge = Bridge(length=60.0, n_elements=40)
    mid_el = bridge.n_elements // 2

    losses = np.linspace(0.0, 0.6, 13)         # up to 60% local stiffness loss
    losses, freqs = damage_sweep(bridge, mid_el, losses, n_modes=4)

    plot_damage_sweep(losses, freqs, "figures/05_damage_sweep.png")
    print("Frequencies (Hz) for increasing damage at mid-span element:")
    for k, l in enumerate(losses):
        print(f"  loss = {l*100:4.1f}%  ->  ",
              ", ".join(f"f{j+1}={freqs[k,j]:.3f}" for j in range(4)))


if __name__ == "__main__":
    main()
