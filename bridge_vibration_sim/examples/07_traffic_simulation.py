"""
Example 7 — Mixed traffic stream.

A realistic stream of mixed vehicles (cars, vans, a bus, two trucks)
crosses the bridge with staggered entry times. The mid-span deflection
shows the superposition of contributions and the peak demand on the deck.
"""
import numpy as np

from bridge_sim import (
    Bridge, Vehicle, CAR, VAN, BUS, TRUCK, HEAVY_TRUCK,
    traffic_stream, newmark_beta,
)
from bridge_sim.post import (
    plot_time_history, midspan_deflection,
    plot_envelope_and_moving_position,
)


def main():
    bridge = Bridge(length=60.0, n_elements=40)

    # Replicate vehicles with offset entry times -> mixed traffic
    fleet = [
        Vehicle(CAR.mass_kg,         CAR.speed_mps,         entry_time_s=0.0,  label="car 1"),
        Vehicle(VAN.mass_kg,         VAN.speed_mps,         entry_time_s=0.6,  label="van"),
        Vehicle(BUS.mass_kg,         BUS.speed_mps,         entry_time_s=1.4,  label="bus"),
        Vehicle(TRUCK.mass_kg,       TRUCK.speed_mps,       entry_time_s=2.6,  label="truck"),
        Vehicle(HEAVY_TRUCK.mass_kg, HEAVY_TRUCK.speed_mps, entry_time_s=4.0,  label="44-t lorry"),
        Vehicle(CAR.mass_kg,         CAR.speed_mps,         entry_time_s=4.6,  label="car 2"),
    ]
    load = traffic_stream(bridge, fleet)
    duration = 12.0
    th = newmark_beta(bridge, load, duration_s=duration, dt=0.005,
                      zeta_1=0.02, zeta_2=0.02)
    w_mid = midspan_deflection(bridge, th) * 1000

    plot_time_history(
        th.t, w_mid, "figures/07_traffic_midspan.png",
        title="Déplacement mi-travée — flux de trafic réaliste (6 véhicules)",
    )
    plot_envelope_and_moving_position(
        bridge, th, "figures/07_traffic_envelope.png",
        title="Enveloppe des déformations — flux mixte voitures + poids-lourds",
    )
    print(f"Max midspan deflection: {np.max(np.abs(w_mid)):.2f} mm")
    print(f"Static-equivalent: ({sum(v.weight_N for v in fleet)/1000:.0f} kN total fleet weight)")


if __name__ == "__main__":
    main()
