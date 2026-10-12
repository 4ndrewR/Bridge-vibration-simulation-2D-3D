"""
Damage scenarios for structural health monitoring studies.

Damage is modelled as a stiffness reduction in one or more beam elements.
The damaged bridge is built by rebuilding K with reduced E*I for each
damaged element and reassembling.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import Bridge, Material, Section


@dataclass
class DamageScenario:
    element_index: int      # 0-based element id
    stiffness_loss: float   # fraction (0.0 ... 0.99)
    label: str = ""


def apply_damage(
    bridge: Bridge,
    scenarios: Sequence[DamageScenario],
) -> Bridge:
    """
    Return a NEW Bridge whose stiffness matrix has reduced E*I in the damaged
    elements. The mass matrix is unchanged (damage usually does not remove
    significant mass).
    """
    new_bridge = deepcopy(bridge)
    new_bridge.K = np.zeros_like(bridge.K)
    new_bridge.M = np.zeros_like(bridge.M)

    E = bridge.material.E
    rho = bridge.material.rho
    A = bridge.section.A
    I = bridge.section.I
    ne = bridge.n_elements
    L_el = bridge.length / ne

    loss_map = {s.element_index: s.stiffness_loss for s in scenarios}

    for e in range(ne):
        loss = loss_map.get(e, 0.0)
        c_k = (1.0 - loss) * E * I / L_el**3
        k_el = c_k * np.array([
            [12,      6 * L_el,    -12,     6 * L_el],
            [6 * L_el, 4 * L_el**2, -6 * L_el, 2 * L_el**2],
            [-12,    -6 * L_el,    12,     -6 * L_el],
            [6 * L_el, 2 * L_el**2, -6 * L_el, 4 * L_el**2],
        ])
        c_m = rho * A * L_el / 420.0
        m_el = c_m * np.array([
            [156,      22 * L_el,    54,      -13 * L_el],
            [22 * L_el, 4 * L_el**2,  13 * L_el, -3 * L_el**2],
            [54,      13 * L_el,    156,     -22 * L_el],
            [-13 * L_el, -3 * L_el**2, -22 * L_el, 4 * L_el**2],
        ])
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        for i in range(4):
            for j in range(4):
                new_bridge.K[dofs[i], dofs[j]] += k_el[i, j]
                new_bridge.M[dofs[i], dofs[j]] += m_el[i, j]

    return new_bridge


def damage_sweep(
    bridge: Bridge,
    element_index: int,
    losses: Sequence[float],
    n_modes: int = 4,
):
    """
    Return frequencies (Hz) as a function of stiffness loss for the given
    element. The result is a (len(losses), n_modes) array.
    """
    from .solver import modal_analysis

    freqs = np.zeros((len(losses), n_modes))
    for k, loss in enumerate(losses):
        damaged = apply_damage(bridge, [DamageScenario(element_index, loss)])
        m = modal_analysis(damaged, n_modes=n_modes)
        freqs[k, :] = m.natural_frequencies_hz[:n_modes]
    return np.asarray(losses), freqs
