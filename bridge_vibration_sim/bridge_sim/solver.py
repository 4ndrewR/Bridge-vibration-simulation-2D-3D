"""
Modal analysis and Newmark-beta time integration.

The Newmark scheme used here is the constant-average-acceleration variant
(gamma=1/2, beta=1/4), which is unconditionally stable for linear systems
and second-order accurate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import eigh

from .model import Bridge
from .damping import rayleigh_coefficients, rayleigh_matrix, modal_damping_ratio


# -----------------------------------------------------------------------------
# Modal analysis
# -----------------------------------------------------------------------------
@dataclass
class ModalResult:
    natural_frequencies_hz: np.ndarray   # (n_modes,)
    angular_frequencies: np.ndarray      # (n_modes,)
    mode_shapes: np.ndarray              # (n_dof_full, n_modes), normalised to mass


def modal_analysis(bridge: Bridge, n_modes: int = 10) -> ModalResult:
    """
    Solve the generalised eigenproblem (K - omega^2 * M) phi = 0 on the
    reduced (constrained) system, then expand mode shapes back to full DOFs.

    Mode shapes are mass-normalised so that phi.T @ M @ phi = I.
    """
    Kr, Mr = bridge.reduce(bridge.K, bridge.M)
    eigvals, eigvecs = eigh(Kr, Mr)

    # discard any non-physical negatives from numerical noise
    eigvals = np.clip(eigvals, 0.0, None)
    omega = np.sqrt(eigvals)
    n_modes = min(n_modes, len(omega))
    omega = omega[:n_modes]
    eigvecs = eigvecs[:, :n_modes]

    # expand to full DOFs (zeros on fixed DOFs)
    phi_full = np.zeros((bridge.n_dof, n_modes))
    phi_full[bridge.free_dofs, :] = eigvecs

    return ModalResult(
        natural_frequencies_hz=omega / (2 * np.pi),
        angular_frequencies=omega,
        mode_shapes=phi_full,
    )


# -----------------------------------------------------------------------------
# Newmark-beta direct integration on the FULL system
# -----------------------------------------------------------------------------
@dataclass
class TimeHistoryResult:
    t: np.ndarray                 # (n_t,)
    u: np.ndarray                 # (n_dof, n_t) displacements
    v: np.ndarray                 # (n_dof, n_t) velocities
    a: np.ndarray                 # (n_dof, n_t) accelerations


def newmark_beta(
    bridge: Bridge,
    load_fn: Callable[[float], np.ndarray],
    duration_s: float,
    dt: float = 0.005,
    zeta_1: float = 0.02,
    zeta_2: float = 0.02,
    u0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    progress: bool = False,
) -> TimeHistoryResult:
    """
    Constant-average-acceleration Newmark integration on the reduced system.

    Damping is Rayleigh, calibrated to give zeta_1 at mode 1 and zeta_2 at
    mode 2 (typical bridge structural damping ~ 1-3%).
    """
    # 1) modal analysis for damping calibration
    modes = modal_analysis(bridge, n_modes=4)
    o1, o2 = modes.angular_frequencies[0], modes.angular_frequencies[1]
    alpha, beta_d = rayleigh_coefficients(o1, o2, zeta_1, zeta_2)
    C_full = rayleigh_matrix(bridge.M, bridge.K, alpha, beta_d)

    # 2) reduce M, K, C
    Mr, Kr = bridge.reduce(bridge.M, bridge.K)
    Cr = C_full[np.ix_(bridge.free_dofs, bridge.free_dofs)]

    # 3) integration parameters
    gamma = 0.5
    beta = 0.25
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    a6 = dt * (1.0 - gamma)
    a7 = dt * gamma

    K_eff = Kr + a0 * Mr + a1 * Cr
    # Factorise once (LU via solve every step would be wasteful)
    from scipy.linalg import lu_factor, lu_solve
    lu = lu_factor(K_eff)

    n_t = int(np.round(duration_s / dt)) + 1
    t = np.linspace(0.0, duration_s, n_t)
    n_dof_red = len(bridge.free_dofs)

    u = np.zeros((n_dof_red, n_t))
    v = np.zeros((n_dof_red, n_t))
    a_ = np.zeros((n_dof_red, n_t))

    if u0 is not None:
        u[:, 0] = u0[bridge.free_dofs]
    if v0 is not None:
        v[:, 0] = v0[bridge.free_dofs]

    # initial acceleration from equation of motion
    F0 = load_fn(0.0)[bridge.free_dofs]
    a_[:, 0] = np.linalg.solve(Mr, F0 - Cr @ v[:, 0] - Kr @ u[:, 0])

    for i in range(n_t - 1):
        t_np1 = t[i + 1]
        F_np1 = load_fn(t_np1)[bridge.free_dofs]
        rhs = (
            F_np1
            + Mr @ (a0 * u[:, i] + a2 * v[:, i] + a3 * a_[:, i])
            + Cr @ (a1 * u[:, i] + a4 * v[:, i] + a5 * a_[:, i])
        )
        u[:, i + 1] = lu_solve(lu, rhs)
        a_[:, i + 1] = a0 * (u[:, i + 1] - u[:, i]) - a2 * v[:, i] - a3 * a_[:, i]
        v[:, i + 1] = v[:, i] + a6 * a_[:, i] + a7 * a_[:, i + 1]

        if progress and (i % max(1, n_t // 20) == 0):
            print(f"  Newmark step {i}/{n_t-1}  ({100*i/(n_t-1):.0f}%)")

    # expand back to full DOFs
    n_dof = bridge.n_dof
    U = np.zeros((n_dof, n_t))
    V = np.zeros((n_dof, n_t))
    A = np.zeros((n_dof, n_t))
    U[bridge.free_dofs, :] = u
    V[bridge.free_dofs, :] = v
    A[bridge.free_dofs, :] = a_

    return TimeHistoryResult(t=t, u=U, v=V, a=A)


# -----------------------------------------------------------------------------
# Frequency response function (FRF)
# -----------------------------------------------------------------------------
def frequency_response(
    bridge: Bridge,
    excitation_node_dof: int,
    observation_node_dof: int,
    freq_hz: np.ndarray,
    zeta_1: float = 0.02, zeta_2: float = 0.02,
    n_modes: int = 10,
) -> np.ndarray:
    """
    Modal-superposition FRF (displacement / force) from a unit harmonic load
    applied at `excitation_node_dof` and observed at `observation_node_dof`.

    Returns a complex array of the same shape as `freq_hz`.
    """
    modes = modal_analysis(bridge, n_modes=n_modes)
    phi = modes.mode_shapes
    omegas = modes.angular_frequencies
    o1, o2 = omegas[0], omegas[1]
    alpha, beta_d = rayleigh_coefficients(o1, o2, zeta_1, zeta_2)
    zetas = np.array([modal_damping_ratio(o, alpha, beta_d) for o in omegas])

    H = np.zeros(len(freq_hz), dtype=complex)
    omega = 2 * np.pi * freq_hz
    for n in range(len(omegas)):
        phi_e = phi[excitation_node_dof, n]
        phi_o = phi[observation_node_dof, n]
        wn = omegas[n]
        zn = zetas[n]
        denom = (wn**2 - omega**2) + 2j * zn * wn * omega
        H += (phi_o * phi_e) / denom
    return H
