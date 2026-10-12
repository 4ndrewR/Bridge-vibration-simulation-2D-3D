"""
Rayleigh proportional damping: C = alpha*M + beta*K.

alpha and beta are determined from two target damping ratios zeta_i, zeta_j at
two angular frequencies omega_i, omega_j (typically the first two modes).
"""

from __future__ import annotations

import numpy as np


def rayleigh_coefficients(
    omega_i: float, omega_j: float,
    zeta_i: float, zeta_j: float,
) -> tuple[float, float]:
    """
    Compute Rayleigh damping coefficients (alpha, beta) so that the modal
    damping ratio equals zeta_i at omega_i and zeta_j at omega_j.
    """
    if abs(omega_i - omega_j) < 1e-12:
        raise ValueError("omega_i and omega_j must differ")
    A = 0.5 * np.array([[1.0 / omega_i, omega_i],
                        [1.0 / omega_j, omega_j]])
    alpha, beta = np.linalg.solve(A, np.array([zeta_i, zeta_j]))
    return float(alpha), float(beta)


def rayleigh_matrix(M: np.ndarray, K: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return alpha * M + beta * K


def modal_damping_ratio(omega: float, alpha: float, beta: float) -> float:
    """Equivalent modal damping ratio at frequency omega for Rayleigh damping."""
    return 0.5 * (alpha / omega + beta * omega)
