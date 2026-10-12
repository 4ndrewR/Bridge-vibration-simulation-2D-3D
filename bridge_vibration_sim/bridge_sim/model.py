"""
Finite-element model of a beam-type bridge.

The bridge is discretised with 2-node Euler-Bernoulli beam elements.
Each node carries two degrees of freedom: vertical displacement w
and rotation about the transverse axis theta. Global mass and
stiffness matrices are assembled from the consistent element matrices.

Supports:
    - simply supported (default): pin at left, roller at right
    - cantilever: clamped at left
    - multi-span continuous: roller supports at interior nodes
    - custom boundary conditions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Material / section presets
# -----------------------------------------------------------------------------
@dataclass
class Material:
    """Linear elastic isotropic material."""
    name: str
    E: float           # Young's modulus [Pa]
    rho: float         # density [kg/m^3]
    nu: float = 0.3    # Poisson's ratio (kept for reference)


STEEL = Material("Structural steel S355", E=2.10e11, rho=7850.0)
CONCRETE_C30 = Material("Reinforced concrete C30/37", E=3.30e10, rho=2500.0)
PRESTRESSED_C40 = Material("Prestressed concrete C40/50", E=3.50e10, rho=2500.0)


@dataclass
class Section:
    """Cross-section properties of the bridge deck (as an equivalent beam)."""
    name: str
    A: float           # area [m^2]
    I: float           # second moment of area about transverse axis [m^4]

    @classmethod
    def rectangular(cls, b: float, h: float, name: str | None = None) -> "Section":
        return cls(name=name or f"Rect {b}x{h} m", A=b * h, I=b * h**3 / 12.0)

    @classmethod
    def box_girder(cls, b: float, h: float, t: float, name: str | None = None) -> "Section":
        """Thin-walled rectangular box girder (single cell)."""
        A_out = b * h
        A_in = max((b - 2 * t) * (h - 2 * t), 0.0)
        I_out = b * h**3 / 12.0
        I_in = (b - 2 * t) * (h - 2 * t) ** 3 / 12.0 if A_in > 0 else 0.0
        return cls(
            name=name or f"Box {b}x{h} t={t} m",
            A=A_out - A_in,
            I=I_out - I_in,
        )


# -----------------------------------------------------------------------------
# Bridge geometry / FEM assembly
# -----------------------------------------------------------------------------
@dataclass
class Bridge:
    """
    Beam-bridge FE model.

    Attributes
    ----------
    length : total span length [m]
    n_elements : number of beam elements along the span
    material, section : preset Material / Section instances
    support_type : 'simply_supported', 'cantilever', or 'multi_span'
    n_spans : when support_type == 'multi_span', number of equal spans
    """
    length: float = 60.0
    n_elements: int = 40
    material: Material = field(default_factory=lambda: STEEL)
    section: Section = field(default_factory=lambda: Section.box_girder(8.0, 2.5, 0.04))
    support_type: str = "simply_supported"
    n_spans: int = 1

    # populated by build()
    n_nodes: int = field(init=False, default=0)
    n_dof: int = field(init=False, default=0)
    node_x: np.ndarray = field(init=False, default=None)
    K: np.ndarray = field(init=False, default=None)
    M: np.ndarray = field(init=False, default=None)
    free_dofs: np.ndarray = field(init=False, default=None)
    fixed_dofs: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.build()

    # ---- assembly ---------------------------------------------------------
    def build(self) -> None:
        ne = self.n_elements
        nn = ne + 1
        self.n_nodes = nn
        self.n_dof = 2 * nn
        self.node_x = np.linspace(0.0, self.length, nn)

        E = self.material.E
        rho = self.material.rho
        A = self.section.A
        I = self.section.I
        L_el = self.length / ne

        # Element stiffness (Euler-Bernoulli)
        c_k = E * I / L_el**3
        k_el = c_k * np.array([
            [12,      6 * L_el,    -12,     6 * L_el],
            [6 * L_el, 4 * L_el**2, -6 * L_el, 2 * L_el**2],
            [-12,    -6 * L_el,    12,     -6 * L_el],
            [6 * L_el, 2 * L_el**2, -6 * L_el, 4 * L_el**2],
        ])

        # Consistent element mass matrix
        c_m = rho * A * L_el / 420.0
        m_el = c_m * np.array([
            [156,      22 * L_el,    54,      -13 * L_el],
            [22 * L_el, 4 * L_el**2,  13 * L_el, -3 * L_el**2],
            [54,      13 * L_el,    156,     -22 * L_el],
            [-13 * L_el, -3 * L_el**2, -22 * L_el, 4 * L_el**2],
        ])

        K = np.zeros((self.n_dof, self.n_dof))
        M = np.zeros((self.n_dof, self.n_dof))

        for e in range(ne):
            dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
            for i in range(4):
                for j in range(4):
                    K[dofs[i], dofs[j]] += k_el[i, j]
                    M[dofs[i], dofs[j]] += m_el[i, j]

        self.K = K
        self.M = M
        self._apply_boundary_conditions()

    def _apply_boundary_conditions(self) -> None:
        nn = self.n_nodes
        fixed: list[int] = []

        if self.support_type == "simply_supported":
            # vertical fixed at first and last nodes (rotations free)
            fixed = [0, 2 * (nn - 1)]

        elif self.support_type == "cantilever":
            # vertical + rotation fixed at left
            fixed = [0, 1]

        elif self.support_type == "multi_span":
            # vertical pin at every span boundary node (n_spans + 1 supports)
            if self.n_spans < 2:
                raise ValueError("multi_span requires n_spans >= 2")
            if (self.n_elements % self.n_spans) != 0:
                raise ValueError("n_elements must be a multiple of n_spans for multi_span")
            step = self.n_elements // self.n_spans
            for k in range(self.n_spans + 1):
                fixed.append(2 * k * step)

        elif self.support_type == "free_free":
            fixed = []

        else:
            raise ValueError(f"Unknown support_type '{self.support_type}'")

        all_dofs = np.arange(self.n_dof)
        self.fixed_dofs = np.array(sorted(set(fixed)), dtype=int)
        self.free_dofs = np.array([d for d in all_dofs if d not in self.fixed_dofs], dtype=int)

    # ---- helpers ----------------------------------------------------------
    def reduce(self, *mats: np.ndarray) -> tuple[np.ndarray, ...]:
        """Reduce given full matrices to free DOFs."""
        fd = self.free_dofs
        return tuple(m[np.ix_(fd, fd)] if m.ndim == 2 else m[fd] for m in mats)

    def expand(self, u_free: np.ndarray) -> np.ndarray:
        """Expand a free-DOF vector to the full DOF vector (zeros at fixed DOFs)."""
        full = np.zeros(self.n_dof)
        full[self.free_dofs] = u_free
        return full

    def nodal_vertical_dofs(self) -> np.ndarray:
        return np.arange(0, self.n_dof, 2)

    def midspan_node(self) -> int:
        """Index of the node closest to mid-span."""
        return int(np.argmin(np.abs(self.node_x - self.length / 2.0)))

    def shape_functions(self, xi: float, L_el: float) -> np.ndarray:
        """Hermite shape functions for a beam element. xi in [0, 1]."""
        N1 = 1 - 3 * xi**2 + 2 * xi**3
        N2 = L_el * (xi - 2 * xi**2 + xi**3)
        N3 = 3 * xi**2 - 2 * xi**3
        N4 = L_el * (-xi**2 + xi**3)
        return np.array([N1, N2, N3, N4])

    def consistent_load_vector(self, x: float, P: float) -> np.ndarray:
        """
        Equivalent nodal load vector for a vertical point load P applied at
        position x along the bridge. Uses Hermite shape functions so the
        result is consistent with the beam element formulation.
        """
        L_el = self.length / self.n_elements
        e = min(int(x // L_el), self.n_elements - 1)
        xi = (x - e * L_el) / L_el
        xi = float(np.clip(xi, 0.0, 1.0))
        N = self.shape_functions(xi, L_el)
        F = np.zeros(self.n_dof)
        dofs = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        for i, d in enumerate(dofs):
            F[d] += P * N[i]
        return F

    def summary(self) -> str:
        return (
            f"Bridge: L={self.length} m, {self.n_elements} elements, "
            f"{self.material.name}, {self.section.name}, "
            f"support={self.support_type}"
            + (f" ({self.n_spans} spans)" if self.support_type == "multi_span" else "")
        )
