"""bridge_sim — finite-element bridge vibration simulation package."""

from .model import Bridge, Material, Section, STEEL, CONCRETE_C30, PRESTRESSED_C40
from .solver import (
    modal_analysis, newmark_beta, frequency_response,
    ModalResult, TimeHistoryResult,
)
from .loads import (
    Vehicle, CAR, VAN, BUS, TRUCK, HEAVY_TRUCK,
    moving_load, traffic_stream,
    HarmonicLoad, PedestrianCrowd, WindBuffeting, Earthquake, ImpulseLoad,
    combine,
)
from .damping import rayleigh_coefficients, modal_damping_ratio
from .damage import DamageScenario, apply_damage, damage_sweep

__all__ = [
    "Bridge", "Material", "Section",
    "STEEL", "CONCRETE_C30", "PRESTRESSED_C40",
    "modal_analysis", "newmark_beta", "frequency_response",
    "ModalResult", "TimeHistoryResult",
    "Vehicle", "CAR", "VAN", "BUS", "TRUCK", "HEAVY_TRUCK",
    "moving_load", "traffic_stream",
    "HarmonicLoad", "PedestrianCrowd", "WindBuffeting", "Earthquake", "ImpulseLoad",
    "combine",
    "rayleigh_coefficients", "modal_damping_ratio",
    "DamageScenario", "apply_damage", "damage_sweep",
]

__version__ = "1.0.0"
