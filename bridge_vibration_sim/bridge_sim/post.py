"""
Post-processing utilities: spectra, deflection extraction, plotting.

Plots use Matplotlib with a clean academic style suitable for a technical
report (white background, sans-serif, single-column friendly).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
import matplotlib as mpl

from .model import Bridge
from .solver import ModalResult, TimeHistoryResult


# -----------------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------------
PALETTE = {
    "navy":   "#1E2761",
    "blue":   "#2A6DB0",
    "sky":    "#5BA3D0",
    "ice":    "#CADCFC",
    "red":    "#C0392B",
    "orange": "#E67E22",
    "green":  "#1E8449",
    "gray":   "#5D6D7E",
    "lightgray": "#D5D8DC",
    "black":  "#212121",
}


def use_report_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#34495E",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E5E7E9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.color": "#34495E",
        "ytick.color": "#34495E",
        "lines.linewidth": 1.8,
    })


# -----------------------------------------------------------------------------
# Spectral utilities
# -----------------------------------------------------------------------------
def fft_spectrum(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (positive freq array [Hz], single-sided amplitude spectrum)."""
    n = len(t)
    dt = t[1] - t[0]
    X = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(n, d=dt)
    amp = 2.0 / n * np.abs(X)
    return f, amp


# -----------------------------------------------------------------------------
# Display helpers
# -----------------------------------------------------------------------------
def midspan_deflection(bridge: Bridge, th: TimeHistoryResult) -> np.ndarray:
    """Vertical displacement (m) of the mid-span node over time."""
    mid = bridge.midspan_node()
    return th.u[2 * mid, :]


def deflection_shape(bridge: Bridge, th: TimeHistoryResult, t_index: int) -> np.ndarray:
    """Vertical displacement of every node at the given time index."""
    return th.u[bridge.nodal_vertical_dofs(), t_index]


# -----------------------------------------------------------------------------
# Schematic of the bridge with mode-shape overlay -- Figure 1 (intro)
# -----------------------------------------------------------------------------
def plot_bridge_schematic_with_modes(
    bridge: Bridge, modes: ModalResult, out_path: str,
    n_modes: int = 4,
) -> None:
    use_report_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    L = bridge.length
    x = bridge.node_x
    vdofs = bridge.nodal_vertical_dofs()

    # deck baseline
    ax.add_patch(Polygon([(0, -0.04), (L, -0.04), (L, 0.04), (0, 0.04)],
                         facecolor="#34495E", edgecolor="#212121", lw=0.8))

    # supports
    _draw_pin(ax, 0.0, -0.04)
    _draw_roller(ax, L, -0.04)

    # overlay first n_modes scaled so they read well
    colors = [PALETTE["red"], PALETTE["orange"], PALETTE["blue"], PALETTE["green"]]
    scale = 0.35 / np.max(np.abs(modes.mode_shapes[vdofs, :n_modes]))
    for k in range(n_modes):
        y = modes.mode_shapes[vdofs, k] * scale
        f = modes.natural_frequencies_hz[k]
        ax.plot(x, y, color=colors[k], lw=2.2,
                label=f"Mode {k+1}: f = {f:.2f} Hz")

    ax.set_xlim(-0.05 * L, 1.05 * L)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect("auto")
    ax.set_xlabel("Position le long du pont x [m]")
    ax.set_ylabel("Déformée modale (normalisée)")
    ax.set_title("Schéma du pont et superposition des premiers modes propres")
    ax.legend(loc="upper right", ncol=2)
    ax.set_yticks([])
    fig.savefig(out_path)
    plt.close(fig)


def _draw_pin(ax, x, y):
    h = 0.10
    ax.add_patch(Polygon([(x, y), (x - h, y - h), (x + h, y - h)],
                         facecolor=PALETTE["gray"], edgecolor="black", lw=0.6))
    ax.plot([x - 0.15, x + 0.15], [y - h, y - h], "k-", lw=0.8)
    for xx in np.linspace(x - 0.12, x + 0.12, 6):
        ax.plot([xx, xx - 0.04], [y - h, y - h - 0.05], "k-", lw=0.6)


def _draw_roller(ax, x, y):
    h = 0.10
    ax.add_patch(plt.Circle((x, y - h / 2), h / 2,
                            facecolor=PALETTE["gray"], edgecolor="black", lw=0.6))
    ax.plot([x - 0.15, x + 0.15], [y - h, y - h], "k-", lw=0.8)
    for xx in np.linspace(x - 0.12, x + 0.12, 6):
        ax.plot([xx, xx - 0.04], [y - h, y - h - 0.05], "k-", lw=0.6)


# -----------------------------------------------------------------------------
# Mode-shapes panel -- Figure 2 (chapter on modal analysis)
# -----------------------------------------------------------------------------
def plot_mode_shapes(
    bridge: Bridge, modes: ModalResult, out_path: str,
    n_modes: int = 3,
) -> None:
    use_report_style()
    fig, axes = plt.subplots(n_modes, 1, figsize=(9.0, 1.6 * n_modes + 0.6),
                             sharex=True)
    if n_modes == 1:
        axes = [axes]

    x = bridge.node_x
    vdofs = bridge.nodal_vertical_dofs()
    colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["sky"],
              PALETTE["red"], PALETTE["orange"]]

    for k, ax in enumerate(axes):
        y = modes.mode_shapes[vdofs, k]
        y = y / np.max(np.abs(y))
        ax.fill_between(x, 0, y, color=colors[k % len(colors)], alpha=0.25)
        ax.plot(x, y, color=colors[k % len(colors)], lw=2.4)
        ax.axhline(0, color="#212121", lw=0.6, ls="--", alpha=0.5)
        ax.set_ylabel(f"Mode {k+1}\n{modes.natural_frequencies_hz[k]:.2f} Hz")
        ax.set_ylim(-1.25, 1.25)
        ax.set_yticks([-1, 0, 1])
        # supports
        _draw_pin(ax, x[0], 0.0)
        _draw_roller(ax, x[-1], 0.0)

    axes[-1].set_xlabel("Position le long du pont x [m]")
    fig.suptitle("Modes propres du pont — analyse modale FEM", y=0.995, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Generic plotters
# -----------------------------------------------------------------------------
def plot_time_history(
    t: np.ndarray, y: np.ndarray, out_path: str,
    title: str, ylabel: str = "Déplacement au mi-travée [mm]",
    color: str = PALETTE["navy"],
    extra_line: tuple[np.ndarray, np.ndarray, str] | None = None,
) -> None:
    use_report_style()
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.plot(t, y, color=color)
    if extra_line is not None:
        t2, y2, label2 = extra_line
        ax.plot(t2, y2, color=PALETTE["red"], lw=1.4, ls="--", label=label2)
        ax.legend()
    ax.set_xlabel("Temps t [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)


def plot_spectrum(
    f_hz: np.ndarray, amp: np.ndarray, out_path: str, title: str,
    f_max_hz: float | None = None,
    natural_freqs_hz: Sequence[float] | None = None,
) -> None:
    use_report_style()
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    if f_max_hz is None:
        f_max_hz = f_hz[-1]
    mask = f_hz <= f_max_hz
    ax.fill_between(f_hz[mask], 0, amp[mask], color=PALETTE["sky"], alpha=0.5)
    ax.plot(f_hz[mask], amp[mask], color=PALETTE["navy"])
    if natural_freqs_hz is not None:
        for fn in natural_freqs_hz:
            if fn <= f_max_hz:
                ax.axvline(fn, color=PALETTE["red"], lw=1.0, ls=":")
                ax.text(fn, ax.get_ylim()[1] * 0.95, f" f={fn:.2f} Hz",
                        color=PALETTE["red"], fontsize=8, rotation=90,
                        va="top", ha="left")
    ax.set_xlabel("Fréquence f [Hz]")
    ax.set_ylabel("Amplitude spectrale")
    ax.set_title(title)
    ax.set_xlim(0, f_max_hz)
    fig.savefig(out_path)
    plt.close(fig)


def plot_frf(
    f_hz: np.ndarray, H: np.ndarray, out_path: str, title: str,
) -> None:
    use_report_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.5), sharex=True)
    ax1.semilogy(f_hz, np.abs(H), color=PALETTE["navy"])
    ax1.set_ylabel("|H(f)| [m/N]")
    ax1.set_title(title)

    ax2.plot(f_hz, np.angle(H, deg=True), color=PALETTE["red"])
    ax2.set_ylabel("Phase [°]")
    ax2.set_xlabel("Fréquence f [Hz]")
    ax2.set_ylim(-200, 200)
    fig.savefig(out_path)
    plt.close(fig)


def plot_damage_sweep(
    losses: np.ndarray, freqs: np.ndarray, out_path: str,
    title: str = "Sensibilité fréquentielle à un endommagement local",
) -> None:
    use_report_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["sky"], PALETTE["orange"]]
    for k in range(freqs.shape[1]):
        ratio = freqs[:, k] / freqs[0, k] * 100
        ax.plot(losses * 100, ratio, marker="o", lw=2.0,
                color=colors[k % len(colors)], label=f"Mode {k+1}")
    ax.set_xlabel("Perte de rigidité locale dans l'élément endommagé [%]")
    ax.set_ylabel("Fréquence normalisée  f / f_intact [%]")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_envelope_and_moving_position(
    bridge: Bridge, th: TimeHistoryResult, out_path: str,
    title: str = "Enveloppe des déformations sous trafic",
) -> None:
    use_report_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    x = bridge.node_x
    vdofs = bridge.nodal_vertical_dofs()
    U = th.u[vdofs, :] * 1000  # mm
    u_min = U.min(axis=1)
    u_max = U.max(axis=1)
    ax.fill_between(x, u_min, u_max, color=PALETTE["ice"], alpha=0.8,
                    label="Enveloppe min/max")
    ax.plot(x, u_max, color=PALETTE["red"], lw=1.5)
    ax.plot(x, u_min, color=PALETTE["navy"], lw=1.5)
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Déplacement vertical [mm]")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_3d_waterfall(
    bridge: Bridge, th: TimeHistoryResult, out_path: str,
    n_snapshots: int = 30,
    title: str = "Réponse spatio-temporelle du pont",
) -> None:
    """Waterfall-style 3D plot of the bridge deflection vs space and time."""
    use_report_style()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(9.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    x = bridge.node_x
    vdofs = bridge.nodal_vertical_dofs()
    n_t = th.u.shape[1]
    indices = np.linspace(0, n_t - 1, n_snapshots).astype(int)
    cmap = plt.cm.viridis
    for k, idx in enumerate(indices):
        y = np.full_like(x, th.t[idx])
        z = th.u[vdofs, idx] * 1000
        ax.plot(x, y, z, color=cmap(k / n_snapshots), lw=1.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Temps [s]")
    ax.set_zlabel("Déplacement [mm]")
    ax.set_title(title)
    ax.view_init(elev=22, azim=-65)
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Animation (saved as GIF)
# -----------------------------------------------------------------------------
def animate_bridge(
    bridge: Bridge, th: TimeHistoryResult, out_path: str,
    fps: int = 25, downsample: int = 4,
    title: str = "Animation – vibrations du pont",
) -> None:
    use_report_style()
    x = bridge.node_x
    vdofs = bridge.nodal_vertical_dofs()
    U = th.u[vdofs, ::downsample] * 1000
    t = th.t[::downsample]
    ymax = np.max(np.abs(U)) * 1.2 + 1e-9

    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.set_xlim(0, bridge.length)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Déplacement [mm]")
    ax.set_title(title)
    line, = ax.plot(x, U[:, 0], color=PALETTE["navy"], lw=2.5)
    txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=10,
                  color=PALETTE["red"])

    _draw_pin(ax, x[0], -ymax * 0.95)
    _draw_roller(ax, x[-1], -ymax * 0.95)

    def update(i):
        line.set_ydata(U[:, i])
        txt.set_text(f"t = {t[i]:.2f} s")
        return line, txt

    anim = animation.FuncAnimation(fig, update, frames=U.shape[1], interval=1000 / fps,
                                   blit=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
