r"""
lorenz_3d_plot.py — 3D visualisation of the Lorenz attractor
=============================================================

Produces a publication-quality 3D rendering of the Lorenz attractor with
a colour gradient that encodes the temporal evolution.

Mathematical background
-----------------------
The Lorenz attractor is the long-term trajectory of the Lorenz system:

.. math::

    \dot{x} &= \sigma (y - x) \\
    \dot{y} &= x (\rho - z) - y \\
    \dot{z} &= x y - \beta z

When :math:`\sigma = 10, \rho = 28, \beta = 8/3`, trajectories spiral
around two unstable fixed points (the "butterfly wings"), switching
lobes aperiodically.  The colour transitions from blue (early time)
through green and yellow to red (late time), revealing the temporal
flow.

References
----------
- Lorenz, E. N. (1963). "Deterministic Nonperiodic Flow".
  *J. Atmos. Sci.*, 20, 130-141.
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 9.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lorenz.lorenz_solver import solve_lorenz, lorenz_fixed_points
from common_styles import (
    apply_dark_theme,
    add_equation_box,
    add_info_box,
)

from typing import Tuple, Optional


# ============================================================================
# Plotting
# ============================================================================

def plot_lorenz(
    initial_conditions: Tuple[float, float, float] = (0.0, 1.0, 1.05),
    t_end: float = 40.0,
    num_points: int = 10000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    elevation: float = 25.0,
    azimuth: float = 45.0,
    colormap: str = "plasma",
    line_width: float = 0.8,
    show_fixed_points: bool = True,
    show_equation: bool = True,
    show_colorbar: bool = True,
    transparent_panes: bool = True,
    figsize: Tuple[float, float] = (12, 9),
    dpi: int = 150,
) -> plt.Figure:
    r"""
    Plot a 3D rendering of the Lorenz attractor.

    Parameters
    ----------
    initial_conditions : tuple of float, default=(0, 1, 1.05)
        Initial state :math:`(x_0, y_0, z_0)`.
    t_end : float, default=40.0
        Total integration time.
    num_points : int, default=10000
        Number of trajectory samples.
    sigma : float, default=10.0
        Prandtl number :math:`\sigma`.
    rho : float, default=28.0
        Rayleigh number :math:`\rho`.
    beta : float, default=8/3
        Aspect ratio :math:`\beta`.
    elevation : float, default=25.0
        Camera elevation angle (degrees).
    azimuth : float, default=45.0
        Camera azimuth angle (degrees).
    colormap : str, default="plasma"
        Matplotlib colormap for the time-encoded trajectory.
    line_width : float, default=0.8
        Width of the trajectory segments.
    show_fixed_points : bool, default=True
        If True, mark the C+ and C− fixed points (the "eyes").
    show_equation : bool, default=True
        If True, display the Lorenz equations in the top-left corner.
    show_colorbar : bool, default=True
        If True, show a colour bar mapping colour → time.
    transparent_panes : bool, default=True
        If True, remove the 3D pane fill for a cleaner look.
    figsize : tuple of float, default=(12, 9)
        Figure dimensions.
    dpi : int, default=150
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> fig = plot_lorenz(elevation=30, azimuth=60, colormap="inferno")
    >>> st.pyplot(fig)
    """
    apply_dark_theme(dpi=dpi)

    # --- solve the system ---
    t, x, y, z = solve_lorenz(
        initial_conditions=initial_conditions,
        t_end=t_end,
        num_points=num_points,
        sigma=sigma, rho=rho, beta=beta,
    )

    # --- build colour-mapped Line3DCollection ---
    points = np.column_stack([x, y, z])
    segments = np.concatenate(
        [points[:-1, np.newaxis, :], points[1:, np.newaxis, :]],
        axis=1,
    )

    norm = Normalize(t.min(), t.max())
    lc = Line3DCollection(
        segments,
        cmap=colormap,
        norm=norm,
        linewidth=line_width,
        antialiased=True,
    )
    lc.set_array(t)

    # --- figure & axes ---
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d", facecolor="#0d1117")
    ax.add_collection3d(lc)

    # --- limits ---
    margin = 2.0
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.set_zlim(z.min() - margin, z.max() + margin)

    # --- labels ---
    ax.set_xlabel(r"$x$", fontsize=13, labelpad=8)
    ax.set_ylabel(r"$y$", fontsize=13, labelpad=8)
    ax.set_zlabel(r"$z$", fontsize=13, labelpad=8)
    ax.set_title(
        "Lorenz Attractor",
        fontsize=17, fontweight="bold", pad=20,
    )

    # --- viewing angle ---
    ax.view_init(elev=elevation, azim=azimuth)

    # --- clean up 3D panes ---
    if transparent_panes:
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#21262d")
            pane.set_linewidth(0.3)
        ax.grid(False)
    else:
        ax.xaxis.pane.set_facecolor("#0d1117")
        ax.yaxis.pane.set_facecolor("#0d1117")
        ax.zaxis.pane.set_facecolor("#0d1117")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_alpha(0.1)
        ax.grid(True, alpha=0.15)

    # --- colour bar ---
    if show_colorbar:
        cbar = fig.colorbar(lc, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label("Time  $t$", fontsize=11)
        cbar.ax.yaxis.label.set_color("#c9d1d9")

    # --- fixed points (C+, C−) ---
    if show_fixed_points:
        fps = lorenz_fixed_points(sigma=sigma, rho=rho, beta=beta)
        for i, fp in enumerate(fps):
            # Skip the origin — it's not visually interesting here
            if i == 0:
                continue
            ax.scatter(
                *fp, color="#f0883e", s=60,
                marker="o", edgecolors="white",
                linewidths=0.5, zorder=5,
            )
            ax.text(
                *fp, rf"  $C_{{{'+' if i == 1 else '-'}}}$",
                color="white", fontsize=10,
                fontweight="bold",
            )

    # --- equation box ---
    if show_equation:
        add_equation_box(
            ax,
            r"$\dot{x} = \sigma (y - x)$"
            r"$\quad \dot{y} = x (\rho - z) - y$"
            r"$\quad \dot{z} = x y - \beta z$",
            x=0.02, y=0.94,
            fontsize=10,
        )

    # --- info box ---
    add_info_box(ax, [
        rf"$\sigma={sigma:.1f},\; \rho={rho:.1f},\; \beta={beta:.3f}$",
        rf"$t \in [{t[0]:.1f}, {t[-1]:.1f}]$",
        rf"Points: {num_points:,}",
        rf"$(x_0, y_0, z_0) = ({initial_conditions[0]:.2f}, "
        rf"{initial_conditions[1]:.2f}, {initial_conditions[2]:.2f})$",
    ])

    plt.tight_layout()
    return fig


# ============================================================================
# CLI demo
# ============================================================================

if __name__ == "__main__":
    import locale
    locale.setlocale(locale.LC_NUMERIC, "C")

    fig = plot_lorenz(
        elevation=25, azimuth=45,
        show_fixed_points=True,
        show_equation=True,
    )
    fig.savefig("lorenz_attractor.png", dpi=300)
    print("Saved lorenz_attractor.png")
