r"""
sensitivity_plot.py — Butterfly Effect visualisation
=====================================================

Demonstrates the extreme sensitivity to initial conditions that is the
hallmark of deterministic chaos — the "butterfly effect".

Mathematical background
-----------------------
In a chaotic system, two trajectories starting from arbitrarily close
initial conditions diverge exponentially in time.  For the Lorenz system:

.. math::

    \delta(t) \sim \delta_0 \, e^{\lambda t}

where :math:`\delta(t)` is the Euclidean distance between trajectories
and :math:`\lambda` is the largest Lyapunov exponent.

This plot takes two trajectories whose initial z-coordinates differ by
only :math:`5 \times 10^{-5}` (a few parts in 10⁵) and shows their
Euclidean distance on a logarithmic scale.  An exponential growth phase
followed by saturation at the size of the attractor is clearly visible.

References
----------
- Lorenz, E. N. (1963). "Deterministic Nonperiodic Flow".
  *J. Atmos. Sci.*, 20, 130-141.
- Lorenz, E. N. (1972). "Predictability: Does the flap of a
  butterfly's wings in Brazil set off a tornado in Texas?"
  (AAAS meeting talk).
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 9.
"""

import numpy as np
import matplotlib.pyplot as plt

from lorenz.lorenz_solver import solve_lorenz
from common_styles import (
    apply_dark_theme,
    figure_size,
    add_equation_box,
    add_info_box,
    add_zero_line,
)

from typing import Tuple, Optional


# ============================================================================
# Butterfly-effect computation
# ============================================================================

def compute_divergence(
    ic1: Tuple[float, float, float] = (0.0, 1.0, 1.05),
    perturbation: float = 0.00001,
    axis: int = 2,
    t_end: float = 40.0,
    num_points: int = 10000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Integrate two trajectories with a small initial perturbation and
    compute their Euclidean distance.

    Parameters
    ----------
    ic1 : tuple of float, default=(0, 1, 1.05)
        Reference initial condition :math:`(x_0, y_0, z_0)`.
    perturbation : float, default=1e-5
        Perturbation magnitude applied to one coordinate of *ic1*.
    axis : int, default=2 (z-coordinate)
        Which coordinate to perturb (0→x, 1→y, 2→z).
    t_end : float, default=40.0
        Integration time horizon.
    num_points : int, default=10000
        Number of time samples.
    sigma, rho, beta : float
        Lorenz system parameters.

    Returns
    -------
    t : ndarray, shape (num_points,)
        Time grid.
    x1, y1, z1 : ndarray
        Reference trajectory.
    x2, y2, z2 : ndarray
        Perturbed trajectory.
    """
    ic2 = list(ic1)
    ic2[axis] += perturbation

    t, x1, y1, z1 = solve_lorenz(
        initial_conditions=ic1,
        t_end=t_end,
        num_points=num_points,
        sigma=sigma, rho=rho, beta=beta,
    )
    _, x2, y2, z2 = solve_lorenz(
        initial_conditions=tuple(ic2),
        t_end=t_end,
        num_points=num_points,
        sigma=sigma, rho=rho, beta=beta,
    )
    return t, x1, y1, z1, x2, y2, z2


# ============================================================================
# Plotting
# ============================================================================

def plot_sensitivity(
    ic1: Tuple[float, float, float] = (0.0, 1.0, 1.05),
    perturbation: float = 0.00001,
    axis: int = 2,
    t_end: float = 40.0,
    num_points: int = 10000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    line_color: str = "#58a6ff",
    line_width: float = 2.0,
    show_exponential_fit: bool = True,
    show_saturation_line: bool = True,
    show_equation: bool = True,
    figsize: Tuple[float, float] = (12, 7),
    dpi: int = 150,
) -> plt.Figure:
    r"""
    Plot the Butterfly Effect — divergence of nearby initial conditions.

    Parameters
    ----------
    ic1 : tuple of float, default=(0, 1, 1.05)
        Reference initial condition.
    perturbation : float, default=1e-5
        Initial perturbation magnitude.
    axis : int, default=2
        Coordinate to perturb (0=x, 1=y, 2=z).
    t_end : float, default=40.0
        Integration time.
    num_points : int, default=10000
        Number of time samples.
    sigma, rho, beta : float
        Lorenz system parameters.
    line_color : str, default="#58a6ff"
        Curve colour.
    line_width : float, default=2.0
        Curve line width.
    show_exponential_fit : bool, default=True
        If True, overlay an exponential :math:`e^{\lambda t}` fit on the
        linear-growth phase.
    show_saturation_line : bool, default=True
        If True, draw a horizontal line indicating the approximate
        attractor diameter (distance saturation).
    show_equation : bool, default=True
        If True, display the governing equations.
    figsize : tuple of float, default=(12, 7)
        Figure dimensions.
    dpi : int, default=150
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> fig = plot_sensitivity(perturbation=1e-6, axis=0)
    >>> st.pyplot(fig)
    """
    apply_dark_theme(dpi=dpi)

    # --- compute trajectories ---
    t, x1, y1, z1, x2, y2, z2 = compute_divergence(
        ic1=ic1, perturbation=perturbation, axis=axis,
        t_end=t_end, num_points=num_points,
        sigma=sigma, rho=rho, beta=beta,
    )

    # --- Euclidean distance ---
    distance = np.sqrt(
        (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    )

    # --- figure ---
    fig, ax = plt.subplots(figsize=figsize)

    # Main curve
    ax.plot(t, distance, color=line_color,
            linewidth=line_width, zorder=3)

    # --- exponential fit on the linear-growth phase ---
    if show_exponential_fit:
        # Find the exponential growth phase: between t=2 and t=15
        # where distance is growing roughly linearly on log scale
        mask = (t > 2.0) & (t < 15.0) & (distance > 0)
        if mask.sum() > 5:
            log_dist = np.log(distance[mask])
            coeffs = np.polyfit(t[mask], log_dist, 1)
            lam_est = coeffs[0]
            ax.plot(
                t[mask], np.exp(coeffs[1] + lam_est * t[mask]),
                color="#f0883e", linestyle="--", linewidth=1.2,
                label=rf"$\lambda \approx {lam_est:.2f}$",
                alpha=0.8,
            )
            ax.legend(fontsize=11, loc="upper left")

    # --- saturation line ---
    if show_saturation_line:
        sat_level = distance.max()
        ax.axhline(sat_level, color="#8b949e", linestyle=":",
                   linewidth=0.8, alpha=0.6)
        ax.text(
            t[-1] * 0.02, sat_level * 1.05,
            f"Attractor span ≈ {sat_level:.1f}",
            color="#8b949e", fontsize=9,
        )

    # --- annotations ---
    ax.set_yscale("log")
    ax.set_xlabel("Time  $t$", fontsize=14)
    ax.set_ylabel(r"Euclidean distance   $\delta(t)$" + "\n(log scale)",
                  fontsize=13)
    ax.set_title(
        "Butterfly Effect — Sensitivity to Initial Conditions",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(top=max(distance.max() * 2, 1e1))

    # --- equation ---
    if show_equation:
        add_equation_box(
            ax,
            r"$\delta(t) \sim \delta_0 \, e^{\lambda t}$",
            x=0.02, y=0.94,
            fontsize=13,
        )

    # --- info box ---
    perturbed_coord = ["x", "y", "z"][axis]
    add_info_box(ax, [
        rf"Perturbation: $\Delta {perturbed_coord} = {perturbation}$",
        rf"$t \in [{t[0]:.1f}, {t[-1]:.1f}]$",
        rf"Points: {num_points:,}",
        rf"Final distance: {distance[-1]:.2f}",
        rf"Expansion factor: "
        rf"$\times{distance[-1] / perturbation:.0f}$",
    ])

    plt.tight_layout()
    return fig


# ============================================================================
# CLI demo
# ============================================================================

if __name__ == "__main__":
    import locale
    locale.setlocale(locale.LC_NUMERIC, "C")

    fig = plot_sensitivity(
        perturbation=1e-5,
        show_exponential_fit=True,
    )
    fig.savefig("butterfly_effect.png", dpi=300)
    print("Saved butterfly_effect.png")
