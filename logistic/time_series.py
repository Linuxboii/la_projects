"""
time_series.py — Logistic Map Time Series
==========================================

Plots the iteration-by-iteration evolution of the logistic map for selected
parameter values, revealing the transition from fixed points to chaos.

Mathematical background
-----------------------
Viewing :math:`x_n` vs. *n* (iteration number) shows how the asymptotic
behaviour depends on *r*:

- **r = 2.8** — rapid convergence to a stable fixed point.
- **r = 3.2** — period-2 oscillation.
- **r = 3.5** — period-4 oscillation (period-doubling cascade).
- **r = 3.9** — aperiodic, chaotic orbit.

For each case the initial condition is the same (x₀ = 0.5), highlighting
that only *r* — not the initial state — determines the asymptotic regime.

References
----------
- May, R. M. (1976). "Simple mathematical models with very complicated
  dynamics". *Nature*, 261, 459-467.
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 10.
"""

import numpy as np
import matplotlib.pyplot as plt

from logistic.logistic_map import generate_orbit
from common_styles import (
    apply_dark_theme,
    figure_size,
    add_equation_box,
    add_info_box,
    LOGISTIC_COLORS,
)

from typing import List, Union, Tuple, Optional


# ============================================================================
# Default r values and their descriptions
# ============================================================================

DEFAULT_R_VALUES: List[float] = [2.8, 3.2, 3.5, 3.9]
"""Canonical parameter values covering fixed-point, periodic, and chaotic regimes."""

REGIME_LABELS = {
    2.8: "Stable fixed point",
    3.2: "Period-2 cycle",
    3.5: "Period-4 cycle",
    3.9: "Chaos",
}
"""Short regime descriptions for annotation."""


# ============================================================================
# Plotting
# ============================================================================

def plot_time_series(
    r_values: Optional[List[float]] = None,
    x0: float = 0.5,
    iterations: int = 100,
    transient: int = 0,
    cols: int = 2,
    figsize: Tuple[float, float] = (14, 10),
    show_scatter: bool = True,
    scatter_size: int = 12,
    line_width: float = 1.5,
    show_regime_label: bool = True,
    show_equation: bool = True,
    dpi: int = 150,
) -> plt.Figure:
    r"""
    Plot time series of the logistic map for multiple *r* values.

    Parameters
    ----------
    r_values : list of float, optional
        Parameter values to plot.  Defaults to [2.8, 3.2, 3.5, 3.9].
    x0 : float, default=0.5
        Initial condition :math:`x_0` for all panels.
    iterations : int, default=100
        Number of post-transient iterations to show.
    transient : int, default=0
        Number of burn-in steps (discarded before plotting).
    cols : int, default=2
        Number of columns in the subplot grid.
    figsize : tuple of float, default=(14, 10)
        Figure dimensions.
    show_scatter : bool, default=True
        If True, overlay discrete markers at each iteration.
    scatter_size : int, default=12
        Marker size for the scatter overlay.
    line_width : float, default=1.5
        Line width of the time-series curve.
    show_regime_label : bool, default=True
        If True, annotate each panel with the dynamical regime name.
    show_equation : bool, default=True
        If True, show the logistic map equation in the first panel.
    dpi : int, default=150
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> fig = plot_time_series(r_values=[3.5, 3.9], iterations=50)
    >>> st.pyplot(fig)
    """
    apply_dark_theme(dpi=dpi)

    if r_values is None:
        r_values = DEFAULT_R_VALUES

    n = len(r_values)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = axes.flatten() if n > 1 else [axes]

    iter_idx = np.arange(1, iterations + 1)

    for ax, r in zip(axes_flat, r_values):
        # Generate the orbit
        values = generate_orbit(
            r=r, x0=x0,
            iterations=iterations,
            transient=transient,
        )

        # Colour
        color = LOGISTIC_COLORS.get(r, "#58a6ff")

        # Line plot
        ax.plot(iter_idx, values, color=color,
                linewidth=line_width, zorder=2)

        # Scatter overlay
        if show_scatter:
            ax.scatter(iter_idx, values, color=color,
                       s=scatter_size, zorder=3, edgecolors="none")
            # Connect first & last points with a faint line to indicate cycle
            if iterations > 2 and r not in (2.8,):
                ax.plot([iterations, 1], [values[-1], values[0]],
                        color=color, linewidth=line_width * 0.3,
                        alpha=0.4, linestyle="--")

        # --- annotations ---
        ax.set_title(rf"$r = {r}$", fontsize=14, fontweight="bold",
                     color=color)

        ax.set_xlabel("Iteration  $n$", fontsize=11)
        ax.set_ylabel(r"$x_n$", fontsize=11)
        ax.set_xlim(0, iterations + 1)
        ax.set_ylim(-0.02, 1.02)

        # Regime label
        if show_regime_label and r in REGIME_LABELS:
            ax.text(
                0.97, 0.03, REGIME_LABELS[r],
                transform=ax.transAxes,
                fontsize=9, color="#8b949e",
                ha="right", va="bottom",
                style="italic",
            )

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # --- global equation ---
    if show_equation:
        add_equation_box(
            axes_flat[0],
            r"$x_{n+1} = r \; x_n \; (1 - x_n)$",
            x=0.02, y=0.88,
            fontsize=11,
        )

    fig.suptitle(
        "Logistic Map Time Series",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ============================================================================
# Single-panel time series (for custom r)
# ============================================================================

def plot_single_time_series(
    r: float,
    x0: float = 0.5,
    iterations: int = 200,
    transient: int = 500,
    color: str = "#58a6ff",
    show_phase_line: bool = True,
    figsize: Tuple[float, float] = (12, 5),
    dpi: int = 150,
) -> plt.Figure:
    r"""
    Plot a single detailed time series for a custom *r* value.

    Parameters
    ----------
    r : float
        Growth parameter.
    x0 : float, default=0.5
        Initial condition.
    iterations : int, default=200
        Iterations to display.
    transient : int, default=500
        Burn-in iterations.
    color : str, default="#58a6ff"
        Line colour.
    show_phase_line : bool, default=True
        If True, draw horizontal lines at the fixed-point values for
        comparison.
    figsize : tuple of float, default=(12, 5)
        Figure dimensions.
    dpi : int, default=150
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    apply_dark_theme(dpi=dpi)

    values = generate_orbit(
        r=r, x0=x0,
        iterations=iterations,
        transient=transient,
    )
    iter_idx = np.arange(1, iterations + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(iter_idx, values, color=color, linewidth=1.5, zorder=2)
    ax.scatter(iter_idx, values, color=color, s=15, zorder=3)

    # Fixed point overlay
    if show_phase_line:
        from logistic.logistic_map import fixed_points
        fps = fixed_points(r)
        for fp in fps:
            ax.axhline(fp, color="#8b949e", linestyle=":",
                       alpha=0.4, linewidth=0.8)

    ax.set_xlabel("Iteration  $n$", fontsize=13)
    ax.set_ylabel(r"$x_n$", fontsize=13)
    ax.set_title(
        rf"Time series —  $r = {r}$",
        fontsize=15, fontweight="bold",
    )
    ax.set_xlim(0, iterations + 1)
    ax.set_ylim(-0.02, 1.02)

    add_info_box(ax, [
        rf"$r = {r}$",
        rf"$x_0 = {x0}$",
        rf"Iterations: {iterations}",
        rf"Transient: {transient}",
    ])

    plt.tight_layout()
    return fig


# ============================================================================
# CLI demo
# ============================================================================

if __name__ == "__main__":
    import locale
    locale.setlocale(locale.LC_NUMERIC, "C")

    fig = plot_time_series(show_scatter=True)
    fig.savefig("logistic_time_series.png", dpi=300)
    print("Saved logistic_time_series.png")

    fig2 = plot_single_time_series(r=3.82843, transient=2000, iterations=300)
    fig2.savefig("logistic_single_ts.png", dpi=300)
    print("Saved logistic_single_ts.png")
