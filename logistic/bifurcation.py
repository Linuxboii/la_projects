"""
bifurcation.py — Logistic Map Bifurcation Diagram
==================================================

Produces the iconic bifurcation diagram of the logistic map, showing the
period-doubling route to chaos.  Every vertical slice at a parameter value
*r* shows the long-term attractor of the system.

Mathematical background
-----------------------
The bifurcation diagram plots the asymptotic (post-transient) orbit values
of :math:`x_{n+1} = r x_n (1 - x_n)` against the parameter *r*.  Key
transition points are labelled:

- **r = 1**: transcritical bifurcation — nonzero fixed point appears.
- **r = 3**: first period-doubling bifurcation — period-1 → period-2.
- **r ≈ 3.45**: second period-doubling — period-2 → period-4.
- **r ≈ 3.57**: accumulation point — onset of chaos.
- **r ≈ 3.83**: periodic window of period-3.

The diagram reveals self-similarity: zooming into any small *r*-interval
reproduces the same branching structure at finer scales (a hallmark of
Feigenbaum universality).

References
----------
- Feigenbaum, M. J. (1978). "Quantitative universality for a class of
  nonlinear transformations". *J. Stat. Phys.*, 19, 25-52.
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 10.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from logistic.logistic_map import logistic
from common_styles import (
    apply_dark_theme,
    figure_size,
    add_critical_line,
    add_equation_box,
    add_info_box,
    TRANSITION_COLORS,
)

from typing import Tuple, Optional


# ============================================================================
# Data generation
# ============================================================================

def generate_bifurcation_data(
    r_min: float = 2.4,
    r_max: float = 4.0,
    num_r: int = 5000,
    transient: int = 2000,
    keep: int = 500,
    x0: float = 0.5,
    randomize_x0: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Generate (r, x) scatter data for the bifurcation diagram.

    For each *r* value, the map is iterated *transient* times to
    eliminate transients, then *keep* values are recorded.

    Parameters
    ----------
    r_min : float, default=2.4
        Minimum parameter value.
    r_max : float, default=4.0
        Maximum parameter value.
    num_r : int, default=5000
        Number of evenly spaced *r* values.
    transient : int, default=2000
        Number of burn-in iterations before recording.
    keep : int, default=500
        Number of post-transient values recorded per *r*.
    x0 : float, default=0.5
        Initial condition.  Only relevant if ``randomize_x0=False``.
    randomize_x0 : bool, default=True
        If True, a random initial condition in (0, 1) is used for each
        *r* value, which helps reveal the full attractor.

    Returns
    -------
    r_flat : ndarray, shape (num_r * keep,)
        Flattened array of parameter values.
    x_flat : ndarray, shape (num_r * keep,)
        Flattened array of attractor values.

    Notes
    -----
    Memory use is approximately ``num_r * keep * 16`` bytes
    (two float64 arrays).  For the defaults (5000 × 500) this is
    about 80 MB.

    Examples
    --------
    >>> r, x = generate_bifurcation_data(r_min=3.5, r_max=3.6, num_r=500)
    >>> r.shape, x.shape
    ((250000,), (250000,))
    """
    r_values = np.linspace(r_min, r_max, num_r)
    total_points = num_r * keep
    r_flat = np.empty(total_points, dtype=np.float32)
    x_flat = np.empty(total_points, dtype=np.float32)

    idx = 0
    for r in r_values:
        x = np.random.rand() if randomize_x0 else x0
        # Burn-in
        for _ in range(transient):
            x = logistic(r, x)
        # Record
        for _ in range(keep):
            x = logistic(r, x)
            r_flat[idx] = r
            x_flat[idx] = x
            idx += 1

    return r_flat, x_flat


# ============================================================================
# Plotting
# ============================================================================

def plot_bifurcation(
    r_min: float = 2.4,
    r_max: float = 4.0,
    num_r: int = 5000,
    transient: int = 2000,
    keep: int = 500,
    marker_size: float = 0.02,
    marker_alpha: float = 0.15,
    colormap: str = "plasma",
    color_by_density: bool = True,
    show_transitions: bool = True,
    show_equation: bool = True,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 150,
) -> plt.Figure:
    r"""
    Plot the logistic map bifurcation diagram with mathematical annotation.

    Parameters
    ----------
    r_min : float, default=2.4
        Left limit of the parameter range.
    r_max : float, default=4.0
        Right limit of the parameter range.
    num_r : int, default=5000
        Number of *r* samples (higher → sharper diagram, more memory).
    transient : int, default=2000
        Burn-in iterations per *r* value.
    keep : int, default=500
        Post-transient points recorded per *r*.
    marker_size : float, default=0.02
        Scatter marker size.
    marker_alpha : float, default=0.15
        Scatter marker transparency.
    colormap : str, default="plasma"
        Matplotlib colormap name for density-coloured mode.
    color_by_density : bool, default=True
        If True, colour points by local density using a 2D histogram
        (reveals attractor structure).  If False, use a uniform colour.
    show_transitions : bool, default=True
        If True, annotate critical bifurcation points (r = 3, 3.45,
        3.57, 3.83).
    show_equation : bool, default=True
        If True, display the logistic map equation in the plot.
    figsize : tuple of float, default=(14, 8)
        Figure dimensions in inches.
    dpi : int, default=150
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.  Use ``st.pyplot(fig)`` in Streamlit.

    Examples
    --------
    >>> fig = plot_bifurcation(r_min=3.5, r_max=3.6, num_r=2000)
    >>> fig.savefig("zoom.png", dpi=300)
    """
    apply_dark_theme(dpi=dpi)

    # --- generate data ---
    r_flat, x_flat = generate_bifurcation_data(
        r_min=r_min, r_max=r_max,
        num_r=num_r, transient=transient, keep=keep,
    )

    # --- figure ---
    fig, ax = plt.subplots(figsize=figsize)

    if color_by_density:
        # 2D histogram → density shading
        bins = [2000, 1000]
        hist, xedges, yedges = np.histogram2d(
            r_flat, x_flat, bins=bins,
            range=[[r_min, r_max], [0, 1]],
        )
        # Log-scale colour to enhance low-density regions
        with np.errstate(divide="ignore", invalid="ignore"):
            log_hist = np.log10(hist + 1)
        extent = [r_min, r_max, 0, 1]
        ax.imshow(
            log_hist.T,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=colormap,
        )
    else:
        ax.scatter(
            r_flat, x_flat,
            s=marker_size,
            c="#58a6ff",
            alpha=marker_alpha,
            rasterized=True,
        )

    # --- aesthetics ---
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Growth parameter   $r$", fontsize=14)
    ax.set_ylabel(r"Attractor values   $x$", fontsize=14)
    ax.set_title(
        "Bifurcation Diagram of the Logistic Map",
        fontsize=16, fontweight="bold", pad=12,
    )

    # --- equation box ---
    if show_equation:
        add_equation_box(
            ax,
            r"$x_{n+1} = r \; x_n \; (1 - x_n)$",
            x=0.02, y=0.96,
            fontsize=12,
        )

    # --- critical transition lines ---
    if show_transitions:
        y_top = ax.get_ylim()[1]
        transitions = [
            (3.0, r"$r = 3$  (period-2)", TRANSITION_COLORS["period_doubling"]),
            (3.44949, r"$r \approx 3.45$  (period-4)", TRANSITION_COLORS["period_doubling"]),
            (3.5699456, r"$r_\infty \approx 3.57$  (chaos onset)", TRANSITION_COLORS["chaos_onset"]),
            (3.82843, r"$r \approx 3.83$  (period-3 window)", TRANSITION_COLORS["periodic_window"]),
        ]
        for x_val, label, clr in transitions:
            if r_min <= x_val <= r_max:
                add_critical_line(ax, x_val, label, color=clr, y_max=y_top)

    # --- info box ---
    add_info_box(ax, [
        f"r ∈ [{r_min:.2f}, {r_max:.2f}]",
        f"{num_r:,} r-values × {keep} iterates",
        f"Transient: {transient} steps",
        f"Total points: {len(r_flat):,}",
    ])

    plt.tight_layout()
    return fig


# ============================================================================
# CLI demo
# ============================================================================

if __name__ == "__main__":
    import locale
    locale.setlocale(locale.LC_NUMERIC, "C")
    fig = plot_bifurcation(show_transitions=True, color_by_density=True)
    fig.savefig("bifurcation_diagram.png", dpi=300)
    print("Saved bifurcation_diagram.png")
