"""
common_styles.py — Shared mathematical styling for Chaos Theory Visualizations
==============================================================================

Provides a unified, publication-quality aesthetic for all plots in the
LA_project (Logistic Map & Lorenz System visualizations).

Key features:
------------
- Dark-themed, journal-grade mathtext configuration
- Consistent figure sizing and layout conventions
- Scientific colour palettes (plasma, viridis, inferno, cividis)
- LaTeX-compatible axis labels, tick formatting, and annotations
- Helper functions for adding equations, critical-point markers,
  Feigenbaum annotations, and statistical info boxes
- Pre-linted for PEP 8 compatibility

References
----------
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*, 2nd ed.
- Lorenz, E. N. (1963). "Deterministic Nonperiodic Flow". J. Atmos. Sci.
- Feigenbaum, M. J. (1978). "Quantitative universality for a class of
  nonlinear transformations". J. Stat. Phys.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
from typing import Optional, Tuple, List, Dict


# ============================================================================
# Global rcParams overrides — call once per session
# ============================================================================

_DARK_THEME_APPLIED = False


def apply_dark_theme(
    font_size: int = 13,
    dpi: int = 150,
    use_latex: bool = False,
):
    """
    Apply a consistent, dark-mathematical rcParams theme.

    Parameters
    ----------
    font_size : int, default=13
        Base font size for all text elements.
    dpi : int, default=150
        Figure resolution in dots per inch.
    use_latex : bool, default=False
        If True, use ``text.usetex = True`` for full LaTeX rendering
        (requires a working LaTeX installation on the host).

    Notes
    -----
    This function is idempotent — calling it repeatedly will only set the
    parameters once per interpreter session, so it is safe to call from
    every plotting function.

    Examples
    --------
    >>> apply_dark_theme(font_size=14, dpi=200)
    """
    global _DARK_THEME_APPLIED
    if _DARK_THEME_APPLIED:
        return

    rcParams.update({
        # --- Backend & figure ---
        "figure.facecolor": "#0d1117",
        "figure.edgecolor": "#0d1117",
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.facecolor": "#0d1117",
        "savefig.edgecolor": "#0d1117",

        # --- Axes ---
        "axes.facecolor": "#0d1117",
        "axes.edgecolor": "#8b949e",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor": "#c9d1d9",
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.grid.which": "major",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,

        # --- Tick parameters ---
        "xtick.color": "#8b949e",
        "xtick.labelcolor": "#c9d1d9",
        "xtick.direction": "out",
        "xtick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.color": "#8b949e",
        "ytick.labelcolor": "#c9d1d9",
        "ytick.direction": "out",
        "ytick.major.size": 4,
        "ytick.minor.size": 2,

        # --- Grid ---
        "grid.color": "#21262d",
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # --- Fonts ---
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": font_size,
        "text.color": "#c9d1d9",

        # --- Mathtext (use DejaVu Sans's built-in mathtext) ---
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "it",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.cal": "sans:italic",

        # --- LaTeX (optional) ---
        "text.usetex": use_latex,

        # --- Legend ---
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "legend.fancybox": True,
        "legend.framealpha": 0.85,
        "legend.fontsize": "small",
    })

    _DARK_THEME_APPLIED = True


# ============================================================================
# Convenience functions
# ============================================================================

def figure_size(
    width: float = 10,
    aspect: float = 0.618,
) -> Tuple[float, float]:
    """
    Return (width, height) for a figure with the given aspect ratio.

    Parameters
    ----------
    width : float, default=10
        Figure width in inches.
    aspect : float, default=0.618 (golden-ratio conjugate)
        Height-to-width ratio.

    Returns
    -------
    tuple of float
        ``(width, height)`` suitable for ``plt.subplots(figsize=...)``.

    Examples
    --------
    >>> fig, ax = plt.subplots(figsize=figure_size(12, 0.5))
    """
    return (width, width * aspect)


def scientific_axis(ax: plt.Axes) -> plt.Axes:
    """
    Apply scientific-offset formatting to both axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format.

    Returns
    -------
    matplotlib.axes.Axes
        The formatted axes (same object, for chaining).
    """
    ax.ticklabel_format(
        style="sci",
        scilimits=(-3, 4),
        useMathText=True,
    )
    return ax


def add_equation_box(
    ax: plt.Axes,
    eq_text: str,
    x: float = 0.02,
    y: float = 0.98,
    fontsize: int = 11,
    color: str = "#58a6ff",
    ha: str = "left",
    va: str = "top",
):
    """
    Add a coloured equation annotation inside an axes, positioned in
    normalized (0-1) coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    eq_text : str
        LaTeX-style equation string (e.g. ``r"$x_{n+1} = r x_n (1 - x_n)$"``).
    x, y : float, default (0.02, 0.98)
        Normalised coordinates for the anchor point.
    fontsize : int, default=11
        Equation font size.
    color : str, default="#58a6ff"
        Text colour.
    ha, va : str
        Horizontal and vertical alignment.

    Examples
    --------
    >>> add_equation_box(ax, r"$\\dot{x} = \\sigma (y - x)$", fontsize=13)
    """
    # Use figure-level text to avoid 3D Axes issues (text() needs z)
    ax.figure.text(
        x if ax.name != "3d" else x * 0.85 + 0.075,
        y if ax.name != "3d" else y * 0.85 + 0.075,
        eq_text,
        fontsize=fontsize,
        color=color,
        ha=ha, va=va,
        transform=None if ax.name == "3d" else ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#0d1117",
            edgecolor=color,
            linewidth=0.8,
            alpha=0.85,
        ),
    )


def add_critical_line(
    ax: plt.Axes,
    x: float,
    label: str,
    color: str = "#f0883e",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    linestyle: str = "--",
    alpha: float = 0.6,
    label_offset: float = 0.02,
):
    """
    Draw a vertical critical-transition line with a text label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : float
        x-coordinate of the vertical line.
    label : str
        Annotation text (e.g. ``r"$r = 3$"``).
    color : str, default="#f0883e"
        Line and label colour.
    y_min, y_max : float or None
        Vertical span of the line.  ``None`` uses the current y-limits.
    linestyle : str, default="--"
        Line style.
    alpha : float, default=0.6
        Transparency.
    label_offset : float, default=0.02
        Fractional offset from y_max for the label.
    """
    if y_min is None:
        y_min, y_max = ax.get_ylim()
    ax.axvline(x, ymin=0, ymax=1, color=color,
               linestyle=linestyle, alpha=alpha, linewidth=1.0)
    ax.text(
        x, y_max + label_offset * (y_max - y_min),
        label,
        color=color,
        fontsize=9,
        ha="center", va="bottom",
        alpha=alpha,
    )


def add_zero_line(
    ax: plt.Axes,
    color: str = "#8b949e",
    linestyle: str = "-.",
    alpha: float = 0.5,
):
    """
    Draw a horizontal line at y = 0.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    color : str, default="#8b949e"
        Line colour.
    linestyle : str, default="-."
        Line style.
    alpha : float, default=0.5
        Transparency.
    """
    ax.axhline(0, color=color, linestyle=linestyle,
               alpha=alpha, linewidth=0.8)


def add_info_box(
    ax: plt.Axes,
    lines: List[str],
    x: float = 0.98,
    y: float = 0.98,
    fontsize: int = 8,
    color: str = "#8b949e",
):
    """
    Add a small multi-line statistics / info box in the corner of an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    lines : list of str
        Each line is rendered on its own row.
    x, y : float, default (0.98, 0.98)
        Normalised top-right corner.
    fontsize : int, default=8
    color : str, default="#8b949e"
    """
    text = "\n".join(lines)
    # Use figure-level text for 3D compatibility
    target = ax if ax.name != "3d" else ax.figure
    tr = ax.transAxes if ax.name != "3d" else None
    target.text(
        x, y, text,
        transform=tr,
        fontsize=fontsize,
        color=color,
        ha="right", va="top",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#161b22",
            edgecolor="#30363d",
            linewidth=0.5,
            alpha=0.8,
        ),
    )


# ============================================================================
# Colour palettes (accessible from any module)
# ============================================================================

#: Sequential palette suitable for line plots (Katie–Brewer inspired).
LINE_PALETTE = [
    "#58a6ff",  # bright blue
    "#f0883e",  # orange
    "#3fb950",  # green
    "#da3633",  # red
    "#bc8cff",  # purple
    "#79c0ff",  # light blue
    "#ff7b72",  # coral
    "#d2a8ff",  # light purple
]

#: Palette for the four classic logistic-map r regimes.
LOGISTIC_COLORS = {
    2.8: "#58a6ff",   # fixed point — blue
    3.2: "#d29922",   # period-2    — gold
    3.5: "#f0883e",   # period-4    — orange
    3.9: "#da3633",   # chaos       — red
}

#: Default highlight colours for transitions.
TRANSITION_COLORS = {
    "period_doubling": "#f0883e",
    "chaos_onset": "#da3633",
    "periodic_window": "#3fb950",
    "critical": "#bc8cff",
}
