r"""
lyapunov.py — Lyapunov Exponent of the Logistic Map
=====================================================

Computes and visualises the Lyapunov exponent λ(r) for the logistic map,
providing a quantitative measure of chaos.

Mathematical background
-----------------------
The Lyapunov exponent measures the average exponential rate of divergence
of nearby trajectories.  For a one-dimensional map :math:`x_{n+1} = f(x_n)`,

.. math::

    \lambda = \lim_{N \to \infty} \frac{1}{N}
              \sum_{n=1}^{N} \ln\left| f'(x_n) \right|.

For the logistic map :math:`f'(x) = r(1 - 2x)`.  The sign of λ
determines the dynamical regime:

- **λ < 0** — stable periodic orbit (contracting).
- **λ = 0** — bifurcation point (neutral stability).
- **λ > 0** — chaos (exponential divergence of nearby initial conditions).

The largest positive λ occurs near r ≈ 4, where the map is fully chaotic.

References
----------
- Lyapunov, A. M. (1892). *The General Problem of the Stability of Motion*.
- Benettin, G. et al. (1980). "Lyapunov Characteristic Exponents for
  smooth dynamical systems". *Meccanica*, 15, 9-20.
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 10.
"""

import numpy as np
import matplotlib.pyplot as plt

from logistic.logistic_map import logistic
from common_styles import (
    apply_dark_theme,
    figure_size,
    add_critical_line,
    add_equation_box,
    add_zero_line,
    add_info_box,
    TRANSITION_COLORS,
)

from typing import Tuple, Optional


# ============================================================================
# Lyapunov exponent computation
# ============================================================================

def lyapunov_exponent(
    r: float,
    x0: float = 0.5,
    transient: int = 1000,
    iterations: int = 2000,
) -> float:
    r"""
    Compute the Lyapunov exponent λ for a single *r* value.

    Parameters
    ----------
    r : float
        Growth parameter.
    x0 : float, default=0.5
        Initial condition.
    transient : int, default=1000
        Number of burn-in steps to discard.
    iterations : int, default=2000
        Number of steps used to estimate λ.

    Returns
    -------
    lam : float
        Estimated Lyapunov exponent.  Positive values indicate chaos.

    Notes
    -----
    The derivative of the logistic map is

    .. math::

        f'(x) = r (1 - 2x).

    The sum is clipped at :math:`\ln(10^{-10})` to prevent singularities
    when :math:`x = 0.5`.

    Examples
    --------
    >>> lyapunov_exponent(2.5)
    -0.219...   # stable
    >>> lyapunov_exponent(3.9)
    0.495...    # chaotic
    """
    x = x0
    # Burn-in
    for _ in range(transient):
        x = logistic(r, x)

    lyap_sum = 0.0
    for _ in range(iterations):
        x = logistic(r, x)
        derivative = abs(r * (1.0 - 2.0 * x))
        # Clip to avoid log(0)
        derivative = max(derivative, 1e-10)
        lyap_sum += np.log(derivative)

    return lyap_sum / iterations


def generate_lyapunov_data(
    r_min: float = 2.4,
    r_max: float = 4.0,
    num_r: int = 3000,
    transient: int = 1000,
    iterations: int = 2000,
    x0: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the Lyapunov spectrum λ(r) over a range of parameter values.

    Parameters
    ----------
    r_min : float, default=2.4
        Start of parameter range.
    r_max : float, default=4.0
        End of parameter range.
    num_r : int, default=3000
        Number of *r* samples.
    transient : int, default=1000
        Burn-in steps per *r*.
    iterations : int, default=2000
        Steps used to compute each λ estimate.
    x0 : float, default=0.5
        Initial condition.

    Returns
    -------
    r_values : ndarray, shape (num_r,)
        Parameter values.
    lyap_values : ndarray, shape (num_r,)
        Corresponding Lyapunov exponent estimates.
    """
    r_values = np.linspace(r_min, r_max, num_r)
    lyap_values = np.empty(num_r)

    for i, r in enumerate(r_values):
        lyap_values[i] = lyapunov_exponent(
            r, x0=x0,
            transient=transient,
            iterations=iterations,
        )

    return r_values, lyap_values


# ============================================================================
# Plotting
# ============================================================================

def plot_lyapunov(
    r_min: float = 2.4,
    r_max: float = 4.0,
    num_r: int = 3000,
    transient: int = 1000,
    iterations: int = 2000,
    line_color: str = "#58a6ff",
    line_width: float = 1.2,
    chaotic_fill_color: str = "#da3633",
    stable_fill_color: str = "#3fb950",
    chaotic_alpha: float = 0.25,
    stable_alpha: float = 0.15,
    show_zero_line: bool = True,
    show_transitions: bool = True,
    show_equation: bool = True,
    figsize: Tuple[float, float] = (14, 6),
    dpi: int = 150,
) -> plt.Figure:
    r"""
    Plot the Lyapunov exponent λ(r) of the logistic map.

    Parameters
    ----------
    r_min : float, default=2.4
        Left limit of the parameter range.
    r_max : float, default=4.0
        Right limit of the parameter range.
    num_r : int, default=3000
        Number of *r* samples.
    transient : int, default=1000
        Burn-in per sample.
    iterations : int, default=2000
        Iterations per λ estimate.
    line_color : str, default="#58a6ff"
        Colour of the λ(r) curve.
    line_width : float, default=1.2
        Line width of the λ(r) curve.
    chaotic_fill_color : str, default="#da3633"
        Fill colour for λ > 0 (chaotic) regions.
    stable_fill_color : str, default="#3fb950"
        Fill colour for λ < 0 (stable) regions.
    chaotic_alpha : float, default=0.25
        Transparency of chaotic fill.
    stable_alpha : float, default=0.15
        Transparency of stable fill.
    show_zero_line : bool, default=True
        If True, draw a horizontal line at λ = 0.
    show_transitions : bool, default=True
        If True, annotate key bifurcation points.
    show_equation : bool, default=True
        If True, show the Lyapunov formula.
    figsize : tuple of float, default=(14, 6)
        Figure dimensions.
    dpi : int, default=150
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> fig = plot_lyapunov(r_min=3.0, r_max=4.0, num_r=2000)
    >>> plt.show()
    """
    apply_dark_theme(dpi=dpi)

    r_values, lyap_values = generate_lyapunov_data(
        r_min=r_min, r_max=r_max,
        num_r=num_r, transient=transient,
        iterations=iterations,
    )

    fig, ax = plt.subplots(figsize=figsize)

    # --- main curve ---
    ax.plot(r_values, lyap_values, color=line_color,
            linewidth=line_width, zorder=3)

    # --- fill chaotic / stable regions ---
    ax.fill_between(
        r_values, 0, lyap_values,
        where=(lyap_values > 0),
        color=chaotic_fill_color, alpha=chaotic_alpha,
        label=r"$\lambda > 0$  (chaotic)",
    )
    ax.fill_between(
        r_values, 0, lyap_values,
        where=(lyap_values < 0),
        color=stable_fill_color, alpha=stable_alpha,
        label=r"$\lambda < 0$  (stable)",
    )

    # --- zero line ---
    if show_zero_line:
        add_zero_line(ax)

    # --- annotations ---
    ax.set_xlabel(r"Growth parameter   $r$", fontsize=14)
    ax.set_ylabel(r"Lyapunov exponent   $\lambda(r)$", fontsize=14)
    ax.set_title(
        "Lyapunov Exponent of the Logistic Map",
        fontsize=16, fontweight="bold", pad=12,
    )

    ax.set_xlim(r_min, r_max)

    # Determine y-limits from data (with some padding)
    y_min = min(lyap_values.min(), -0.5) - 0.1
    y_max = max(lyap_values.max(), 0.5) + 0.1
    ax.set_ylim(y_min, y_max)

    # --- equation box ---
    if show_equation:
        add_equation_box(
            ax,
            r"$\lambda = \frac{1}{N}\sum_{n=1}^{N} \ln| r(1-2x_n) |$",
            x=0.02, y=0.96,
            fontsize=11,
        )

    # --- transition lines ---
    if show_transitions:
        y_top = y_max
        for x_val, label, clr in [
            (3.0, r"$r=3$", TRANSITION_COLORS["period_doubling"]),
            (3.5699456, r"$r_\infty$", TRANSITION_COLORS["chaos_onset"]),
            (3.82843, r"$r\approx3.83$", TRANSITION_COLORS["periodic_window"]),
        ]:
            if r_min <= x_val <= r_max:
                add_critical_line(ax, x_val, label, color=clr, y_max=y_top)

    ax.legend(loc="lower right", fontsize=10)

    # --- info box ---
    chaotic_frac = np.sum(lyap_values > 0) / len(lyap_values)
    add_info_box(ax, [
        f"r ∈ [{r_min:.2f}, {r_max:.2f}]",
        f"Samples: {num_r:,}",
        f"Chaotic fraction: {chaotic_frac:.1%}",
        f"Max λ = {lyap_values.max():.4f}  (r ≈ {r_values[lyap_values.argmax()]:.4f})",
    ])

    plt.tight_layout()
    return fig


# ============================================================================
# CLI demo
# ============================================================================

if __name__ == "__main__":
    import locale
    locale.setlocale(locale.LC_NUMERIC, "C")
    fig = plot_lyapunov(show_transitions=True)
    fig.savefig("lyapunov_exponent.png", dpi=300)
    print("Saved lyapunov_exponent.png")
