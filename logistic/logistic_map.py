"""
logistic_map.py — The Logistic Map kernel
==========================================

Defines the canonical logistic recurrence relation and provides orbit
generation for arbitrary parameter values.

Mathematical background
-----------------------
The logistic map is defined by the recurrence

    x_{n+1} = r · x_n · (1 - x_n),    x_n ∈ [0, 1],  r ∈ [0, 4].

Despite its simplicity, it exhibits a rich variety of dynamical regimes:
fixed points, periodic cycles of period 2^k, and deterministic chaos.
The period-doubling route to chaos, first studied in detail by
Feigenbaum (1978), is one of the universal scaling routes found in
many nonlinear systems.

References
----------
- May, R. M. (1976). "Simple mathematical models with very complicated
  dynamics". *Nature*, 261, 459-467.
- Feigenbaum, M. J. (1978). "Quantitative universality for a class of
  nonlinear transformations". *J. Stat. Phys.*, 19, 25-52.
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*, 2nd ed.,
  Westview Press, Ch. 10.
"""

import numpy as np
from typing import Optional, Union, Tuple


# ============================================================================
# Core map
# ============================================================================

def logistic(r: Union[float, np.ndarray],
            x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Compute one iteration of the logistic map.

    .. math::

        x_{n+1} = r \, x_n \, (1 - x_n)

    Parameters
    ----------
    r : float or ndarray
        Growth parameter (typically 0 ≤ r ≤ 4).  The dynamical regime
        is determined by *r*:

        - ``0 ≤ r < 1``   — extinction (x → 0)
        - ``1 ≤ r < 3``   — stable fixed point
        - ``3 ≤ r < 3.45`` — period-2 cycle
        - ``3.45 ≤ r < 3.57`` — period-4, -8, … (period doubling)
        - ``3.57 ≤ r ≤ 4`` — chaotic (with periodic windows)

    x : float or ndarray
        Current population value in [0, 1].

    Returns
    -------
    float or ndarray
        Next population value.

    Examples
    --------
    >>> logistic(3.9, 0.5)
    0.975
    >>> logistic(3.9, np.array([0.2, 0.5]))
    array([0.624, 0.975])
    """
    return r * x * (1.0 - x)


# ============================================================================
# Orbit generation
# ============================================================================

def generate_orbit(
    r: float,
    x0: float = 0.5,
    iterations: int = 100,
    transient: int = 0,
    return_burnin: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""
    Generate a sequence of iterates (orbit) of the logistic map.

    Parameters
    ----------
    r : float
        Growth parameter.
    x0 : float, default=0.5
        Initial condition :math:`x_0`.
    iterations : int, default=100
        Number of points to record **after** the transient.
    transient : int, default=0
        Number of burn-in iterations to discard before recording.
        Use a large transient (e.g. 1000) to approach the attractor.
    return_burnin : bool, default=False
        If True, also return the transient points as a separate array.

    Returns
    -------
    values : ndarray, shape (iterations,)
        The orbit after discarding *transient* steps.

    burnin : ndarray, shape (transient,), optional
        Only returned when ``return_burnin=True``.

    Examples
    --------
    >>> orbit = generate_orbit(3.9, x0=0.2, iterations=10)
    >>> orbit.shape
    (10,)

    >>> orbit, burn = generate_orbit(3.5, transient=50, return_burnin=True)
    >>> len(burn)
    50
    """
    # --- burn-in phase ---
    x = x0
    if return_burnin:
        burnin = np.empty(transient) if transient > 0 else np.array([])
        for i in range(transient):
            x = logistic(r, x)
            burnin[i] = x

    else:
        for _ in range(transient):
            x = logistic(r, x)

    # --- recording phase ---
    values = np.empty(iterations)
    for i in range(iterations):
        x = logistic(r, x)
        values[i] = x

    if return_burnin:
        return values, burnin
    return values


# ============================================================================
# Fixed-point analysis helpers
# ============================================================================

def fixed_points(r: float) -> np.ndarray:
    r"""
    Compute the fixed points of the logistic map for a given *r*.

    Fixed points satisfy :math:`x = r x (1 - x)`, giving:

    .. math::

        x^*_0 = 0, \qquad x^*_1 = 1 - \frac{1}{r}.

    Stability is determined by :math:`|f'(x^*)| < 1`.

    Parameters
    ----------
    r : float
        Growth parameter.

    Returns
    -------
    ndarray
        Array of fixed-point values (one or two).
    """
    x0 = 0.0
    if r == 0:
        return np.array([x0])
    x1 = 1.0 - 1.0 / r
    return np.array([x0, x1])


def fixed_point_stability(r: float, x_star: float) -> str:
    r"""
    Determine the linear stability of a fixed point.

    The multiplier is :math:`f'(x^*) = r (1 - 2 x^*)`.

    Parameters
    ----------
    r : float
        Growth parameter.
    x_star : float
        Fixed-point value.

    Returns
    -------
    str
        ``"stable"``, ``"unstable"``, or ``"neutral"``.
    """
    derivative = abs(r * (1.0 - 2.0 * x_star))
    if derivative < 1.0:
        return "stable"
    if derivative == 1.0:
        return "neutral"
    return "unstable"


# ============================================================================
# Superstable r-values (Feigenbaum attractor)
# ============================================================================

def superstable_r(n: int) -> float:
    r"""
    Return the approximate superstable r value for the n-th period-doubling.

    Superstable orbits satisfy :math:`f^{2^n}(x^*) = 0` and lie at the
    centre of period-n windows.  They converge geometrically to the
    accumulation point :math:`r_\infty \approx 3.5699456` with the
    Feigenbaum constant :math:`\delta \approx 4.6692`.

    Parameters
    ----------
    n : int
        Period-doubling level (n=1 → period-2, n=2 → period-4, …).

    Returns
    -------
    float
        Approximate superstable parameter value.
    """
    # Empirically known superstable r values (Strogatz, Table 10.1)
    known = {
        1: 3.0,
        2: 3.44949,
        3: 3.54409,
        4: 3.56441,
        5: 3.56876,
        6: 3.56969,
        7: 3.56989,
    }
    if n in known:
        return known[n]
    # Extrapolate using Feigenbaum's delta
    delta = 4.669201609102990
    r_inf = 3.569945671870944
    return r_inf - (r_inf - known[5]) / (delta ** (n - 5))


# ============================================================================
# CLI demo
# ============================================================================

if __name__ == "__main__":
    import locale
    locale.setlocale(locale.LC_NUMERIC, "C")

    r = 3.9
    orbit = generate_orbit(r=r, x0=0.5, iterations=20)
    print(f"Logistic map orbit — r = {r}")
    print("-" * 40)
    for i, val in enumerate(orbit, start=1):
        print(f"  x[{i:2d}] = {val:.12f}")

    print(f"\nFixed points for r = {r}:")
    for fp in fixed_points(r):
        stab = fixed_point_stability(r, fp)
        print(f"  x* = {fp:.6f}  [{stab}]")
