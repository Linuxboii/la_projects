r"""
lorenz_solver.py — Numerical integration of the Lorenz system
=============================================================

Solves the classic Lorenz (1963) system of ordinary differential equations.

Mathematical background
-----------------------
The Lorenz system is a set of three coupled, nonlinear ODEs:

.. math::

    \dot{x} &= \sigma (y - x) \\
    \dot{y} &= x (\rho - z) - y \\
    \dot{z} &= x y - \beta z

where :math:`\sigma` (Prandtl number), :math:`\rho` (Rayleigh number),
and :math:`\beta` (aspect ratio) are positive parameters.

With the classical values :math:`\sigma = 10, \rho = 28, \beta = 8/3`,
the system has three fixed points and exhibits deterministic chaos for most
initial conditions.  The trajectory converges to the famous **Lorenz
attractor** — a fractal set of zero volume but infinite surface area.

References
----------
- Lorenz, E. N. (1963). "Deterministic Nonperiodic Flow".
  *Journal of the Atmospheric Sciences*, 20(2), 130-141.
- Sparrow, C. (1982). *The Lorenz Equations: Bifurcations, Chaos,
  and Strange Attractors*. Springer.
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 9.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional, Callable


# ============================================================================
# Right-hand side function
# ============================================================================

def lorenz_deriv(
    t: float,
    coords: Tuple[float, float, float],
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> Tuple[float, float, float]:
    r"""
    Compute the time derivatives of the Lorenz system.

    Parameters
    ----------
    t : float
        Time variable (required by ``solve_ivp``, unused in the
        autonomous system).
    coords : tuple of float
        Current state ``(x, y, z)``.
    sigma : float, default=10.0
        Prandtl number  :math:`\sigma`.
    rho : float, default=28.0
        Rayleigh number :math:`\rho`.
    beta : float, default=8/3
        Aspect ratio    :math:`\beta`.

    Returns
    -------
    tuple of float
        Derivatives ``(dx/dt, dy/dt, dz/dt)``.

    Examples
    --------
    >>> lorenz_deriv(0, (1.0, 1.0, 1.0), sigma=10, rho=28, beta=8/3)
    (0.0, 26.0, 0.333...)
    """
    x, y, z = coords
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


# ============================================================================
# Fixed points
# ============================================================================

def lorenz_fixed_points(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> Tuple[Tuple[float, float, float], ...]:
    r"""
    Compute the fixed points of the Lorenz system.

    For :math:`\rho < 1`, the origin is the only fixed point.
    For :math:`\rho > 1`, there are three fixed points:

    - **C0**  — origin :math:`(0, 0, 0)` (unstable for :math:`\rho > 1`).
    - **C+**  — :math:`\bigl(\sqrt{\beta(\rho - 1)},
                \sqrt{\beta(\rho - 1)}, \rho - 1\bigr)`.
    - **C-**   — :math:`\bigl(-\sqrt{\beta(\rho - 1)},
                 -\sqrt{\beta(\rho - 1)}, \rho - 1\bigr)`.

    For the classical parameters (:math:`\sigma=10, \rho=28`), C+ and C-
    are the two "eyes" of the butterfly attractor.

    Parameters
    ----------
    sigma : float, default=10.0
    rho : float, default=28.0
    beta : float, default=8/3

    Returns
    -------
    tuple of (x, y, z) tuples
        The fixed points of the system.
    """
    if rho < 1:
        return ((0.0, 0.0, 0.0),)
    c = np.sqrt(beta * (rho - 1.0))
    return (
        (0.0, 0.0, 0.0),
        (c, c, rho - 1.0),
        (-c, -c, rho - 1.0),
    )


# ============================================================================
# Numerical solver
# ============================================================================

def solve_lorenz(
    initial_conditions: Tuple[float, float, float] = (0.0, 1.0, 1.05),
    t_start: float = 0.0,
    t_end: float = 40.0,
    num_points: int = 10000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Numerically integrate the Lorenz system using ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    initial_conditions : tuple of float, default=(0, 1, 1.05)
        Initial state :math:`(x_0, y_0, z_0)`.
    t_start : float, default=0.0
        Start time.
    t_end : float, default=40.0
        End time.  At the default parameters, 40 time units correspond
        to roughly 50 orbits around each lobe.
    num_points : int, default=10000
        Number of time samples (controls curve smoothness).
    sigma : float, default=10.0
        Prandtl number :math:`\sigma`.
    rho : float, default=28.0
        Rayleigh number :math:`\rho`.
    beta : float, default=8/3
        Aspect ratio :math:`\beta`.
    method : str, default="RK45"
        ODE solver method (``"RK45"``, ``"DOP853"``, ``"LSODA"``, etc.).
    rtol, atol : float
        Relative and absolute tolerances for the adaptive integrator.
    max_step : float, default=np.inf
        Maximum step size allowed.

    Returns
    -------
    t : ndarray, shape (num_points,)
        Time grid.
    x : ndarray, shape (num_points,)
        x-coordinate trajectory.
    y : ndarray, shape (num_points,)
        y-coordinate trajectory.
    z : ndarray, shape (num_points,)
        z-coordinate trajectory.

    Raises
    ------
    RuntimeError
        If the integration fails or does not converge.

    Examples
    --------
    >>> t, x, y, z = solve_lorenz(t_end=10, num_points=2000)
    >>> x.min(), x.max()
    (-18.0..., 18.0...)
    """
    t_eval = np.linspace(t_start, t_end, num_points)

    solution = solve_ivp(
        fun=lorenz_deriv,
        t_span=(t_start, t_end),
        y0=list(initial_conditions),
        t_eval=t_eval,
        args=(sigma, rho, beta),
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    if not solution.success:
        raise RuntimeError(
            f"Lorenz integration failed: {solution.message}"
        )

    x = solution.y[0]
    y = solution.y[1]
    z = solution.y[2]

    return t_eval, x, y, z


# ============================================================================
# CLI demo
# ============================================================================

if __name__ == "__main__":
    t, x, y, z = solve_lorenz()

    fps = lorenz_fixed_points()
    print("Lorenz system — classical parameters")
    print(f"  σ = 10, ρ = 28, β = 8/3")
    print(f"  Generated {len(x):,} points  (t ∈ [{t[0]:.1f}, {t[-1]:.1f}])")
    print(f"\nFixed points:")
    for i, fp in enumerate(fps):
        print(f"  C{i}: ({fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f})")
    print(f"\nFirst point:  x = {x[0]:.4f}, y = {y[0]:.4f}, z = {z[0]:.4f}")
    print(f"Last point:   x = {x[-1]:.4f}, y = {y[-1]:.4f}, z = {z[-1]:.4f}")
