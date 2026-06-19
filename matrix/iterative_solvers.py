r"""
iterative_solvers.py — Stationary iterative methods with full history
=====================================================================

Each solver starts from a guess :math:`\mathbf{x}^{(0)}` and produces a
sequence :math:`\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots` that, for a
*diagonally dominant* matrix, converges to the true solution.

- **Jacobi**:        :math:`x_i^{(k+1)} = \frac{1}{a_{ii}}\bigl(b_i -
  \sum_{j\ne i} a_{ij} x_j^{(k)}\bigr)`
- **Gauss-Seidel**:  uses already-updated components within a sweep.
- **SOR**:           Gauss-Seidel with relaxation factor :math:`\omega`.

Every method records the complete iterate history and the residual
:math:`\lVert A\mathbf{x}^{(k)} - \mathbf{b}\rVert_2` at each step so the
convergence (or divergence) can be visualised.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class IterationResult:
    """Full record of an iterative solve."""

    history: np.ndarray   # shape (iterations + 1, n)
    residuals: np.ndarray  # shape (iterations + 1,)
    converged: bool
    iterations: int

    @property
    def solution(self) -> np.ndarray:
        return self.history[-1]


def _residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(A @ x - b))


def is_diagonally_dominant(A) -> bool:
    r"""True if ``A`` is (weakly) diagonally dominant with at least one
    strictly dominant row."""
    A = np.asarray(A, dtype=float)
    diag = np.abs(np.diag(A))
    off = np.sum(np.abs(A), axis=1) - diag
    return bool(np.all(diag >= off)) and bool(np.any(diag > off))


def _prepare(A, b, x0):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    return A, b, n, x


def jacobi(A, b, x0=None, max_iter: int = 25, tol: float = 1e-8) -> IterationResult:
    A, b, n, x = _prepare(A, b, x0)
    diag = np.diag(A)
    R = A - np.diagflat(diag)
    history = [x.copy()]
    residuals = [_residual(A, b, x)]
    converged = False
    for _ in range(max_iter):
        x = (b - R @ x) / diag
        history.append(x.copy())
        residuals.append(_residual(A, b, x))
        if residuals[-1] < tol:
            converged = True
            break
    return IterationResult(np.array(history), np.array(residuals), converged, len(history) - 1)


def gauss_seidel(A, b, x0=None, max_iter: int = 25, tol: float = 1e-8) -> IterationResult:
    A, b, n, x = _prepare(A, b, x0)
    history = [x.copy()]
    residuals = [_residual(A, b, x)]
    converged = False
    for _ in range(max_iter):
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i + 1:] @ x[i + 1:]
            x[i] = (b[i] - s) / A[i, i]
        history.append(x.copy())
        residuals.append(_residual(A, b, x))
        if residuals[-1] < tol:
            converged = True
            break
    return IterationResult(np.array(history), np.array(residuals), converged, len(history) - 1)


def sor(A, b, omega: float = 1.1, x0=None, max_iter: int = 25, tol: float = 1e-8) -> IterationResult:
    A, b, n, x = _prepare(A, b, x0)
    history = [x.copy()]
    residuals = [_residual(A, b, x)]
    converged = False
    for _ in range(max_iter):
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i + 1:] @ x[i + 1:]
            x[i] = (1.0 - omega) * x[i] + omega * (b[i] - s) / A[i, i]
        history.append(x.copy())
        residuals.append(_residual(A, b, x))
        if residuals[-1] < tol:
            converged = True
            break
    return IterationResult(np.array(history), np.array(residuals), converged, len(history) - 1)


if __name__ == "__main__":
    A = [[4, 1, 1], [1, 5, 2], [1, 2, 6]]
    b = [6, 8, 9]
    for name, fn in (("jacobi", jacobi), ("gauss_seidel", gauss_seidel)):
        res = fn(A, b, max_iter=100)
        print(f"{name}: converged={res.converged} in {res.iterations} iters -> {res.solution}")
