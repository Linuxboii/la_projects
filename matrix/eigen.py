r"""
eigen.py — Eigen-decomposition and power iteration
==================================================

An eigenvector :math:`\mathbf{v}` of :math:`A` keeps its direction under the
transform: :math:`A\mathbf{v} = \lambda\mathbf{v}`. The **power iteration**
repeatedly applies :math:`\mathbf{v} \leftarrow A\mathbf{v}/\lVert A\mathbf{v}
\rVert`, converging to the dominant eigenvector; the eigenvalue is estimated by
the Rayleigh quotient :math:`\lambda = \mathbf{v}^\top A \mathbf{v}` (for a
unit vector).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def eig_decompose(A) -> Tuple[np.ndarray, np.ndarray]:
    """Exact eigenvalues and (column) eigenvectors via ``numpy.linalg.eig``."""
    A = np.asarray(A, dtype=float)
    values, vectors = np.linalg.eig(A)
    return values, vectors


@dataclass
class PowerIterationResult:
    """Full record of a power-iteration run."""

    vectors: np.ndarray             # shape (iterations + 1, n), unit vectors
    eigenvalue_estimates: np.ndarray  # shape (iterations + 1,)
    converged: bool
    iterations: int

    @property
    def eigenvector(self) -> np.ndarray:
        return self.vectors[-1]

    @property
    def eigenvalue(self) -> float:
        return float(self.eigenvalue_estimates[-1])


def power_iteration(A, x0=None, max_iter: int = 25, tol: float = 1e-9) -> PowerIterationResult:
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    v = np.ones(n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    v = v / np.linalg.norm(v)
    vectors = [v.copy()]
    lam = float(v @ A @ v)
    estimates = [lam]
    converged = False
    for _ in range(max_iter):
        w = A @ v
        nrm = np.linalg.norm(w)
        if nrm < 1e-15:
            break
        v_new = w / nrm
        if v_new @ v < 0:          # keep the sign stable for a clean animation
            v_new = -v_new
        lam_new = float(v_new @ A @ v_new)
        vectors.append(v_new.copy())
        estimates.append(lam_new)
        if abs(lam_new - lam) < tol:
            v, lam, converged = v_new, lam_new, True
            break
        v, lam = v_new, lam_new
    return PowerIterationResult(np.array(vectors), np.array(estimates), converged, len(vectors) - 1)


if __name__ == "__main__":
    A = [[2, 0, 0], [0, 3, 0], [0, 0, 5]]
    vals, _ = eig_decompose(A)
    res = power_iteration(A, max_iter=200)
    print("exact eigenvalues:", np.round(vals, 4))
    print(f"power iteration -> dominant λ ≈ {res.eigenvalue:.4f} "
          f"(converged={res.converged} in {res.iterations} iters)")
