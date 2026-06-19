r"""
linear_system.py — Direct solving and classification of 3×3 systems
===================================================================

For a linear system :math:`A\mathbf{x} = \mathbf{b}` the solution set is
determined by comparing the rank of the coefficient matrix :math:`A` with the
rank of the augmented matrix :math:`[A\,|\,\mathbf{b}]`:

- ``rank(A) == rank([A|b]) == n`` → a **unique** solution.
- ``rank(A) == rank([A|b]) <  n`` → **infinitely many** solutions.
- ``rank(A) <  rank([A|b])``      → **no** solution (inconsistent).

Three exact solvers are provided for comparison: Gaussian elimination with
partial pivoting, the matrix-inverse method, and Cramer's rule.
"""

from typing import Tuple, List, Dict

import numpy as np

_SINGULAR_TOL = 1e-12


class SingularMatrixError(Exception):
    """Raised when a solver is given a singular (non-invertible) matrix."""


def classify_system(A, b) -> Dict[str, object]:
    r"""Classify the solution set of ``A x = b`` by comparing ranks."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.shape[1]
    Ab = np.column_stack([A, b])
    rank_A = int(np.linalg.matrix_rank(A))
    rank_Ab = int(np.linalg.matrix_rank(Ab))
    det = float(np.linalg.det(A)) if A.shape[0] == A.shape[1] else float("nan")
    if rank_A == rank_Ab == n:
        kind = "unique"
    elif rank_A == rank_Ab and rank_A < n:
        kind = "infinite"
    else:
        kind = "none"
    return {"type": kind, "rank_A": rank_A, "rank_Ab": rank_Ab, "det": det, "n": n}


def solve_inverse(A, b) -> np.ndarray:
    r"""Solve via the matrix inverse: :math:`\mathbf{x} = A^{-1}\mathbf{b}`."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if abs(np.linalg.det(A)) < _SINGULAR_TOL:
        raise SingularMatrixError("Matrix is singular; inverse does not exist.")
    return np.linalg.inv(A) @ b


def solve_cramer(A, b) -> Tuple[np.ndarray, Dict[str, object]]:
    r"""Solve via Cramer's rule: :math:`x_i = \det(A_i)/\det(A)`."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    det_A = float(np.linalg.det(A))
    if abs(det_A) < _SINGULAR_TOL:
        raise SingularMatrixError("Matrix is singular; Cramer's rule fails.")
    n = A.shape[0]
    det_Ai: List[float] = []
    x = np.empty(n)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        di = float(np.linalg.det(Ai))
        det_Ai.append(di)
        x[i] = di / det_A
    return x, {"det_A": det_A, "det_Ai": det_Ai}


def solve_gaussian(A, b) -> Tuple[np.ndarray, List[np.ndarray]]:
    r"""Solve via Gaussian elimination with partial pivoting.

    Returns the solution and a list of snapshots of the augmented matrix
    ``[A | b]`` after each pivot/elimination stage (for visualisation).
    """
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).reshape(-1).copy()
    n = A.shape[0]
    M = np.column_stack([A, b])
    steps: List[np.ndarray] = [M.copy()]
    for col in range(n):
        piv = int(np.argmax(np.abs(M[col:, col]))) + col
        if abs(M[piv, col]) < _SINGULAR_TOL:
            raise SingularMatrixError("Matrix is singular; elimination failed.")
        if piv != col:
            M[[col, piv]] = M[[piv, col]]
        for r in range(col + 1, n):
            factor = M[r, col] / M[col, col]
            M[r, col:] -= factor * M[col, col:]
        steps.append(M.copy())
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - M[i, i + 1:n] @ x[i + 1:n]) / M[i, i]
    return x, steps


if __name__ == "__main__":
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    print("System A x = b — classification:", classify_system(A, b)["type"])
    print("  inverse :", solve_inverse(A, b))
    print("  cramer  :", solve_cramer(A, b)[0])
    print("  gauss   :", solve_gaussian(A, b)[0])
