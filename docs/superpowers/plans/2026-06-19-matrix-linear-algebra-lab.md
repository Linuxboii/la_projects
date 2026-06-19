# Matrix — Linear Algebra Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `matrix/` module to `LA_project` that solves 3×3 linear systems (direct + iterative methods) and performs eigen-analysis, visualised in interactive Plotly 3D inside the existing Streamlit dashboard.

**Architecture:** Pure NumPy computation modules (`linear_system`, `iterative_solvers`, `eigen`, `presets`) kept separate from Plotly figure builders (`plot_planes`, `plot_transform`), wired into `dashboard.py` as three new pages. Iterative solvers return full per-iteration history so the UI can show convergence stepping toward the solution. Matches the existing `lorenz/` and `logistic/` module style (documented modules, `__main__` demo, `plot_*` returning figures).

**Tech Stack:** Python 3, NumPy, SciPy, Plotly (interactive 3D), Streamlit, pandas (for `st.data_editor` grids), pytest (tests).

## Global Constraints

- All commands run from repo root: `/home/parzival/LA_project`.
- The `matrix` package is imported as `from matrix.<module> import ...` (cwd = repo root).
- Computation functions accept array-likes and coerce with `np.asarray(..., dtype=float)`; they never depend on Plotly/Streamlit.
- Plot builders return a `plotly.graph_objects.Figure`; they never call `st.*` or `fig.show()`.
- Singular/invalid matrices raise `matrix.linear_system.SingularMatrixError` (a clear, catchable exception) — never an uncaught traceback in the UI.
- Match house style: NumPy-style docstrings with a short math note, type hints, and a `__main__` demo in each computation module.
- Default iterative `max_iter=25`, `tol=1e-8`; UI iteration slider minimum is 5.

---

### Task 1: Package skeleton, test setup, dependencies

**Files:**
- Create: `matrix/__init__.py`
- Create: `tests/matrix/__init__.py`
- Create: `tests/matrix/test_smoke.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: importable `matrix` package; `pytest` runnable from repo root.

- [ ] **Step 1: Install pytest and add it to requirements**

Run: `pip install pytest`

Then add a line to `requirements.txt` (append, keep existing lines):

```
pytest
```

- [ ] **Step 2: Create the package `__init__.py`**

Create `matrix/__init__.py`:

```python
"""
matrix — Linear Algebra Lab
===========================

Solve and visualise 3×3 linear systems and eigen-problems in interactive 3D.

Modules
-------
linear_system      Solution classification and direct solvers (elimination,
                   inverse, Cramer's rule).
iterative_solvers  Jacobi / Gauss-Seidel / SOR with full iteration history.
eigen              Eigen-decomposition and power iteration.
presets            Curated example systems (unique / none / infinite /
                   convergent / divergent).
plot_planes        Plotly 3D: equation planes, intersection, convergence path.
plot_transform     Plotly 3D: matrix transform of a unit cube + eigenvectors.

References
----------
- Strang, G. (2016). *Introduction to Linear Algebra*, 5th ed.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
"""
```

(Submodule imports are added as later tasks create the files, to avoid import errors now.)

- [ ] **Step 3: Create the tests package and a smoke test**

Create `tests/matrix/__init__.py` (empty file).

Create `tests/matrix/test_smoke.py`:

```python
def test_matrix_package_imports():
    import matrix
    assert matrix.__doc__ is not None
```

- [ ] **Step 4: Run the smoke test from repo root**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add matrix/__init__.py tests/matrix/__init__.py tests/matrix/test_smoke.py requirements.txt
git commit -m "feat(matrix): package skeleton and pytest setup"
```

---

### Task 2: `linear_system.py` — classification and direct solvers

**Files:**
- Create: `matrix/linear_system.py`
- Test: `tests/matrix/test_linear_system.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class SingularMatrixError(Exception)`
  - `classify_system(A, b) -> dict` with keys `type` (`"unique"|"infinite"|"none"`), `rank_A:int`, `rank_Ab:int`, `det:float`, `n:int`.
  - `solve_inverse(A, b) -> np.ndarray` (shape `(n,)`); raises `SingularMatrixError`.
  - `solve_cramer(A, b) -> tuple[np.ndarray, dict]` where dict has `det_A:float`, `det_Ai:list[float]`; raises `SingularMatrixError`.
  - `solve_gaussian(A, b) -> tuple[np.ndarray, list[np.ndarray]]` (solution, list of augmented-matrix snapshots); raises `SingularMatrixError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/matrix/test_linear_system.py`:

```python
import numpy as np
import pytest

from matrix.linear_system import (
    SingularMatrixError,
    classify_system,
    solve_inverse,
    solve_cramer,
    solve_gaussian,
)

# Classic system with solution (2, 3, -1)
A_UNIQUE = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
B_UNIQUE = [8, -11, -3]
X_UNIQUE = np.array([2.0, 3.0, -1.0])

A_NONE = [[1, 1, 1], [1, 1, 1], [1, 2, 3]]
B_NONE = [1, 2, 4]

A_INF = [[1, 1, 1], [2, 2, 2], [1, 2, 3]]
B_INF = [6, 12, 14]

A_SINGULAR = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
B_SING = [1, 2, 3]


def test_classify_unique():
    info = classify_system(A_UNIQUE, B_UNIQUE)
    assert info["type"] == "unique"
    assert info["rank_A"] == 3 and info["rank_Ab"] == 3 and info["n"] == 3


def test_classify_none():
    assert classify_system(A_NONE, B_NONE)["type"] == "none"


def test_classify_infinite():
    assert classify_system(A_INF, B_INF)["type"] == "infinite"


def test_solve_inverse_matches_numpy():
    x = solve_inverse(A_UNIQUE, B_UNIQUE)
    assert np.allclose(x, X_UNIQUE)


def test_solve_cramer_matches_numpy():
    x, dets = solve_cramer(A_UNIQUE, B_UNIQUE)
    assert np.allclose(x, X_UNIQUE)
    assert len(dets["det_Ai"]) == 3


def test_solve_gaussian_matches_numpy_and_records_steps():
    x, steps = solve_gaussian(A_UNIQUE, B_UNIQUE)
    assert np.allclose(x, X_UNIQUE)
    assert len(steps) >= 2  # at least initial + final
    assert steps[0].shape == (3, 4)


def test_singular_raises():
    for solver in (solve_inverse, solve_cramer, solve_gaussian):
        with pytest.raises(SingularMatrixError):
            solver(A_SINGULAR, B_SING)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_linear_system.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'matrix.linear_system'`.

- [ ] **Step 3: Write the implementation**

Create `matrix/linear_system.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_linear_system.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Update package `__init__.py`**

In `matrix/__init__.py`, add at the end:

```python
from . import linear_system  # noqa: E402,F401
```

- [ ] **Step 6: Commit**

```bash
git add matrix/linear_system.py matrix/__init__.py tests/matrix/test_linear_system.py
git commit -m "feat(matrix): direct solvers and solution classification"
```

---

### Task 3: `iterative_solvers.py` — Jacobi / Gauss-Seidel / SOR with history

**Files:**
- Create: `matrix/iterative_solvers.py`
- Test: `tests/matrix/test_iterative_solvers.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass IterationResult` with fields `history: np.ndarray` (shape `(k+1, n)`), `residuals: np.ndarray` (shape `(k+1,)`), `converged: bool`, `iterations: int`, and a `solution` property returning `history[-1]`.
  - `is_diagonally_dominant(A) -> bool`
  - `jacobi(A, b, x0=None, max_iter=25, tol=1e-8) -> IterationResult`
  - `gauss_seidel(A, b, x0=None, max_iter=25, tol=1e-8) -> IterationResult`
  - `sor(A, b, omega=1.1, x0=None, max_iter=25, tol=1e-8) -> IterationResult`

- [ ] **Step 1: Write the failing tests**

Create `tests/matrix/test_iterative_solvers.py`:

```python
import numpy as np

from matrix.iterative_solvers import (
    IterationResult,
    is_diagonally_dominant,
    jacobi,
    gauss_seidel,
    sor,
)

# Diagonally dominant → all methods converge
A_DD = [[4.0, 1.0, 1.0], [1.0, 5.0, 2.0], [1.0, 2.0, 6.0]]
B_DD = [6.0, 8.0, 9.0]

# Strongly off-diagonal → Jacobi diverges
A_BAD = [[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0]]
B_BAD = [1.0, 1.0, 1.0]


def _true(A, b):
    return np.linalg.solve(np.asarray(A, float), np.asarray(b, float))


def test_diagonal_dominance_check():
    assert is_diagonally_dominant(A_DD) is True
    assert is_diagonally_dominant(A_BAD) is False


def test_jacobi_converges_on_dominant():
    res = jacobi(A_DD, B_DD, max_iter=100)
    assert isinstance(res, IterationResult)
    assert res.converged
    assert np.allclose(res.solution, _true(A_DD, B_DD), atol=1e-6)


def test_history_and_residual_shapes_align():
    res = jacobi(A_DD, B_DD, max_iter=100)
    assert res.history.shape[0] == res.residuals.shape[0]
    assert res.history.shape[0] == res.iterations + 1
    # residual decreases overall on a convergent system
    assert res.residuals[-1] < res.residuals[0]


def test_gauss_seidel_converges():
    res = gauss_seidel(A_DD, B_DD, max_iter=100)
    assert res.converged
    assert np.allclose(res.solution, _true(A_DD, B_DD), atol=1e-6)


def test_sor_converges():
    res = sor(A_DD, B_DD, omega=1.1, max_iter=100)
    assert res.converged
    assert np.allclose(res.solution, _true(A_DD, B_DD), atol=1e-6)


def test_jacobi_diverges_on_bad_matrix():
    res = jacobi(A_BAD, B_BAD, max_iter=15)
    assert not res.converged
    assert res.residuals[-1] > res.residuals[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_iterative_solvers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'matrix.iterative_solvers'`.

- [ ] **Step 3: Write the implementation**

Create `matrix/iterative_solvers.py`:

```python
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
from typing import Optional

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_iterative_solvers.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Update package `__init__.py`**

Append to `matrix/__init__.py`:

```python
from . import iterative_solvers  # noqa: E402,F401
```

- [ ] **Step 6: Commit**

```bash
git add matrix/iterative_solvers.py matrix/__init__.py tests/matrix/test_iterative_solvers.py
git commit -m "feat(matrix): Jacobi/Gauss-Seidel/SOR iterative solvers with history"
```

---

### Task 4: `presets.py` — curated example systems

**Files:**
- Create: `matrix/presets.py`
- Test: `tests/matrix/test_presets.py`

**Interfaces:**
- Consumes: `matrix.linear_system.classify_system`, `matrix.iterative_solvers.is_diagonally_dominant`.
- Produces:
  - `PRESETS: dict[str, dict]` (each value has `A`, `b`, `note`).
  - `list_presets() -> list[str]`
  - `get_preset(name) -> tuple[np.ndarray, np.ndarray]` (A as `(3,3)` float, b as `(3,)` float).

- [ ] **Step 1: Write the failing tests**

Create `tests/matrix/test_presets.py`:

```python
import numpy as np

from matrix.presets import PRESETS, list_presets, get_preset
from matrix.linear_system import classify_system
from matrix.iterative_solvers import is_diagonally_dominant


def test_list_presets_nonempty():
    names = list_presets()
    assert len(names) >= 5
    assert "Unique solution" in names


def test_get_preset_shapes():
    A, b = get_preset("Unique solution")
    assert A.shape == (3, 3)
    assert b.shape == (3,)


def test_solution_type_presets_classify_correctly():
    assert classify_system(*get_preset("Unique solution"))["type"] == "unique"
    assert classify_system(*get_preset("No solution"))["type"] == "none"
    assert classify_system(*get_preset("Infinite solutions"))["type"] == "infinite"


def test_convergence_presets_have_expected_dominance():
    assert is_diagonally_dominant(get_preset("Diagonally dominant (converges)")[0])
    assert not is_diagonally_dominant(get_preset("Not dominant (diverges)")[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_presets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'matrix.presets'`.

- [ ] **Step 3: Write the implementation**

Create `matrix/presets.py`:

```python
r"""
presets.py — Curated 3×3 example systems
========================================

A small gallery of systems chosen to illustrate each behaviour: the three
solution types (unique / none / infinite) and the two convergence regimes
(diagonally dominant → iterative methods converge; strongly off-diagonal →
they diverge).
"""

from typing import Dict, List, Tuple

import numpy as np

PRESETS: Dict[str, Dict[str, object]] = {
    "Unique solution": {
        "A": [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],
        "b": [8, -11, -3],
        "note": "Three planes meet at the single point (2, 3, -1).",
    },
    "No solution": {
        "A": [[1, 1, 1], [1, 1, 1], [1, 2, 3]],
        "b": [1, 2, 4],
        "note": "Two parallel, inconsistent planes — the system is unsolvable.",
    },
    "Infinite solutions": {
        "A": [[1, 1, 1], [2, 2, 2], [1, 2, 3]],
        "b": [6, 12, 14],
        "note": "One equation is a multiple of another; planes share a line.",
    },
    "Diagonally dominant (converges)": {
        "A": [[4, 1, 1], [1, 5, 2], [1, 2, 6]],
        "b": [6, 8, 9],
        "note": "Diagonally dominant — Jacobi/Gauss-Seidel/SOR all converge.",
    },
    "Not dominant (diverges)": {
        "A": [[1, 2, 2], [2, 1, 2], [2, 2, 1]],
        "b": [1, 1, 1],
        "note": "Off-diagonal dominates — Jacobi diverges away from the answer.",
    },
}


def list_presets() -> List[str]:
    return list(PRESETS.keys())


def get_preset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    entry = PRESETS[name]
    A = np.asarray(entry["A"], dtype=float)
    b = np.asarray(entry["b"], dtype=float).reshape(-1)
    return A, b


if __name__ == "__main__":
    for name in list_presets():
        A, b = get_preset(name)
        print(f"{name}: {PRESETS[name]['note']}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_presets.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Update package `__init__.py` and commit**

Append to `matrix/__init__.py`:

```python
from . import presets  # noqa: E402,F401
```

```bash
git add matrix/presets.py matrix/__init__.py tests/matrix/test_presets.py
git commit -m "feat(matrix): curated preset systems"
```

---

### Task 5: `plot_planes.py` — Plotly 3D figures for systems & convergence

**Files:**
- Create: `matrix/plot_planes.py`
- Test: `tests/matrix/test_plot_planes.py`

**Interfaces:**
- Consumes: nothing (takes plain arrays).
- Produces:
  - `plane_figure(A, b, solution=None, size=10.0) -> go.Figure` — three plane surfaces, plus a marker at `solution` if given (so total traces = 3 or 4).
  - `convergence_figure(A, b, history, size=10.0) -> go.Figure` — three plane surfaces plus one `Scatter3d` path of the iterates (total traces = 4).
  - `residual_figure(residuals) -> go.Figure` — a 2D line chart of residual vs iteration (1 trace).

- [ ] **Step 1: Write the failing tests**

Create `tests/matrix/test_plot_planes.py`:

```python
import numpy as np
import plotly.graph_objects as go

from matrix.plot_planes import plane_figure, convergence_figure, residual_figure

A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
B = [8, -11, -3]
SOL = [2.0, 3.0, -1.0]


def test_plane_figure_three_planes_no_solution():
    fig = plane_figure(A, B)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_plane_figure_adds_solution_marker():
    fig = plane_figure(A, B, solution=SOL)
    assert len(fig.data) == 4


def test_convergence_figure_has_path():
    history = np.array([[0, 0, 0], [1, 1, -0.5], [1.8, 2.6, -0.9], SOL])
    fig = convergence_figure(A, B, history)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # 3 planes + 1 path


def test_residual_figure_one_trace():
    fig = residual_figure([10.0, 3.0, 0.8, 0.05])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_plot_planes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'matrix.plot_planes'`.

- [ ] **Step 3: Write the implementation**

Create `matrix/plot_planes.py`:

```python
r"""
plot_planes.py — Interactive Plotly 3D for linear systems
=========================================================

Each equation :math:`a x + b y + c z = d` is a plane in 3-space. A 3×3 system
is three planes; a unique solution is the single point where all three meet.
These builders render the planes as translucent quads (computed from an
in-plane basis so they work at any orientation, including vertical planes),
mark the solution point, and draw the path of an iterative solver walking
toward it.
"""

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

_PLANE_COLORS = ("#4C72B0", "#DD8452", "#55A868")


def _plane_surface(coef: np.ndarray, d: float, size: float):
    r"""Return (X, Y, Z) 2×2 grids for the plane ``coef · p = d`` within a box
    of half-width ``size`` centred on the plane's closest point to the origin."""
    coef = np.asarray(coef, dtype=float)
    norm2 = float(coef @ coef)
    p0 = d * coef / norm2  # closest point on the plane to the origin
    # Build an in-plane orthonormal basis (u, v) perpendicular to the normal.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(coef[0]) > 0.9 * np.linalg.norm(coef):
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(coef, seed)
    u /= np.linalg.norm(u)
    v = np.cross(coef, u)
    v /= np.linalg.norm(v)
    s, t = np.meshgrid([-size, size], [-size, size])
    pts = p0[None, None, :] + s[..., None] * u + t[..., None] * v
    return pts[..., 0], pts[..., 1], pts[..., 2]


def _add_planes(fig: go.Figure, A, b, size: float) -> None:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    for i in range(A.shape[0]):
        X, Y, Z = _plane_surface(A[i], b[i], size)
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                showscale=False, opacity=0.5,
                colorscale=[[0, _PLANE_COLORS[i % 3]], [1, _PLANE_COLORS[i % 3]]],
                name=f"Eq {i + 1}",
            )
        )


def _layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )


def plane_figure(A, b, solution: Optional[Sequence[float]] = None, size: float = 10.0) -> go.Figure:
    fig = go.Figure()
    _add_planes(fig, A, b, size)
    if solution is not None:
        s = np.asarray(solution, dtype=float).reshape(-1)
        fig.add_trace(
            go.Scatter3d(
                x=[s[0]], y=[s[1]], z=[s[2]],
                mode="markers+text", text=["solution"], textposition="top center",
                marker=dict(size=6, color="crimson"), name="solution",
            )
        )
    _layout(fig, "System of equations as intersecting planes")
    return fig


def convergence_figure(A, b, history, size: float = 10.0) -> go.Figure:
    fig = go.Figure()
    _add_planes(fig, A, b, size)
    h = np.asarray(history, dtype=float)
    fig.add_trace(
        go.Scatter3d(
            x=h[:, 0], y=h[:, 1], z=h[:, 2],
            mode="lines+markers",
            marker=dict(size=4, color=np.arange(len(h)), colorscale="Viridis"),
            line=dict(color="crimson", width=4),
            name="iterates",
        )
    )
    _layout(fig, "Iterative solver converging to the intersection")
    return fig


def residual_figure(residuals) -> go.Figure:
    r = np.asarray(residuals, dtype=float)
    fig = go.Figure(
        go.Scatter(x=np.arange(len(r)), y=r, mode="lines+markers", name="residual")
    )
    fig.update_layout(
        title="Residual ‖A x − b‖ per iteration",
        xaxis_title="iteration", yaxis_title="residual",
        yaxis_type="log", height=350, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_plot_planes.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Update package `__init__.py` and commit**

Append to `matrix/__init__.py`:

```python
from . import plot_planes  # noqa: E402,F401
```

```bash
git add matrix/plot_planes.py matrix/__init__.py tests/matrix/test_plot_planes.py
git commit -m "feat(matrix): Plotly 3D plane and convergence figures"
```

---

### Task 6: Dashboard pages — System Solver & Iterative Convergence

**Files:**
- Modify: `dashboard.py` (add two entries to the sidebar `selectbox` list around line 95; append two page blocks at end of file)
- Test: `tests/matrix/test_dashboard_pages.py` (import-smoke for the helper, no Streamlit runtime)

**Interfaces:**
- Consumes: `matrix.presets`, `matrix.linear_system`, `matrix.iterative_solvers`, `matrix.plot_planes`.
- Produces: a reusable helper `matrix/dashboard_helpers.py::system_from_editor(df) -> tuple[np.ndarray, np.ndarray]` to keep parsing logic testable.

- [ ] **Step 1: Write the failing test for the parsing helper**

Create `tests/matrix/test_dashboard_pages.py`:

```python
import numpy as np
import pandas as pd

from matrix.dashboard_helpers import system_from_editor


def test_system_from_editor_splits_A_and_b():
    df = pd.DataFrame(
        [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]],
        columns=["x", "y", "z", "= b"],
    )
    A, b = system_from_editor(df)
    assert A.shape == (3, 3)
    assert b.shape == (3,)
    assert np.allclose(b, [8, -11, -3])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_dashboard_pages.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'matrix.dashboard_helpers'`.

- [ ] **Step 3: Create the helper**

Create `matrix/dashboard_helpers.py`:

```python
r"""dashboard_helpers.py — small pure helpers shared by the Streamlit pages."""

from typing import Tuple

import numpy as np
import pandas as pd


def system_from_editor(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Split an edited 3×4 coefficient table into ``A`` (3×3) and ``b`` (3,)."""
    values = df.to_numpy(dtype=float)
    A = values[:, :3]
    b = values[:, 3]
    return A, b
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_dashboard_pages.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Add the two new pages to the sidebar list**

In `dashboard.py`, find the `selectbox` options list (the page list passed to `st.sidebar.selectbox`, around line 95) and add two entries to the end of that list:

```python
        "LA — System Solver",
        "LA — Iterative Convergence",
```

- [ ] **Step 6: Append the two page blocks at the end of `dashboard.py`**

Append to the end of `dashboard.py`:

```python
# ============================================================================
# Page — Linear Algebra: System Solver
# ============================================================================

elif page == "LA — System Solver":
    import numpy as np
    import pandas as pd

    from matrix.presets import list_presets, get_preset, PRESETS
    from matrix.linear_system import (
        classify_system, solve_inverse, solve_cramer, solve_gaussian,
        SingularMatrixError,
    )
    from matrix.plot_planes import plane_figure
    from matrix.dashboard_helpers import system_from_editor

    st.header("Linear Algebra — System Solver")
    st.markdown(
        r"""
        Each equation $a x + b y + c z = d$ is a **plane**. The solution of the
        system is where the three planes meet. Edit the coefficients or pick a
        preset, and compare three exact solving methods.
        """
    )

    preset = st.sidebar.selectbox("Preset system", list_presets())
    st.sidebar.caption(PRESETS[preset]["note"])
    A0, b0 = get_preset(preset)

    df = pd.DataFrame(
        np.column_stack([A0, b0]), columns=["x", "y", "z", "= b"]
    )
    edited = st.data_editor(df, key=f"sys_{preset}", use_container_width=True)
    A, b = system_from_editor(edited)

    info = classify_system(A, b)
    st.subheader(f"Solution type: **{info['type'].upper()}**")
    st.write(
        f"rank(A) = {info['rank_A']}, rank([A|b]) = {info['rank_Ab']}, "
        f"det(A) = {info['det']:.3f}"
    )

    solution = None
    if info["type"] == "unique":
        try:
            x_inv = solve_inverse(A, b)
            x_cra, _ = solve_cramer(A, b)
            x_gau, _ = solve_gaussian(A, b)
            solution = x_gau
            c1, c2, c3 = st.columns(3)
            c1.metric("Inverse method", f"({x_inv[0]:.3f}, {x_inv[1]:.3f}, {x_inv[2]:.3f})")
            c2.metric("Cramer's rule", f"({x_cra[0]:.3f}, {x_cra[1]:.3f}, {x_cra[2]:.3f})")
            c3.metric("Gaussian elim.", f"({x_gau[0]:.3f}, {x_gau[1]:.3f}, {x_gau[2]:.3f})")
        except SingularMatrixError as exc:
            st.warning(f"Direct solve failed: {exc}")
    else:
        st.info("No single point solution — see the plane arrangement below.")

    st.plotly_chart(plane_figure(A, b, solution=solution), use_container_width=True)


# ============================================================================
# Page — Linear Algebra: Iterative Convergence
# ============================================================================

elif page == "LA — Iterative Convergence":
    import numpy as np
    import pandas as pd

    from matrix.presets import list_presets, get_preset, PRESETS
    from matrix.iterative_solvers import jacobi, gauss_seidel, sor, is_diagonally_dominant
    from matrix.plot_planes import convergence_figure, residual_figure
    from matrix.dashboard_helpers import system_from_editor

    st.header("Linear Algebra — Iterative Convergence")
    st.markdown(
        r"""
        Iterative methods start from a guess and **step toward** the solution.
        Watch each iterate walk through 3-space toward the intersection — and
        see how a non-diagonally-dominant matrix makes them diverge instead.
        """
    )

    preset = st.sidebar.selectbox(
        "Preset system", list_presets(),
        index=list_presets().index("Diagonally dominant (converges)"),
    )
    st.sidebar.caption(PRESETS[preset]["note"])
    method_name = st.sidebar.selectbox("Method", ["Jacobi", "Gauss-Seidel", "SOR"])
    max_iter = st.sidebar.slider("Iterations", min_value=5, max_value=100, value=25)
    omega = st.sidebar.slider("SOR ω", 0.5, 1.9, 1.1, 0.1) if method_name == "SOR" else 1.1

    A0, b0 = get_preset(preset)
    df = pd.DataFrame(np.column_stack([A0, b0]), columns=["x", "y", "z", "= b"])
    edited = st.data_editor(df, key=f"iter_{preset}", use_container_width=True)
    A, b = system_from_editor(edited)

    dominant = is_diagonally_dominant(A)
    st.write(f"Diagonally dominant: **{dominant}** "
             f"({'methods should converge' if dominant else 'methods may diverge'})")

    if method_name == "Jacobi":
        res = jacobi(A, b, max_iter=max_iter)
    elif method_name == "Gauss-Seidel":
        res = gauss_seidel(A, b, max_iter=max_iter)
    else:
        res = sor(A, b, omega=omega, max_iter=max_iter)

    st.write(
        f"**Converged: {res.converged}** after {res.iterations} iterations; "
        f"final residual = {res.residuals[-1]:.2e}"
    )

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(convergence_figure(A, b, res.history), use_container_width=True)
    with col_b:
        st.plotly_chart(residual_figure(res.residuals), use_container_width=True)

    table = pd.DataFrame(res.history, columns=["x", "y", "z"])
    table["residual"] = res.residuals
    table.index.name = "iteration"
    st.dataframe(table, use_container_width=True)
```

- [ ] **Step 7: Verify the dashboard imports without error**

Run: `cd /home/parzival/LA_project && python -c "import ast; ast.parse(open('dashboard.py').read()); print('dashboard.py parses OK')"`
Expected: `dashboard.py parses OK`

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix -v`
Expected: all tests PASS.

- [ ] **Step 8: Update package `__init__.py` and commit**

Append to `matrix/__init__.py`:

```python
from . import dashboard_helpers  # noqa: E402,F401
```

```bash
git add dashboard.py matrix/dashboard_helpers.py matrix/__init__.py tests/matrix/test_dashboard_pages.py
git commit -m "feat(matrix): System Solver and Iterative Convergence dashboard pages"
```

---

### Task 7: `eigen.py` — eigen-decomposition and power iteration

**Files:**
- Create: `matrix/eigen.py`
- Test: `tests/matrix/test_eigen.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `eig_decompose(A) -> tuple[np.ndarray, np.ndarray]` (eigenvalues, eigenvectors-as-columns), via `numpy.linalg.eig`.
  - `@dataclass PowerIterationResult` with `vectors: np.ndarray` (shape `(k+1, n)`), `eigenvalue_estimates: np.ndarray` (shape `(k+1,)`), `converged: bool`, `iterations: int`, plus `eigenvector` and `eigenvalue` properties returning the last entries.
  - `power_iteration(A, x0=None, max_iter=25, tol=1e-9) -> PowerIterationResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/matrix/test_eigen.py`:

```python
import numpy as np

from matrix.eigen import eig_decompose, power_iteration, PowerIterationResult

# Symmetric matrix with known dominant eigenvalue
A_SYM = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]]


def test_eig_decompose_matches_numpy():
    vals, vecs = eig_decompose(A_SYM)
    assert np.allclose(np.sort(vals), [2.0, 3.0, 5.0])
    assert vecs.shape == (3, 3)


def test_power_iteration_finds_dominant_eigenvalue():
    res = power_iteration(A_SYM, max_iter=200)
    assert isinstance(res, PowerIterationResult)
    assert res.converged
    assert np.isclose(res.eigenvalue, 5.0, atol=1e-4)
    # dominant eigenvector is the z-axis (up to sign)
    assert np.isclose(abs(res.eigenvector[2]), 1.0, atol=1e-3)


def test_power_iteration_history_shapes_align():
    res = power_iteration(A_SYM, max_iter=50)
    assert res.vectors.shape[0] == res.eigenvalue_estimates.shape[0]
    assert res.vectors.shape[0] == res.iterations + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_eigen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'matrix.eigen'`.

- [ ] **Step 3: Write the implementation**

Create `matrix/eigen.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_eigen.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Update package `__init__.py` and commit**

Append to `matrix/__init__.py`:

```python
from . import eigen  # noqa: E402,F401
```

```bash
git add matrix/eigen.py matrix/__init__.py tests/matrix/test_eigen.py
git commit -m "feat(matrix): eigen-decomposition and power iteration"
```

---

### Task 8: `plot_transform.py` — Plotly 3D transform of a unit cube + eigenvectors

**Files:**
- Create: `matrix/plot_transform.py`
- Test: `tests/matrix/test_plot_transform.py`

**Interfaces:**
- Consumes: `matrix.eigen.eig_decompose`.
- Produces:
  - `transform_figure(A, show_eigen=True) -> go.Figure` — draws the original unit cube edges, the transformed cube edges, and (when `show_eigen` and eigenvalues are real) one arrow per real eigenvector. Total traces: 1 (original edges) + 1 (transformed edges) + up to 3 eigenvector arrows.

- [ ] **Step 1: Write the failing tests**

Create `tests/matrix/test_plot_transform.py`:

```python
import plotly.graph_objects as go

from matrix.plot_transform import transform_figure

A_DIAG = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]]


def test_transform_figure_returns_figure_with_cubes():
    fig = transform_figure(A_DIAG, show_eigen=False)
    assert isinstance(fig, go.Figure)
    # original cube edges + transformed cube edges
    assert len(fig.data) == 2


def test_transform_figure_adds_eigenvectors():
    fig = transform_figure(A_DIAG, show_eigen=True)
    # 2 cubes + 3 real eigenvector arrows
    assert len(fig.data) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_plot_transform.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'matrix.plot_transform'`.

- [ ] **Step 3: Write the implementation**

Create `matrix/plot_transform.py`:

```python
r"""
plot_transform.py — Interactive Plotly 3D for matrix transformations
====================================================================

A 3×3 matrix maps the unit cube to a parallelepiped. Eigenvectors are the
special directions that are only **scaled** (not rotated) by the map — drawn
here as arrows whose length reflects the eigenvalue.
"""

import numpy as np
import plotly.graph_objects as go

from .eigen import eig_decompose

# 8 corners of the unit cube and the 12 edges connecting them.
_CUBE = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=float)
_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _edge_trace(points: np.ndarray, color: str, name: str) -> go.Scatter3d:
    xs, ys, zs = [], [], []
    for a, c in _EDGES:
        xs += [points[a, 0], points[c, 0], None]
        ys += [points[a, 1], points[c, 1], None]
        zs += [points[a, 2], points[c, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=color, width=4), name=name)


def transform_figure(A, show_eigen: bool = True) -> go.Figure:
    A = np.asarray(A, dtype=float)
    transformed = _CUBE @ A.T
    fig = go.Figure()
    fig.add_trace(_edge_trace(_CUBE, "#999999", "unit cube"))
    fig.add_trace(_edge_trace(transformed, "#DD8452", "A · cube"))

    if show_eigen:
        values, vectors = eig_decompose(A)
        for i in range(len(values)):
            if abs(values[i].imag) > 1e-9:
                continue  # skip complex eigenpairs (no real direction to draw)
            lam = float(values[i].real)
            vec = vectors[:, i].real
            vec = vec / (np.linalg.norm(vec) or 1.0) * lam
            fig.add_trace(
                go.Scatter3d(
                    x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
                    mode="lines+markers",
                    line=dict(color="crimson", width=6),
                    marker=dict(size=4),
                    name=f"eigvec λ={lam:.2f}",
                )
            )

    fig.update_layout(
        title="Matrix transform of the unit cube (with eigenvectors)",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0), height=600,
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix/test_plot_transform.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Update package `__init__.py` and commit**

Append to `matrix/__init__.py`:

```python
from . import plot_transform  # noqa: E402,F401
```

```bash
git add matrix/plot_transform.py matrix/__init__.py tests/matrix/test_plot_transform.py
git commit -m "feat(matrix): Plotly 3D matrix-transform and eigenvector figure"
```

---

### Task 9: Dashboard page — Eigen Explorer

**Files:**
- Modify: `dashboard.py` (add one entry to the sidebar `selectbox` list; append one page block at end)

**Interfaces:**
- Consumes: `matrix.eigen`, `matrix.plot_transform`.
- Produces: nothing new (terminal page).

- [ ] **Step 1: Add the page to the sidebar list**

In `dashboard.py`, add to the `selectbox` options list (after the two `LA —` entries from Task 6):

```python
        "LA — Eigen Explorer",
```

- [ ] **Step 2: Append the Eigen Explorer page block at the end of `dashboard.py`**

```python
# ============================================================================
# Page — Linear Algebra: Eigen Explorer
# ============================================================================

elif page == "LA — Eigen Explorer":
    import numpy as np
    import pandas as pd

    from matrix.eigen import eig_decompose, power_iteration
    from matrix.plot_transform import transform_figure

    st.header("Linear Algebra — Eigen Explorer")
    st.markdown(
        r"""
        A matrix transforms 3-space. **Eigenvectors** are the directions that
        are only stretched (by the **eigenvalue**), not rotated. Edit the
        matrix and watch the unit cube deform; the red arrows are the
        eigenvectors.
        """
    )

    default = pd.DataFrame(
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]],
        columns=["c1", "c2", "c3"],
    )
    edited = st.data_editor(default, key="eigen_matrix", use_container_width=True)
    A = edited.to_numpy(dtype=float)

    values, vectors = eig_decompose(A)
    st.subheader("Eigenvalues")
    st.write(", ".join(f"{v:.3f}" for v in values))

    show_eigen = st.sidebar.checkbox("Show eigenvectors", value=True)
    st.plotly_chart(transform_figure(A, show_eigen=show_eigen), use_container_width=True)

    st.subheader("Power iteration → dominant eigenvector")
    max_iter = st.sidebar.slider("Power-iteration steps", 5, 100, 25)
    res = power_iteration(A, max_iter=max_iter)
    st.write(
        f"Dominant eigenvalue ≈ **{res.eigenvalue:.4f}** "
        f"(converged={res.converged} in {res.iterations} iterations)"
    )
    est = pd.DataFrame({"eigenvalue estimate": res.eigenvalue_estimates})
    est.index.name = "iteration"
    st.line_chart(est)
```

- [ ] **Step 3: Verify the dashboard parses and the full suite passes**

Run: `cd /home/parzival/LA_project && python -c "import ast; ast.parse(open('dashboard.py').read()); print('dashboard.py parses OK')"`
Expected: `dashboard.py parses OK`

Run: `cd /home/parzival/LA_project && python -m pytest tests/matrix -v`
Expected: all tests PASS.

- [ ] **Step 4: Manual smoke check (optional but recommended)**

Run: `cd /home/parzival/LA_project && streamlit run dashboard.py`
Expected: dashboard launches; the three `LA — …` pages render interactive 3D Plotly figures. Stop with Ctrl+C.

- [ ] **Step 5: Commit**

```bash
git add dashboard.py
git commit -m "feat(matrix): Eigen Explorer dashboard page"
```

---

## Self-Review Notes

**Spec coverage:**
- Direct solvers (elimination/inverse/Cramer) → Task 2. ✓
- Solution classifier (unique/none/infinite) → Task 2. ✓
- Iterative solvers (Jacobi/Gauss-Seidel/SOR) with history + diagonal-dominance + divergence demo → Task 3, presets Task 4, page Task 6. ✓
- Eigen-decomposition + power iteration → Task 7. ✓
- Plotly 3D: planes + intersection + convergence path + residual chart → Task 5; cube transform + eigenvectors → Task 8. ✓
- Editable grid + presets input → Tasks 6 & 9 (`st.data_editor`, presets). ✓
- `__main__` demos + separated pure/plot functions + unit tests → every task. ✓
- Phasing (Phase 1 = Tasks 1–6, Phase 2 = Tasks 7–9). ✓

**Type consistency:** `IterationResult` (Task 3) used unchanged by Task 6; `PowerIterationResult` (Task 7) used by Task 9; `classify_system` dict keys (`type/rank_A/rank_Ab/det/n`) consistent between Tasks 2, 4, 6; `system_from_editor` signature consistent between Tasks 6 and its callers; figure trace counts asserted in tests match builder code.

**Placeholder scan:** No TBD/TODO; all code blocks complete.
