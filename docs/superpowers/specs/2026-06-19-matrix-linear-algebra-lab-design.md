# Matrix — Linear Algebra Lab (Design Spec)

**Date:** 2026-06-19
**Status:** Approved, pending implementation plan
**Module:** `matrix/` inside `LA_project`

## 1. Purpose

Add a `matrix/` module to the existing `LA_project` (which already contains
`lorenz/` and `logistic/` chaos-theory modules). The matrix module is the
linear-algebra core of the project: it solves systems of linear equations and
performs eigen-analysis on 3×3 matrices, and it visualises everything in
**interactive 3D** inside the existing Streamlit dashboard.

The defining requirements from the user:

- A pure-Python math project about **matrices and their solutions**.
- **3D visuals.**
- Deals with **equations and finding solutions**.
- **Iterate at least 5 times per equation** to show "all sorts of solutions".

The interpretation (confirmed with the user):

- **Both** linear systems (`Ax = b`) **and** eigenvalues/eigenvectors.
- Show **both** exact/direct methods **and** iterative convergence (the
  iterative methods literally step ≥5 times toward the answer).
- **Interactive Plotly** 3D (drag-rotate/zoom) in the dashboard.
- Input via an **editable coefficient grid + curated presets**.

## 2. Existing patterns to follow

The `lorenz/` and `logistic/` modules establish the house style, which this
module matches:

- Heavily documented modules (module docstring with mathematical background +
  references; function docstrings with NumPy-style sections and LaTeX math).
- Pure computation functions with type hints, kept **separate** from plotting,
  so they are unit-testable.
- `plot_*` functions that build and return a figure object.
- A `__main__` CLI demo block printing a summary (see `lorenz_solver.py`).
- Pages wired into `dashboard.py` via the sidebar `selectbox`.

Deviation from existing modules: the existing `plot_*` functions return
**matplotlib** figures; this module returns **Plotly** figures (rendered with
`st.plotly_chart`) because interactive rotatable 3D is required. `plotly` is
already in `requirements.txt`.

## 3. Module layout

```
matrix/
├── __init__.py
├── linear_system.py      # Core: A & b, classify solution type, direct solvers
├── iterative_solvers.py  # Jacobi / Gauss-Seidel / SOR — return iteration history
├── eigen.py              # Eigenvalues/vectors + power iteration (with history)
├── plot_planes.py        # Plotly 3D: planes + intersection + convergence path
├── plot_transform.py     # Plotly 3D: matrix warping a unit cube + eigenvector arrows
└── presets.py            # Curated example systems
```

Dashboard gains a "Linear Algebra" group with three pages: **System Solver**,
**Iterative Convergence**, **Eigen Explorer**.

## 4. Math specification

### 4.1 `linear_system.py` (3×3 system `Ax = b`)

Direct solvers (all exact, for comparison):

- **Gaussian elimination** with partial pivoting.
- **Inverse method**: `x = A⁻¹ b` (only when `A` is invertible).
- **Cramer's rule**: `xᵢ = det(Aᵢ) / det(A)`, where `Aᵢ` is `A` with column
  `i` replaced by `b`.

Solution-type classifier (compare ranks):

- `rank(A) == rank([A|b]) == n` → **unique** (planes meet at a point).
- `rank(A) == rank([A|b]) <  n` → **infinitely many** (planes share a line).
- `rank(A) <  rank([A|b])`      → **no solution** (inconsistent / parallel).

Functions (indicative signatures):

- `solve_gaussian(A, b) -> (x, steps)`
- `solve_inverse(A, b) -> x`
- `solve_cramer(A, b) -> (x, determinants)`
- `classify_system(A, b) -> {"type", "rank_A", "rank_Ab", "det"}`

Errors: singular `A` raises a clear, caught exception surfaced in the UI
(never an uncaught traceback). Each direct solver guards against singular `A`.

### 4.2 `iterative_solvers.py` (the "iterate ≥5 times" centrepiece)

Stationary iterative methods, each returning the **full history** of iterates
and residuals:

- **Jacobi**: `x⁽ᵏ⁺¹⁾ = D⁻¹ (b − (L+U) x⁽ᵏ⁾)`.
- **Gauss–Seidel**: uses already-updated components within an iteration.
- **SOR**: Gauss–Seidel with relaxation factor `ω` (`0 < ω < 2`).

Each runs at least 5 iterations (user-settable, default e.g. 25, minimum 5),
or until `‖A x⁽ᵏ⁾ − b‖ < tol`. Returns:

- `history`: array of every iterate `x⁽⁰⁾ … x⁽ᵏ⁾`.
- `residuals`: `‖A x⁽ᵏ⁾ − b‖₂` at each step.
- `converged: bool`, `iterations: int`.

Convergence requires diagonal dominance; the UI reports the diagonal-dominance
check, and presets include both a **convergent** and a **divergent** system so
users observe both behaviours rather than only success.

Signature: `jacobi(A, b, x0=None, max_iter=25, tol=1e-8) -> IterationResult`
(same shape for `gauss_seidel` and `sor` with an extra `omega`).

### 4.3 `eigen.py`

- **Exact**: `numpy.linalg.eig(A)` → eigenvalues + eigenvectors.
- **Power iteration** with history: repeatedly `v ← A v / ‖A v‖`, converging to
  the dominant eigenvector; returns the per-iteration vector estimates and the
  Rayleigh-quotient eigenvalue estimate (another visibly-iterating method).

Signatures: `eig_decompose(A)`, `power_iteration(A, max_iter=25, tol=1e-9) ->
IterationResult`.

## 5. Visualisation specification (interactive Plotly)

### 5.1 System Solver page (`plot_planes.py`)

- Each equation `aᵢx + bᵢy + cᵢz = dᵢ` rendered as a translucent plane
  (Plotly `Surface`/`Mesh3d`) over a sensible bounding box.
- The intersection point (when unique) marked with a labelled `Scatter3d`.
- Updates live as the coefficient grid is edited; the solution-type badge and
  the three direct-method results (elimination / inverse / Cramer) shown
  alongside.

### 5.2 Iterative Convergence page (`plot_planes.py`)

- Same three planes, plus the sequence of guesses
  `x⁽⁰⁾ → x⁽¹⁾ → … → x⁽ᵏ⁾` drawn as a connected path of markers walking toward
  the true intersection.
- A residual-vs-iteration line chart beside the 3D view.
- A per-iteration table (k, x, y, z, residual).
- Controls: method (Jacobi/Gauss–Seidel/SOR), `ω` for SOR, iteration count,
  initial guess.

### 5.3 Eigen Explorer page (`plot_transform.py`)

- A unit cube / grid wireframe and its image under `A` (showing rotation,
  scaling, shear).
- Eigenvectors drawn as arrows that keep their direction under the transform
  (scaled by their eigenvalue), making "eigen" visually obvious.
- Optionally overlay the power-iteration estimates converging onto the
  dominant eigenvector.

### 5.4 Input

Every page exposes an editable 3×3 coefficient grid (and 3×1 RHS for systems)
plus a preset dropdown. Presets (`presets.py`) include at minimum: a unique
solution, a no-solution case, an infinite-solution case, a diagonally-dominant
(convergent) system, and a non-dominant (divergent) system.

## 6. Testing & CLI

- Pure functions are separated from plotting and covered by unit tests:
  direct solvers agree with `numpy.linalg.solve` on well-posed systems; the
  classifier returns the correct type for each preset; iterative methods
  converge on the dominant-diagonal preset and the residual decreases
  monotonically there; power iteration matches `numpy.linalg.eig`'s dominant
  eigenpair.
- Each core module (`linear_system`, `iterative_solvers`, `eigen`) has a
  `__main__` demo printing a summary, mirroring `lorenz_solver.py`.

## 7. Phasing

- **Phase 1**: `linear_system.py`, `iterative_solvers.py`, `presets.py`,
  `plot_planes.py`, and the System Solver + Iterative Convergence pages.
- **Phase 2**: `eigen.py`, `plot_transform.py`, and the Eigen Explorer page.

Designed as one coherent module; built in two phases.

## 8. Out of scope (YAGNI)

- Matrices larger than 3×3 (3D visualisation is the point; the math
  generalises but the UI targets 3×3).
- Sparse/large-scale solvers, preconditioning beyond SOR relaxation.
- Saving/loading user sessions; PDF export.
