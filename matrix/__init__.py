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

from . import linear_system  # noqa: E402,F401
from . import iterative_solvers  # noqa: E402,F401
from . import presets  # noqa: E402,F401
from . import plot_planes  # noqa: E402,F401
