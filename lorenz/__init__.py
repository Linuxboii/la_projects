"""
lorenz — Lorenz System Analysis Package
==========================================

Modules
-------
lorenz_solver       Numerical integration (solve_ivp) and fixed-point analysis.
lorenz_3d_plot      3D rendering of the Lorenz strange attractor.
sensitivity_plot    Butterfly Effect — sensitivity to initial conditions.

References
----------
- Lorenz, E. N. (1963). *J. Atmos. Sci.*, 20(2), 130-141.
- Sparrow, C. (1982). *The Lorenz Equations*. Springer.
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*, 2nd ed.
"""

from . import lorenz_solver
from . import lorenz_3d_plot
from . import sensitivity_plot
