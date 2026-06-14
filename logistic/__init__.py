"""
logistic — Logistic Map Analysis Package
==========================================

Modules
-------
logistic_map    Core recurrence relation, orbit generation, fixed-point analysis.
bifurcation     Bifurcation diagram — period-doubling route to chaos.
lyapunov        Lyapunov exponent spectrum — quantitative chaos measure.
time_series     Iteration-by-iteration time series for selected r values.

References
----------
- May, R. M. (1976). *Nature*, 261, 459-467.
- Feigenbaum, M. J. (1978). *J. Stat. Phys.*, 19, 25-52.
- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*, 2nd ed.
"""

from . import logistic_map
from . import bifurcation
from . import lyapunov
from . import time_series
