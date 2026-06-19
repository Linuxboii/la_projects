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
