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
