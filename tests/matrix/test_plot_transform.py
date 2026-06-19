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


def test_transform_figure_skips_complex_eigenvectors():
    # A 2D rotation block embedded in 3D: eigenvalues are ±i (complex) and 2 (real).
    A_rot = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    fig = transform_figure(A_rot, show_eigen=True)
    # 2 cube traces + only the single REAL eigenvector arrow (complex pair skipped)
    assert len(fig.data) == 3
