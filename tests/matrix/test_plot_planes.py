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
