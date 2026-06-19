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
