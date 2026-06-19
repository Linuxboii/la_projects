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
