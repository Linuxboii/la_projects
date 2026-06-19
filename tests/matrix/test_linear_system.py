import numpy as np
import pytest

from matrix.linear_system import (
    SingularMatrixError,
    classify_system,
    solve_inverse,
    solve_cramer,
    solve_gaussian,
)

# Classic system with solution (2, 3, -1)
A_UNIQUE = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
B_UNIQUE = [8, -11, -3]
X_UNIQUE = np.array([2.0, 3.0, -1.0])

A_NONE = [[1, 1, 1], [1, 1, 1], [1, 2, 3]]
B_NONE = [1, 2, 4]

A_INF = [[1, 1, 1], [2, 2, 2], [1, 2, 3]]
B_INF = [6, 12, 14]

A_SINGULAR = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
B_SING = [1, 2, 3]


def test_classify_unique():
    info = classify_system(A_UNIQUE, B_UNIQUE)
    assert info["type"] == "unique"
    assert info["rank_A"] == 3 and info["rank_Ab"] == 3 and info["n"] == 3


def test_classify_none():
    assert classify_system(A_NONE, B_NONE)["type"] == "none"


def test_classify_infinite():
    assert classify_system(A_INF, B_INF)["type"] == "infinite"


def test_solve_inverse_matches_numpy():
    x = solve_inverse(A_UNIQUE, B_UNIQUE)
    assert np.allclose(x, X_UNIQUE)


def test_solve_cramer_matches_numpy():
    x, dets = solve_cramer(A_UNIQUE, B_UNIQUE)
    assert np.allclose(x, X_UNIQUE)
    assert len(dets["det_Ai"]) == 3


def test_solve_gaussian_matches_numpy_and_records_steps():
    x, steps = solve_gaussian(A_UNIQUE, B_UNIQUE)
    assert np.allclose(x, X_UNIQUE)
    assert len(steps) >= 2  # at least initial + final
    assert steps[0].shape == (3, 4)


def test_singular_raises():
    for solver in (solve_inverse, solve_cramer, solve_gaussian):
        with pytest.raises(SingularMatrixError):
            solver(A_SINGULAR, B_SING)
