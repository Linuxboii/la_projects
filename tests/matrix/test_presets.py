import numpy as np

from matrix.presets import PRESETS, list_presets, get_preset
from matrix.linear_system import classify_system
from matrix.iterative_solvers import is_diagonally_dominant


def test_list_presets_nonempty():
    names = list_presets()
    assert len(names) >= 5
    assert "Unique solution" in names


def test_get_preset_shapes():
    A, b = get_preset("Unique solution")
    assert A.shape == (3, 3)
    assert b.shape == (3,)


def test_solution_type_presets_classify_correctly():
    assert classify_system(*get_preset("Unique solution"))["type"] == "unique"
    assert classify_system(*get_preset("No solution"))["type"] == "none"
    assert classify_system(*get_preset("Infinite solutions"))["type"] == "infinite"


def test_convergence_presets_have_expected_dominance():
    assert is_diagonally_dominant(get_preset("Diagonally dominant (converges)")[0])
    assert not is_diagonally_dominant(get_preset("Not dominant (diverges)")[0])
