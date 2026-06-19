r"""
presets.py — Curated 3×3 example systems
========================================

A small gallery of systems chosen to illustrate each behaviour: the three
solution types (unique / none / infinite) and the two convergence regimes
(diagonally dominant → iterative methods converge; strongly off-diagonal →
they diverge).
"""

from typing import Dict, List, Tuple

import numpy as np

PRESETS: Dict[str, Dict[str, object]] = {
    "Unique solution": {
        "A": [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],
        "b": [8, -11, -3],
        "note": "Three planes meet at the single point (2, 3, -1).",
    },
    "No solution": {
        "A": [[1, 1, 1], [1, 1, 1], [1, 2, 3]],
        "b": [1, 2, 4],
        "note": "Two parallel, inconsistent planes — the system is unsolvable.",
    },
    "Infinite solutions": {
        "A": [[1, 1, 1], [2, 2, 2], [1, 2, 3]],
        "b": [6, 12, 14],
        "note": "One equation is a multiple of another; planes share a line.",
    },
    "Diagonally dominant (converges)": {
        "A": [[4, 1, 1], [1, 5, 2], [1, 2, 6]],
        "b": [6, 8, 9],
        "note": "Diagonally dominant — Jacobi/Gauss-Seidel/SOR all converge.",
    },
    "Not dominant (diverges)": {
        "A": [[1, 2, 2], [2, 1, 2], [2, 2, 1]],
        "b": [1, 1, 1],
        "note": "Off-diagonal dominates — Jacobi diverges away from the answer.",
    },
}


def list_presets() -> List[str]:
    return list(PRESETS.keys())


def get_preset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    entry = PRESETS[name]
    A = np.asarray(entry["A"], dtype=float)
    b = np.asarray(entry["b"], dtype=float).reshape(-1)
    return A, b


if __name__ == "__main__":
    for name in list_presets():
        A, b = get_preset(name)
        print(f"{name}: {PRESETS[name]['note']}")
