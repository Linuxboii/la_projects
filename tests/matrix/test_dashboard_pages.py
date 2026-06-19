import numpy as np
import pandas as pd

from matrix.dashboard_helpers import system_from_editor


def test_system_from_editor_splits_A_and_b():
    df = pd.DataFrame(
        [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]],
        columns=["x", "y", "z", "= b"],
    )
    A, b = system_from_editor(df)
    assert A.shape == (3, 3)
    assert b.shape == (3,)
    assert np.allclose(b, [8, -11, -3])
