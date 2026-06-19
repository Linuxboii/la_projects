r"""dashboard_helpers.py — small pure helpers shared by the Streamlit pages."""

from typing import Tuple

import numpy as np
import pandas as pd


def system_from_editor(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Split an edited 3×4 coefficient table into ``A`` (3×3) and ``b`` (3,)."""
    values = df.to_numpy(dtype=float)
    A = values[:, :3]
    b = values[:, 3]
    return A, b
