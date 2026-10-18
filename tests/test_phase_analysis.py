import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.metrics import finite_mean


def test_finite_mean_accepts_generator_and_ignores_nonfinite_values():
    values = (value for value in [1.0, np.nan, 3.0, np.inf])

    assert finite_mean(values) == 2.0


def test_finite_mean_returns_nan_without_finite_observations():
    assert np.isnan(finite_mean([]))
    assert np.isnan(finite_mean([np.inf, np.nan]))
