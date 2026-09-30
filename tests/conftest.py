"""Deterministic fixtures for research protocol regression tests."""

import os

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsbm-pytest-mpl")


@pytest.fixture
def parameter_row():
    """Return a feasible canonical structural configuration."""
    return {
        "stratum": "assortative",
        "K": 3,
        "p0": 0.3,
        "centre_u": 5.0,
        "contrast_u": 3.0,
        "sigma_ratio": 2.0,
        "rho": 2.0,
        "size_alpha": np.inf,
        "f": 0.5,
        "core_fraction": 0.2,
        "seed": 42,
        "csv_idx": 7,
        "frac_negative": 0.1,
    }
