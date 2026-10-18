import csv

import pytest

from research.metrics import bootstrap_interval, load_shell_runs
from research.shell import CONDITIONS


def test_bootstrap_interval_is_ordered_for_positive_differences():
    import numpy as np

    low, high = bootstrap_interval(
        np.asarray([1.0, 2.0, 3.0]), np.random.default_rng(7), count=1000
    )

    assert 0 < low <= high


def test_formal_curve_loader_uses_named_success_column(tmp_path):
    path = tmp_path / "curve.csv"
    rows = []
    for condition in CONDITIONS:
        for seed in range(100, 132):
            rows.extend(
                [
                    {
                        "condition": condition.name,
                        "seed": seed,
                        "iteration": 0,
                        "stage0_interactions": 0,
                        "discovery": 0.0,
                        "precision": 0.0,
                        "success": 0.0,
                    },
                    {
                        "condition": condition.name,
                        "seed": seed,
                        "iteration": 1,
                        "stage0_interactions": 10,
                        "discovery": 1.0,
                        "precision": 1.0,
                        "success": 1.0,
                    },
                ]
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_shell_runs(path, conditions=CONDITIONS, expected_seeds=range(100, 132))

    assert len(loaded) == 7 * 32
    assert all(row["auc"] == 0.5 for row in loaded)


def test_shell_loader_rejects_missing_registered_seed(tmp_path):
    path = tmp_path / "curve.csv"
    path.write_text("condition,seed,stage0_interactions,success,discovery,precision\n")

    with pytest.raises(ValueError, match="Expected.*seeds"):
        load_shell_runs(path, conditions=CONDITIONS, expected_seeds=(71746,))
