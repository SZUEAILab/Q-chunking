import csv

from analyze_learned_shell_formal import bootstrap_interval, load_per_run


def test_bootstrap_interval_is_ordered_for_positive_differences():
    import numpy as np

    low, high = bootstrap_interval(
        np.asarray([1.0, 2.0, 3.0]), np.random.default_rng(7), count=1000
    )

    assert 0 < low <= high


def test_formal_curve_loader_uses_named_success_column(tmp_path):
    path = tmp_path / "curve.csv"
    rows = []
    for condition in __import__("train_synthetic_shell_rl").CONDITIONS:
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

    loaded = load_per_run(path)

    assert len(loaded) == 7 * 32
    assert all(row["auc"] == 0.5 for row in loaded)
