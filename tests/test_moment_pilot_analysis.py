import csv
import json
import tempfile
import unittest
from pathlib import Path

from analyze_robot_moment_control_task1_pilot import load_complete_runs


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class MomentPilotAnalysisTest(unittest.TestCase):
    def test_noise_only_gate_ignores_radius_mean_error(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            run_group = "noise-only-test"
            (run / "flags.json").write_text(
                json.dumps(
                    {
                        "run_group": run_group,
                        "seed": 64,
                        "ds_mode": "integrated",
                        "agent": {
                            "ds_moment_controlled": True,
                            "ds_moment_control_speed_mean": False,
                            "ds_moment_target_radius_std": 0.025,
                            "ds_moment_target_angle": 0.169,
                        },
                    }
                )
            )
            steps = range(50_000, 300_001, 50_000)
            write_csv(
                run / "eval.csv",
                [
                    {"step": step, "episode.return": -1000.0, "success": 0.5}
                    for step in steps
                ],
            )
            write_csv(run / "online_agent.csv", [{"step": 5000, "loss": 1.0}])
            write_csv(
                run / "moment_controller.csv",
                [
                    {
                        "step": step,
                        "angle_abs_error": 0.01,
                        "radius_mean_abs_error": 10.0,
                        "radius_std_abs_error": 0.005,
                    }
                    for step in steps
                ],
            )

            loaded = load_complete_runs(Path(directory), run_group=run_group)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["manipulation_pass_count"], 6)
        self.assertTrue(loaded[0]["manipulation_gate"])
        self.assertEqual(loaded[0].get("radius_std_target"), 0.025)
        self.assertEqual(loaded[0].get("angle_target"), 0.169)


if __name__ == "__main__":
    unittest.main()
