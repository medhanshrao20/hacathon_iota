"""Small synthetic end-to-end regression; never starts the full NAB search."""
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from sentinel.pipeline import ROOT, search_series


class SyntheticPipeline(unittest.TestCase):
    def test_cli_freezes_all_selections_before_test_and_predicts_new_data(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        config.update(windows=[8], alphas=[10.], horizon_sets=[[1], [1, 4]],
                      threshold_quantiles=[.99], adaptation=[False], persistence=[[1, 1]],
                      threshold_methods=["empirical"], incident_profiles=["guarded"],
                      max_seasonal_period=64, min_calibration_rows=16, reference_size=32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            stamps = pd.date_range("2026-01-01", periods=600, freq="min", tz="UTC")
            labels = {}
            for name in ("one", "two"):
                folder = data / name
                folder.mkdir()
                values = np.sin(np.arange(600)/12.)
                pd.DataFrame({"timestamp":stamps, "value":values}).to_csv(folder/"stream.csv", index=False)
                labels[name+"/stream.csv"] = [[str(stamps[330]), str(stamps[335])]]
            label_file = root / "labels.json"
            label_file.write_text(json.dumps(labels), encoding="utf-8")
            config.update(data_dir=str(data), labels_file=str(label_file), output_dir=str(root/"results"))
            config_file = root / "config.json"
            config_file.write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT/"run.py"), "--config", str(config_file)],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            output = next((root/"results").glob("experiment_v3_*"))
            frozen = output / "selection_frozen_before_test.json"
            self.assertEqual(len(json.loads(frozen.read_text())), 2)
            comparisons = pd.read_csv(output/"comparison.csv")
            self.assertEqual(len(comparisons), 14)
            self.assertEqual(json.loads((output/"failures.json").read_text()), [])
            audit = subprocess.run([sys.executable, str(ROOT/"run.py"), "--config", str(config_file),
                                    "--selection-audit", str(output)], capture_output=True, text=True, timeout=120)
            self.assertEqual(audit.returncode, 0, audit.stdout+audit.stderr)
            self.assertTrue(next((root/"results").glob("selection_audit_*/selection_policy_development.json")).is_file())
            self.assertTrue(all(f.stat().st_mtime >= frozen.stat().st_mtime for f in output.glob("series/*/predictions.csv")))
            artifact = next(output.glob("series/*/model.joblib"))
            new = root / "new.csv"
            pd.DataFrame({"timestamp":pd.date_range(stamps[-1]+pd.Timedelta(minutes=1), periods=10, freq="min"),
                          "value":np.zeros(10)}).to_csv(new, index=False)
            result = subprocess.run([sys.executable, str(ROOT/"run.py"), "--config", str(config_file),
                                     "--predict", str(new), "--artifact", str(artifact)],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            predicted = next((root/"results").glob("predict_*/predictions.csv"))
            self.assertEqual(len(pd.read_csv(predicted)), 10)

    def test_seven_methods_finish_and_saved_model_can_predict(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        config.update(windows=[8], alphas=[1.0], horizon_sets=[[1], [1, 4]],
                      threshold_quantiles=[.99], adaptation=[False, True],
                      persistence=[[1, 1]], min_calibration_rows=16,
                      reference_size=32, plot_max_points=600)
        rng = np.random.default_rng(42)
        values = np.sin(np.arange(600) / 12.) + rng.normal(0, .05, 600)
        values[460:465] += 5
        values[550:555] += 5
        stamps = pd.date_range("2026-01-01", periods=600, freq="min", tz="UTC")
        windows = [[str(stamps[460]), str(stamps[464])],
                   [str(stamps[550]), str(stamps[554])]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic.csv"
            pd.DataFrame({"timestamp": stamps, "value": values}).to_csv(path, index=False)
            output = root / "result"
            records = search_series(path, path.name, windows, config, output)
            self.assertEqual({r["family"] for r in records},
                             {"recent_mean", "seasonal_naive", "rolling_median_mad", "ridge_single", "ridge_multi", "boosted_forecast", "pattern_hybrid"})
            self.assertEqual(sum(r["selected_on_validation"] for r in records), 1)
            self.assertTrue(all(r["eligible_points"] == 90 for r in records))
            self.assertTrue((output / "chart.html").is_file())
            self.assertTrue((output / "validation_chart.html").is_file())
            self.assertTrue(pd.read_csv(output / "predictions.csv").incident_profile.isin(["plain", "guarded"]).all())
            bundle = joblib.load(output / "model.joblib")
            predictions = bundle["model"].predict_aligned(values)
            self.assertEqual(predictions.shape, (600, len(bundle["model"].horizons)))
            self.assertTrue(np.isfinite(predictions[-1]).all())
            trials = pd.read_csv(output / "experiment_log.csv")
            self.assertEqual(len(trials), 168)
            self.assertTrue(trials.groupby("candidate").fold.nunique().eq(3).all())
            self.assertEqual(bundle["format_version"], 3)


if __name__ == "__main__":
    unittest.main()
