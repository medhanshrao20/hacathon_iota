"""Small synthetic end-to-end regression; never starts the full NAB search."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from sentinel.pipeline import ROOT, search_series


class SyntheticPipeline(unittest.TestCase):
    def test_four_methods_finish_and_saved_model_can_predict(self):
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
                             {"recent_mean", "rolling_median_mad", "ridge_single", "ridge_multi"})
            self.assertEqual(sum(r["selected_on_validation"] for r in records), 1)
            self.assertTrue(all(r["eligible_points"] == 90 for r in records))
            self.assertTrue((output / "chart.html").is_file())
            self.assertTrue((output / "validation_chart.html").is_file())
            bundle = joblib.load(output / "model.joblib")
            predictions = bundle["model"].predict_aligned(values)
            self.assertEqual(predictions.shape, (600, len(bundle["model"].horizons)))
            self.assertTrue(np.isfinite(predictions[-1]).all())
            trials = pd.read_csv(output / "experiment_log.csv")
            self.assertEqual(len(trials), 8)


if __name__ == "__main__":
    unittest.main()
