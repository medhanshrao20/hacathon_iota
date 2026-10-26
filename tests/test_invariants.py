"""Manual checks: python -m unittest discover -s tests -v

These use tiny synthetic arrays only, not downloaded competition data.
They are deliberately not called by run.py.
"""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from sentinel.data import event_ids, load_series, split_points
from sentinel.detection import Detector
from sentinel.evaluation import evaluate
from sentinel.models import Forecaster, baseline_scales


class Invariants(unittest.TestCase):
    def test_single_horizon_ridge_keeps_forecast_column(self):
        values = np.sin(np.arange(160) / 8.)
        observed = np.ones(len(values), dtype=bool)
        model = Forecaster("ridge", 12, (1,), 10).fit(values, observed, 80)
        prediction = model.predict_aligned(values)
        self.assertEqual(prediction.shape, (160, 1))
        self.assertTrue(np.isfinite(prediction[12:]).all())
        expected = np.asarray(model.estimator.predict(values[88:100].reshape(1, -1))).item()
        self.assertAlmostEqual(prediction[100, 0], expected)

    def test_timestamp_cleanup_preserves_first_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("timestamp,value\n2026-01-01 00:01:00,2\n"
                            "2026-01-01 00:00:00,1\n2026-01-01 00:01:00,999\n",
                            encoding="utf-8")
            stamps, values, observed, audit = load_series(path)
            np.testing.assert_array_equal(values, [1., 2.])
            self.assertEqual(audit["duplicate_rows_removed"], 1)
            self.assertTrue(audit["timestamps_reordered"])

    def test_mean_alignment_excludes_current_value(self):
        values = np.array([1., 2., 3., 100., 5.])
        prediction = Forecaster("mean", 3, (1,)).predict_aligned(values)
        self.assertEqual(prediction[3, 0], 2.)
        values[3] = -1000
        self.assertEqual(Forecaster("mean", 3, (1,)).predict_aligned(values)[3, 0], 2.)

    def test_ridge_causality_and_training_boundary(self):
        rng = np.random.default_rng(5)
        values = rng.normal(size=240).cumsum()
        observed = np.ones(len(values), dtype=bool)
        a = Forecaster("ridge", 12, (1, 4, 8), 10).fit(values, observed, 100)
        changed = values.copy()
        changed[100:] += 1000
        b = Forecaster("ridge", 12, (1, 4, 8), 10).fit(changed, observed, 100)
        np.testing.assert_allclose(a.estimator[-1].coef_, b.estimator[-1].coef_)
        original = a.predict_aligned(values)
        changed = values.copy()
        changed[170:] += 1000
        updated = a.predict_aligned(changed)
        np.testing.assert_allclose(original[:170], updated[:170], equal_nan=True)
        for column, horizon in enumerate(a.horizons):
            target = 160
            source = values[target-horizon-11:target-horizon+1]
            expected = a.estimator.predict(source.reshape(1, -1))[0, column]
            self.assertAlmostEqual(original[target, column], expected)

    def test_mad_scale_excludes_current(self):
        values = np.array([1., 2., 3., 4., 5., 6.])
        original = baseline_scales(values, np.ones(6, bool), 3)
        values[3:] = 10000
        updated = baseline_scales(values, np.ones(6, bool), 3)
        self.assertEqual(original[3, 0], updated[3, 0])

    def test_score_before_update_and_chunk_equivalence(self):
        config = {"reference_size": 32, "reference_update_every": 1, "min_calibration_rows": 16}
        errors = np.linspace(-1, 1, 64).reshape(-1, 1)
        detector = Detector.calibrate(errors, .99, True, config, 1e-6, [1, 1])
        initial_center = detector.center.copy()
        values = np.array([100., 0., .2, -.1, .1])
        predictions = np.zeros((len(values), 1))
        observed = np.ones(len(values), bool)
        whole = copy.deepcopy(detector).replay(values, predictions, observed, 0, 5)
        sequential = copy.deepcopy(detector)
        first = sequential.replay(values, predictions, observed, 0, 1)
        np.testing.assert_allclose(sequential.center, initial_center)
        second = sequential.replay(values, predictions, observed, 1, 5)
        np.testing.assert_allclose(whole["score"], np.r_[first["score"], second["score"]])
        self.assertTrue(first["flag"][0])

    def test_alerts_not_point_adjusted_or_duplicated(self):
        stamps = pd.Series(pd.date_range("2026-01-01", periods=10, freq="min", tz="UTC"))
        ids = np.array([-1, -1, 0, 0, 0, -1, -1, 1, 1, -1])
        flags = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0], bool)
        result = evaluate(stamps, ids, flags, np.ones(10, bool))
        self.assertEqual(result["detected_events"], 2)
        self.assertEqual(result["false_alerts"], 1)
        self.assertEqual(result["mean_delay_seconds"], 0)
        # A permanent alarm beginning before both events does not detect both.
        result = evaluate(stamps, ids, np.ones(10, bool), np.ones(10, bool))
        self.assertEqual(result["detected_events"], 0)
        self.assertEqual(result["false_alerts"], 1)

    def test_split_and_labels(self):
        self.assertEqual(split_points(1000, [.5, .2, .15, .15]), [500, 700, 850])
        with self.assertRaises(ValueError):
            split_points(100, [.5, .2, .15, .15],
                         {"train_end": 80, "calibration_end": 70, "validation_end": 90})
        stamps = pd.Series(pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC"))
        ids = event_ids(stamps, [[str(stamps[1]), str(stamps[2])]])
        np.testing.assert_array_equal(ids, [-1, 0, 0, -1])


if __name__ == "__main__":
    unittest.main()
