"""Boundary, label-isolation, selection-fallback and incident regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from sentinel.detection import Detector
from sentinel.models import Forecaster, detect_period, noise_floor
from sentinel.pipeline import ROOT
from sentinel.validation import (chronological_folds, develop_series, pooled_by_group,
                                 prepare_development, select_methods)


def small_config():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config.update(windows=[8], alphas=[10.], horizon_sets=[[1], [1, 4]],
                  threshold_quantiles=[.99], adaptation=[False], persistence=[[1, 1]],
                  threshold_methods=["empirical"], incident_profiles=["guarded"],
                  max_seasonal_period=64, reference_size=32, min_calibration_rows=16)
    return config


class LeakageControls(unittest.TestCase):
    def test_fold_boundaries_keep_targets_before_test(self):
        config = small_config()
        folds, test_start = chronological_folds(1000, config)
        self.assertEqual(folds, [(200, 400, 550), (350, 550, 700), (500, 700, 850)])
        self.assertEqual(test_start, 850)
        self.assertTrue(all(t < c < v <= test_start for t, c, v in folds))

    def test_training_anomaly_filter_excludes_whole_input_target_span(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=300)
        normal = np.ones(300, bool)
        normal[80:90] = False
        observed = np.ones(300, bool)
        a = Forecaster("ridge", 12, (1, 4), 10).fit(values, observed, 200, normal)
        changed = values.copy()
        changed[80:90] += 100000
        b = Forecaster("ridge", 12, (1, 4), 10).fit(changed, observed, 200, normal)
        np.testing.assert_allclose(a.estimator[-1].coef_, b.estimator[-1].coef_)
        np.testing.assert_allclose(a.estimator[0].mean_, b.estimator[0].mean_)
        self.assertGreater(a.filtered_examples, 10)

    def test_seasonality_and_noise_floor_use_training_only(self):
        config = small_config()
        values = np.sin(np.arange(600)*2*np.pi/24)
        observed = np.ones(600, bool)
        model = Forecaster("ridge", 8, (1, 4), 10, use_seasonality=True)
        model.fit(values, observed, 240, config=config)
        changed = values.copy()
        changed[240:] = 1e9
        other = Forecaster("ridge", 8, (1, 4), 10, use_seasonality=True)
        other.fit(changed, observed, 240, config=config)
        self.assertEqual(model.period, 24)
        self.assertEqual(other.period, 24)
        np.testing.assert_allclose(model.estimator[-1].coef_, other.estimator[-1].coef_)
        np.testing.assert_allclose(model.predict_aligned(values)[:240],
                                   other.predict_aligned(changed)[:240], equal_nan=True)
        self.assertEqual(detect_period(np.ones(240), np.ones(240, bool)), 1)
        self.assertGreater(noise_floor(np.ones(240), np.ones(240, bool)), 0)

    def test_future_values_and_labels_cannot_change_cv_or_selection(self):
        config = small_config()
        rng = np.random.default_rng(42)
        values = np.sin(np.arange(600)/12) + rng.normal(0, .05, 600)
        stamps = pd.date_range("2026-01-01", periods=600, freq="min", tz="UTC")
        windows = [[str(stamps[330]), str(stamps[335])]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "stream.csv"
            pd.DataFrame({"timestamp": stamps, "value": values}).to_csv(path, index=False)
            a = prepare_development(path, "group/stream.csv", windows, config)
            # Guard against accidentally passing the complete stream to fitting.
            original_fit = Forecaster.fit
            def guarded_fit(model, fit_values, fit_observed, train_end, *args, **kwargs):
                self.assertEqual(len(fit_values), train_end)
                self.assertEqual(len(fit_observed), train_end)
                return original_fit(model, fit_values, fit_observed, train_end, *args, **kwargs)
            with patch.object(Forecaster, "fit", guarded_fit):
                develop_series(a, config, root / "a")
            values[510:] = 1e8
            pd.DataFrame({"timestamp": stamps, "value": values}).to_csv(path, index=False)
            windows += [[str(stamps[520]), str(stamps[590])]]
            b = prepare_development(path, "group/stream.csv", windows, config)
            develop_series(b, config, root / "b")
            self.assertEqual(len(a["values"]), 510)
            self.assertEqual(a["summaries"], b["summaries"])
            sa, wa = select_methods(a, {}, config)
            sb, wb = select_methods(b, {}, config)
            self.assertEqual(wa, wb)
            self.assertEqual({k:v["candidate"] for k,v in sa.items()},
                             {k:v["candidate"] for k,v in sb.items()})
            self.assertTrue(all(v["selection_source"] == "predeclared_preset_insufficient_events"
                                for v in sa.values()))

    def test_pooled_fallback_excludes_entire_target_source_group(self):
        config = small_config()
        params = {"kind":"mean", "family":"recent_mean", "window":3, "horizons":[1],
                  "alpha":0, "use_seasonality":False, "adaptive":False, "method":"empirical",
                  "quantile":.999, "incident_profile":"guarded", "persistence":[1,1]}
        metrics = {"detected_events":6, "false_alerts":1, "missed_events":0, "true_events":6,
                   "normal_points":10000, "event_fold_count":2, "event_fold_f1_sum":1.8,
                   "event_fold_f1_sq_sum":1.62}
        own = {"group":"A", "relative":"A/own.csv", "selection_events":0, "event_folds":0,
               "stamps":pd.Series(pd.to_datetime(["2026-01-01"], utc=True)),
               "summaries":{"x":{"parameters":params, "validation":metrics}}}
        peer = copy.deepcopy(own)
        peer["relative"] = "A/peer.csv"
        peer["summaries"]["x"]["validation"]["detected_events"] = 999
        other = copy.deepcopy(own)
        other["group"] = "B"
        other["relative"] = "B/other.csv"
        future = copy.deepcopy(other)
        future["relative"] = "B/future.csv"
        future["stamps"] = pd.Series(pd.to_datetime(["2027-01-01"], utc=True))
        future["summaries"]["x"]["validation"]["detected_events"] = 9999
        pools = pooled_by_group([own, peer, other, future])
        self.assertEqual(pools["A/own.csv"]["x"]["detected_events"], 6)
        selected, _ = select_methods(own, pools, config)
        self.assertEqual(selected["recent_mean"]["selection_source"], "other_source_groups_development_only")

    def test_incident_hysteresis_and_cooldown_do_not_use_labels(self):
        config = small_config()
        config.update(recovery_readings=2, cooldown_readings=2)
        detector = Detector.calibrate(np.zeros((64, 1)), .99, False, config, 1., [1, 1],
                                      "empirical", "guarded")
        # A brief dip must not create a second incident. Recovery closes it;
        # cooldown suppresses two new high readings, not an entire later event.
        decisions = [detector._decision(x)[0] for x in [3., .8, 3., 0., 0., 3., 3., 3.]]
        self.assertEqual(decisions, [True, True, True, True, False, False, False, True])
        self.assertGreaterEqual(detector.threshold, 1)

    def test_incident_replay_is_identical_across_chunks(self):
        config = small_config()
        errors = np.linspace(-1, 1, 64).reshape(-1, 1)
        detector = Detector.calibrate(errors, .99, True, config, .1, [1, 1], "empirical", "guarded")
        values = np.r_[np.zeros(5), np.full(5, 10.), np.zeros(15)]
        predictions, observed = np.zeros((25, 1)), np.ones(25, bool)
        full = copy.deepcopy(detector).replay(values, predictions, observed, 0, 25)
        first = detector.replay(values, predictions, observed, 0, 8)
        rest = detector.replay(values, predictions, observed, 8, 25)
        np.testing.assert_array_equal(full["flag"], np.r_[first["flag"], rest["flag"]])
        np.testing.assert_allclose(full["score"], np.r_[first["score"], rest["score"]])


if __name__ == "__main__":
    unittest.main()
