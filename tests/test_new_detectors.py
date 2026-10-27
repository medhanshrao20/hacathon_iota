"""Independent sequential oracle and causal checks for v3 score channels."""
import copy
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from sentinel.detection import Detector, robust_reference
from sentinel.models import Forecaster
from sentinel.reporting import prediction_frame
from sentinel.validation import aggregate, select_methods


class NewDetectors(unittest.TestCase):
    def test_sparse_transfer_checks_local_false_alerts(self):
        def metrics(tp, fp, fn):
            return aggregate([dict(detected_events=tp, false_alerts=fp, missed_events=fn,
                                   normal_points=1000, true_events=tp+fn,
                                   event_f1=2*tp/max(1, 2*tp+fp+fn))]*2)
        params = dict(kind="mean", family="recent_mean", window=3, horizons=[1], alpha=0)
        stream = dict(relative="target.csv", selection_events=0, event_folds=0,
                      summaries={
                          "noisy":dict(parameters=params, validation=metrics(0, 20, 0)),
                          "quiet":dict(parameters=params, validation=metrics(0, 0, 0))})
        pools = {"target.csv":{"noisy":metrics(6, 0, 0), "quiet":metrics(5, 1, 1)}}
        config = dict(min_selection_events=5, min_event_folds=2,
                      false_alert_budget_per_1000=1, selection_prior_events=5,
                      local_false_alert_penalty=.05)
        selected, winner = select_methods(stream, pools, config)
        self.assertEqual(winner, "recent_mean")
        self.assertEqual(selected[winner]["candidate"], "quiet")
        stream["selection_events"], stream["event_folds"] = 2, 1
        selected, _ = select_methods(stream, pools, config)
        self.assertAlmostEqual(selected[winner]["local_evidence_weight"], 2/7)
        self.assertEqual(selected[winner]["selection_source"], "shrunk_own_cv_with_historical_peer_prior")

    def test_vectorized_replay_matches_scalar_reference(self):
        rng = np.random.default_rng(71)
        calibration = rng.normal(size=(100, 3))
        config = dict(min_calibration_rows=16, reference_size=40, reference_update_every=7)
        values = rng.normal(size=250)
        values[70:90] += 8
        predictions = np.zeros((250, 2))
        scales = rng.uniform(.5, 2, (250, 2))
        auxiliary = np.abs(rng.normal(size=(250, 1)))
        observed = np.ones(250, bool)
        observed[[3, 44, 110]] = False
        errors = np.column_stack([values[:, None]/scales, auxiliary])
        for adaptive in (False, True):
            for profile in ("plain", "guarded"):
                detector = Detector.calibrate(calibration, .99, adaptive, config, .01,
                                              (2, 3), incident_profile=profile,
                                              upper_only=[False, False, True])
                oracle = copy.deepcopy(detector)
                expected, flags, centers = [], [], []
                for i, error in enumerate(errors):
                    if not observed[i]:
                        oracle.votes.clear()
                        expected.append(np.nan)
                        flags.append(False)
                        centers.append([np.nan, np.nan])
                        continue
                    deviation = error - oracle.center
                    score = np.max(np.where(oracle.upper_only, np.maximum(deviation, 0),
                                            np.abs(deviation))/oracle.scale)
                    expected.append(score)
                    centers.append(predictions[i]+oracle.center[:2]*scales[i])
                    flag, raw = oracle._decision(score)
                    flags.append(flag)
                    if adaptive and not flag and not raw:
                        oracle.history.append(error.copy())
                        oracle.accepted_since_update += 1
                        if oracle.accepted_since_update >= oracle.update_every:
                            oracle.center, scale = robust_reference(np.asarray(oracle.history), oracle.floor,
                                                                     oracle.method, oracle.quantile)
                            oracle.scale = np.maximum(scale, oracle.initial_scale*oracle.min_scale_fraction)
                            oracle.accepted_since_update = 0
                actual = detector.replay(values, predictions, observed, 0, len(values), scales, auxiliary)
                np.testing.assert_allclose(actual["score"], expected, equal_nan=True)
                np.testing.assert_allclose(actual["center"], centers, equal_nan=True)
                np.testing.assert_array_equal(actual["flag"], flags)
                np.testing.assert_allclose(detector.center, oracle.center)
                np.testing.assert_allclose(detector.scale, oracle.scale)

    def test_new_models_ignore_future_and_known_training_anomalies(self):
        values = np.sin(np.arange(500)*2*np.pi/24)
        observed = np.ones(500, bool)
        normal = np.ones(500, bool)
        normal[70:80] = False
        changed = values.copy()
        changed[70:80] = 1e6
        changed[350:] = -1e6
        config = dict(boosting_iterations=15, pattern_reference_size=64)
        with threadpool_limits(limits=2):
            for kind in ("boosted", "pattern"):
                a = Forecaster(kind, 24, (1,), 10).fit(values, observed, 250, normal, config)
                b = Forecaster(kind, 24, (1,), 10).fit(changed, observed, 250, normal, config)
                np.testing.assert_allclose(a.predict_aligned(values), b.predict_aligned(values), equal_nan=True)
                # Current-window novelty may use the current observed value; forecasts cannot.
                later = values.copy()
                later[350:] = 1e6
                np.testing.assert_allclose(a.predict_aligned(values)[:351], a.predict_aligned(later)[:351], equal_nan=True)
                if kind == "pattern":
                    np.testing.assert_allclose(a.auxiliary_scores(values), b.auxiliary_scores(values), equal_nan=True)
                    np.testing.assert_allclose(a.auxiliary_scores(values)[:350],
                                               a.auxiliary_scores(later)[:350], equal_nan=True)
                    flat = values.copy()
                    flat[350:] = 0
                    self.assertGreater(a.auxiliary_scores(flat)[400:, 1].mean(),
                                       a.auxiliary_scores(values)[250:340, 1].mean()+.1)
                else:
                    self.assertFalse(a.estimator[-1].early_stopping)

    def test_novelty_evidence_and_upper_only_score(self):
        values = np.sin(np.arange(200)*2*np.pi/24)
        model = Forecaster("pattern", 24, (1,)).fit(values, np.ones(200, bool), 100)
        predictions = model.predict_aligned(values)
        extras = model.auxiliary_scores(values)
        config = dict(min_calibration_rows=16, reference_size=40, reference_update_every=7)
        calibration = np.column_stack([np.linspace(-1, 1, 40), np.ones((40, 2))*10])
        detector = Detector.calibrate(calibration, .99, False, config, .01, (1, 1),
                                      upper_only=[False, True, True])
        extras[150:] = 100
        replay = detector.replay(values, predictions, np.ones(200, bool), 150, 200, auxiliary_scores=extras)
        frame = prediction_frame(pd.Series(pd.date_range("2026-01-01", periods=200, freq="min")),
                                 values, np.ones(200, bool), predictions, replay, model, detector.threshold, 150)
        self.assertTrue(frame.evidence.str.contains("novelty").all())
        self.assertTrue(frame.score_shape_novelty.gt(1).all())
        lower = detector.replay(np.zeros(2), np.zeros((2, 1)), np.ones(2, bool), 0, 2,
                                auxiliary_scores=np.zeros((2, 2)))
        np.testing.assert_array_equal(lower["z"][:, 1:], 0)


if __name__ == "__main__":
    unittest.main()
