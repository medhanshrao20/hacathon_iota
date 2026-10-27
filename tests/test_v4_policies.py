"""Causal checks for the optional channels; baseline decisions remain untouched."""
import copy
import unittest

import numpy as np
import pandas as pd

from sentinel.detection import Detector
from sentinel.experiment_metrics import choose_policy
from sentinel.policies import EvidenceDetector, cusum_reference, fit_policy


class EvidencePolicies(unittest.TestCase):
    def config(self):
        return dict(min_calibration_rows=16, reference_size=32, reference_update_every=7,
                    recovery_readings=3, cooldown_readings=8,
                    v4=dict(cusum_drift=.5, cusum_reference_quantile=.99,
                            confirmation_percentile=.95, rate_quantiles=[.99,.999],
                            alert_budget_per_10000=10))

    def detector(self):
        detector = EvidenceDetector.calibrate(np.tile([[-1.],[1.]], (32,1)), .99,
            False, self.config(), .01, (2,3), incident_profile="guarded")
        detector.center, detector.scale, detector.threshold = np.zeros(1), np.ones(1), 2.
        return detector

    def test_spike_bypasses_votes_and_cooldown_but_uses_one_incident(self):
        plain, severe = self.detector(), self.detector()
        severe.spike_level = 4.
        values = np.r_[0., 5., np.zeros(10), 5., np.zeros(5)]
        predictions, observed = np.zeros((len(values),1)), np.ones(len(values),bool)
        baseline = plain.replay(values,predictions,observed,0,len(values))
        actual = severe.replay(values,predictions,observed,0,len(values))
        self.assertFalse(baseline["flag"].any())
        self.assertTrue(actual["flag"][1])
        self.assertTrue(actual["flag"][12])
        self.assertEqual(int((actual["flag"] & ~np.r_[False,actual["flag"][:-1]]).sum()),2)
        self.assertEqual(actual["decision_trigger"][1], "spike")

    def test_two_sided_cusum_recovers_subthreshold_changes_and_is_chunk_causal(self):
        for direction in (-1., 1.):
            detector = self.detector()
            detector.cusum_center, detector.cusum_scale = np.zeros(1), np.ones(1)
            detector.cusum_drift, detector.cusum_level = .2, 3.
            values = np.r_[np.zeros(5), np.ones(30)*direction*.7, np.zeros(15)]
            predictions, observed = np.zeros((len(values),1)), np.ones(len(values),bool)
            full = copy.deepcopy(detector).replay(values,predictions,observed,0,len(values))
            self.assertFalse(full["raw_flag"].any())
            self.assertTrue(full["flag"][5:35].any())
            self.assertIn("cusum", full["decision_trigger"])
            first = detector.replay(values,predictions,observed,0,20)
            second = detector.replay(values,predictions,observed,20,len(values))
            np.testing.assert_array_equal(full["flag"], np.r_[first["flag"],second["flag"]])
            np.testing.assert_allclose(full["cusum_statistic"], np.r_[first["cusum_statistic"],second["cusum_statistic"]])
            changed = values.copy()
            changed[20:] = 1e8
            fresh = self.detector()
            fresh.cusum_center, fresh.cusum_scale = np.zeros(1),np.ones(1)
            fresh.cusum_drift, fresh.cusum_level = .2,3.
            other = fresh.replay(changed,predictions,observed,0,len(values))
            np.testing.assert_array_equal(full["flag"][:20],other["flag"][:20])

    def test_confirmation_has_its_own_percentile_and_severe_bypass(self):
        detector = self.detector()
        detector.confirmation_reference = (np.arange(100.),np.arange(100.))
        detector.confirmation_level = .95
        detector.spike_level = 9.
        values = np.array([3.,3.,3.,0.,0.,0.,10.])
        secondary = np.zeros((len(values),2))
        result = detector.replay(values,np.zeros((len(values),1)),np.ones(len(values),bool),
                                 0,len(values),confirmation_scores=secondary)
        self.assertFalse(result["flag"][:6].any())
        self.assertTrue(result["flag"][-1])
        self.assertTrue((result["confirmation_percentile"] < .95).all())

    def test_control_calibration_and_missing_sequence_are_preserved(self):
        params=dict(horizons=[1],quantile=.99,adaptive=False,persistence=[2,3],method="mad",incident_profile="guarded")
        errors=np.linspace(-1,1,80)[:,None]
        control,_=fit_policy(errors,params,.01,self.config(),dict(name="control_v2"))
        expected=Detector.calibrate(errors,.99,False,self.config(),.01,(2,3),"mad","guarded")
        np.testing.assert_allclose(control.center,expected.center)
        np.testing.assert_allclose(control.scale,expected.scale)
        self.assertEqual(control.threshold,expected.threshold)
        broken=errors.copy()
        broken[60]=np.nan
        candidate,audit=fit_policy(broken,params,.01,self.config(),
            dict(name="B_calibration",revised=True,spike=False,cusum=False,confirm=False))
        self.assertEqual(audit["tuning_rows"],40)  # Missing row was not removed from the sequence.
        self.assertTrue(np.isfinite(candidate.threshold))
        reference=cusum_reference(np.array([[1.],[1.],[np.nan],[1.]]),np.zeros(1),np.ones(1),.5)
        np.testing.assert_allclose(reference,[.5,1.,np.nan,.5],equal_nan=True)

    def test_adoption_rejects_future_related_and_uncertain_evidence(self):
        stamps=pd.Series(pd.to_datetime(["2026-01-01"],utc=True))
        target=dict(relative="own",group="A",stamps=stamps)
        settings=dict(bootstrap_repetitions=100,confidence_level=.9,minimum_policy_streams=4,
            minimum_policy_events=5,minimum_f1_gain=.02,maximum_recall_drop=0,
            alert_budget_per_10000=10)
        rows=[]
        for name,group in [("own","A"),("b","B"),("c","C"),("d","D")]:
            for policy,tp,fp,fn in [("control_v2",1,10,1),("A_spike",2,0,0)]:
                rows.append(dict(series=name,group=group,policy=policy,fold=1,
                    validation_end="2025-12-31T00:00:00+00:00",detected_events=tp,
                    false_alerts=fp,missed_events=fn,true_events=2,normal_points=10000,
                    event_f1=2*tp/(2*tp+fp+fn)))
        future={**rows[-1],"series":"future","group":"Z","validation_end":"2027-01-01T00:00:00+00:00"}
        related={**rows[-1],"series":"related","group":"A"}
        winner,audit=choose_policy(target,rows+[future,related],[dict(name="control_v2"),dict(name="A_spike")],settings,42)
        self.assertEqual(winner,"A_spike")
        self.assertEqual(len(audit["historical_sources"]),4)
        equal=copy.deepcopy(rows)
        for i in range(1,len(equal),2):
            equal[i].update({k:equal[i-1][k] for k in ("detected_events","false_alerts","missed_events","event_f1")})
        winner,_=choose_policy(target,equal,[dict(name="control_v2"),dict(name="A_spike")],settings,42)
        self.assertEqual(winner,"control_v2")
