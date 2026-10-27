"""Optional spike, two-sided CUSUM and confirmation channels on one incident state."""
import copy
from dataclasses import dataclass

import numpy as np

from .detection import Detector, robust_reference


@dataclass
class EvidenceDetector(Detector):
    spike_level: float | None = None
    cusum_level: float | None = None
    cusum_drift: float = .5
    cusum_center: object = None
    cusum_scale: object = None
    positive: object = None
    negative: object = None
    confirmation_level: float | None = None
    confirmation_reference: object = None
    pending_cusum: float = 0.
    pending_confirmation: float = 1.
    pending_quiet: bool = True
    cusum_incident: bool = False

    def _decision(self, score):
        raw = bool(score > self.threshold)
        severe = self.spike_level is not None and score > self.spike_level
        accumulated = self.cusum_level is not None and self.pending_cusum > self.cusum_level
        confirmed = self.confirmation_level is None or self.pending_confirmation >= self.confirmation_level
        self.votes.append(raw)
        opening = (sum(self.votes) >= self.votes_required or accumulated) and confirmed
        if self.incident_profile == "plain":
            return bool(severe or opening), raw
        if self.incident_open:
            quiet = score < self.threshold*self.close_ratio
            quiet &= not severe and (not self.cusum_incident or self.pending_quiet)
            self.recovery_count = self.recovery_count + 1 if quiet else 0
            if self.recovery_count >= self.recovery_readings:
                self.incident_open = False
                self.cusum_incident = False
                self.recovery_count = 0
                self.cooldown_remaining = self.cooldown_readings
                self.votes.clear()
                if self.positive is not None:
                    self.positive.fill(0)
                    self.negative.fill(0)
        elif severe:  # Soft budget: a severe observation is never blocked by a quota.
            self.incident_open = True
            self.cooldown_remaining = 0
            self.cusum_incident = bool(accumulated)
        elif self.cooldown_remaining:
            self.cooldown_remaining -= 1
        elif opening:
            self.incident_open = True
            self.cusum_incident = bool(accumulated)
        return self.incident_open, raw

    def replay(self, values, predictions, observed, start, stop, local_scales=None,
               auxiliary_scores=None, confirmation_scores=None):
        if start == stop:
            return super().replay(values, predictions, observed, start, stop, local_scales, auxiliary_scores)
        pieces, spike, cusum, confirmed, triggers = [], [], [], [], []
        # ponytail: reuse the checked causal scorer row by row; vectorize only if measured latency requires it.
        for t in range(start, stop):
            unit = 1. if local_scales is None else local_scales[t]
            error = (values[t]-predictions[t])/unit
            valid = (observed[t] and np.isfinite(error).all() and np.isfinite(np.asarray(unit)).all()
                     and np.all(np.asarray(unit) > 0))
            self.pending_cusum, self.pending_quiet = 0., True
            if self.cusum_level is not None:
                if self.positive is None:
                    self.positive, self.negative = np.zeros_like(error), np.zeros_like(error)
                if valid:
                    signed = (error-self.cusum_center)/self.cusum_scale
                    self.pending_quiet = bool(np.max(np.abs(signed)) <= self.cusum_drift)
                    if not self.incident_open:
                        self.positive = np.maximum(0, self.positive+signed-self.cusum_drift)
                        self.negative = np.maximum(0, self.negative-signed-self.cusum_drift)
                        self.pending_cusum = float(max(self.positive.max(), self.negative.max()))
                    else:
                        self.positive.fill(0)
                        self.negative.fill(0)
                else:
                    self.positive.fill(0)
                    self.negative.fill(0)
            self.pending_confirmation = 1.
            if self.confirmation_level is not None:
                if confirmation_scores is None:
                    raise ValueError("Confirmation policy requires secondary scores")
                sample = confirmation_scores[t]
                self.pending_confirmation = (max(np.searchsorted(ref, value, side="right")/len(ref)
                    for ref, value in zip(self.confirmation_reference, sample))
                    if np.isfinite(sample).all() else float("nan"))
            was_open = self.incident_open
            part = super().replay(values, predictions, observed, t, t+1, local_scales, auxiliary_scores)
            score = part["score"][0]
            is_spike = bool(self.spike_level is not None and np.isfinite(score) and score > self.spike_level)
            is_cusum = bool(self.cusum_level is not None and self.pending_cusum > self.cusum_level)
            trigger = "none"
            if part["flag"][0]:
                trigger = ("incident_active" if was_open else "spike" if is_spike else
                           "cusum" if is_cusum else "persistence")
            pieces.append(part)
            spike.append(is_spike)
            cusum.append(self.pending_cusum)
            confirmed.append(self.pending_confirmation)
            triggers.append(trigger)
        result = {key: (np.concatenate([p[key] for p in pieces]) if isinstance(value, np.ndarray) else value)
                  for key, value in pieces[0].items()}
        result.update(spike_triggered=np.asarray(spike), cusum_statistic=np.asarray(cusum),
                      confirmation_percentile=np.asarray(confirmed), decision_trigger=np.asarray(triggers))
        result.update(spike_threshold=np.full(stop-start, np.nan if self.spike_level is None else self.spike_level),
                      cusum_threshold=np.full(stop-start, np.nan if self.cusum_level is None else self.cusum_level),
                      confirmation_threshold=np.full(stop-start, np.nan if self.confirmation_level is None else self.confirmation_level))
        return result


def score_errors(errors, detector):
    difference = errors-detector.center
    z = np.where(detector.upper_only, np.maximum(difference, 0), np.abs(difference))/detector.scale
    return z.max(axis=1)


def replay_policy(detector, values, predictions, observed, start, stop, local=None, secondary=None):
    if isinstance(detector, EvidenceDetector):
        return detector.replay(values, predictions, observed, start, stop, local,
                               confirmation_scores=secondary)
    return detector.replay(values, predictions, observed, start, stop, local)


def cusum_reference(errors, center, scale, drift):
    positive, negative = np.zeros(errors.shape[1]), np.zeros(errors.shape[1])
    path = []
    for signed in (errors-center)/scale:
        if not np.isfinite(signed).all():
            positive.fill(0)
            negative.fill(0)
            path.append(float("nan"))
            continue
        positive = np.maximum(0, positive+signed-drift)
        negative = np.maximum(0, negative-signed-drift)
        path.append(max(positive.max(), negative.max()))
    return np.asarray(path)


def policy_specs():
    yield {"name": "control_v2", "revised": False, "spike": False, "cusum": False, "confirm": False}
    yield {"name": "A_spike", "revised": False, "spike": True, "cusum": False, "confirm": False}
    yield {"name": "B_calibration", "revised": True, "spike": False, "cusum": False, "confirm": False}
    yield {"name": "C_spike_calibration", "revised": True, "spike": True, "cusum": False, "confirm": False}
    yield {"name": "D_cusum", "revised": True, "spike": True, "cusum": True, "confirm": False}
    yield {"name": "E_confirmation", "revised": True, "spike": False, "cusum": False, "confirm": True}
    yield {"name": "F_confirmation_bypass", "revised": True, "spike": True, "cusum": False, "confirm": True}
    yield {"name": "G_single_exceedance", "revised": True, "spike": False, "cusum": False,
           "confirm": False, "single": True}


def fit_policy(errors, params, floor, config, spec, secondary=None):
    """No labels accepted. Revised references and rate tuning use disjoint cal blocks."""
    from .validation import calibrate
    if spec["name"] == "control_v2":
        return calibrate(errors[np.isfinite(errors).all(axis=1)], None, floor, params, config), {"mode": "unchanged_v2"}
    valid = np.isfinite(errors).all(axis=1)
    if secondary is not None:
        valid &= np.isfinite(secondary).all(axis=1)
    split = len(errors)//2
    if min(int(valid[:split].sum()), int(valid[split:].sum())) < config["min_calibration_rows"]:
        raise ValueError("Two calibration blocks need more observed errors")
    reference, tuning = errors[:split][valid[:split]], errors[split:]
    revised = spec["revised"]
    local_params = {**params, "method": "empirical" if revised else params["method"]}
    if spec.get("single"):
        local_params["persistence"] = [1, 1]  # Explicitly a single-exceedance variant, not corroboration.
    source = reference if revised else errors[np.isfinite(errors).all(axis=1)]
    detector = EvidenceDetector.calibrate(source, .99 if revised else params["quantile"],
        params["adaptive"], config, floor, local_params["persistence"], local_params["method"],
        params["incident_profile"])
    scores = score_errors(tuning if revised else errors, detector)
    scores = scores[np.isfinite(scores)]
    if spec["spike"]:
        detector.spike_level = max(detector.threshold, float(np.max(scores)))
    if spec["cusum"]:
        detector.cusum_center, detector.cusum_scale = robust_reference(reference, floor)
        detector.cusum_drift = config["v4"]["cusum_drift"]
        path = cusum_reference(errors[:split], detector.cusum_center, detector.cusum_scale, detector.cusum_drift)
        detector.cusum_level = max(1., float(np.nanquantile(path, config["v4"]["cusum_reference_quantile"])))
    if spec["confirm"]:
        if secondary is None:
            raise ValueError("Missing training-normal pattern detector calibration")
        detector.confirmation_reference = tuple(np.sort(reference_column) for reference_column in secondary[:split][valid[:split]].T)
        detector.confirmation_level = config["v4"]["confirmation_percentile"]
    # Whole-policy episode rate, including spike bypass, confirmation and CUSUM.
    candidates = [detector.threshold]
    if revised:
        candidates = sorted(set(max(1., float(np.quantile(scores, q)))
                                for q in config["v4"]["rate_quantiles"]))
    trials = []
    for threshold in candidates:
        probe = copy.deepcopy(detector)
        probe.threshold = threshold
        if probe.spike_level is not None:
            probe.spike_level = max(probe.spike_level, threshold)
        n = len(tuning)
        replay = probe.replay(np.zeros(n), -tuning, np.isfinite(tuning).all(axis=1), 0, n,
                              confirmation_scores=None if secondary is None else secondary[split:])
        flags = replay["flag"]
        episodes = int((flags & ~np.r_[False, flags[:-1]]).sum())
        rate = 10000*episodes/max(1, int(np.isfinite(tuning).all(axis=1).sum()))
        trials.append({"threshold": threshold, "reference_alerts_per_10000": rate})
    budget = config["v4"]["alert_budget_per_10000"]
    # Soft target: excess is minimized, then retain the most sensitive threshold.
    chosen = min(trials, key=lambda row: (max(0, row["reference_alerts_per_10000"]-budget), row["threshold"]))
    detector.threshold = chosen["threshold"]
    if detector.spike_level is not None:
        detector.spike_level = max(detector.spike_level, detector.threshold)
    # Replay the later calibration block so continuation starts with historical state only.
    n = len(tuning)
    if revised:
        detector.replay(np.zeros(n), -tuning, np.isfinite(tuning).all(axis=1), 0, n,
                        confirmation_scores=None if secondary is None else secondary[split:])
    return detector, {"mode": "split_calibration" if revised else "v2_reference_plus_spike",
        "reference_rows": len(reference), "tuning_rows": len(tuning), "rate_trials": trials,
        "chosen": chosen, "spike_threshold": detector.spike_level, "cusum_threshold": detector.cusum_level,
        "confirmation_percentile": detector.confirmation_level,
        "warning": "Unlabeled calibration exposure is not certified normal data; rate is not a false-alarm guarantee"}
