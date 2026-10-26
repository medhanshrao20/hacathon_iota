"""Calibrated error limits and causal incident state; no labels at inference."""
from collections import deque
from dataclasses import dataclass, field
import numpy as np


def robust_reference(errors, floor, method="mad", quantile=.99):
    center = np.median(errors, axis=0)
    deviations = np.abs(errors - center)
    if method == "empirical":
        scale = np.quantile(deviations, quantile, axis=0)
    else:
        scale = np.maximum(1.4826 * np.median(deviations, axis=0),
                           0.1 * np.quantile(deviations, .9, axis=0))
    return center, np.maximum(scale, floor)


@dataclass
class Detector:
    center: np.ndarray
    scale: np.ndarray
    threshold: float
    adaptive: bool
    reference_size: int
    update_every: int
    floor: float
    votes_required: int
    vote_window: int
    history: deque
    votes: deque
    accepted_since_update: int = 0
    method: str = "mad"
    quantile: float = .99
    incident_profile: str = "plain"
    close_ratio: float = .6
    recovery_readings: int = 3
    cooldown_readings: int = 8
    min_scale_fraction: float = .5
    initial_scale: np.ndarray = field(default_factory=lambda: np.ones(1))
    incident_open: bool = False
    recovery_count: int = 0
    cooldown_remaining: int = 0

    @classmethod
    def calibrate(cls, errors, quantile, adaptive, config, floor, persistence,
                  method="mad", incident_profile="plain"):
        if len(errors) < config["min_calibration_rows"] or not np.isfinite(errors).all():
            raise ValueError("Insufficient finite, observed calibration errors")
        reference = errors[-config["reference_size"]:] if adaptive else errors
        center, scale = robust_reference(reference, floor, method, quantile)
        scores = np.max(np.abs(errors - center) / scale, axis=1)
        # Never narrow ranges below the physical/numerical training tolerance.
        threshold = max(float(np.quantile(scores, quantile)), 1.0)
        return cls(center, scale, threshold, adaptive, config["reference_size"],
                   config["reference_update_every"], floor, *persistence,
                   deque(reference.copy(), maxlen=config["reference_size"]),
                   deque(maxlen=persistence[1]), method=method, quantile=quantile,
                   incident_profile=incident_profile,
                   close_ratio=config.get("close_threshold_ratio", .6),
                   recovery_readings=config.get("recovery_readings", 3),
                   cooldown_readings=config.get("cooldown_readings", 8),
                   min_scale_fraction=config.get("adaptive_min_scale_fraction", .5),
                   initial_scale=scale.copy())

    def _decision(self, score):
        raw_alert = bool(score > self.threshold)
        self.votes.append(raw_alert)
        opening_vote = sum(self.votes) >= self.votes_required
        if self.incident_profile == "plain":
            return opening_vote, raw_alert
        if self.incident_open:
            self.recovery_count = self.recovery_count + 1 if score < self.threshold*self.close_ratio else 0
            if self.recovery_count >= self.recovery_readings:
                self.incident_open = False
                self.recovery_count = 0
                self.cooldown_remaining = self.cooldown_readings
                self.votes.clear()
        elif self.cooldown_remaining:
            self.cooldown_remaining -= 1
        elif opening_vote:
            self.incident_open = True
        return self.incident_open, raw_alert

    def replay(self, values, predictions, observed, start, stop, local_scales=None):
        n, width = stop - start, predictions.shape[1]
        scores = np.full(n, np.nan)
        flags = np.zeros(n, dtype=bool)
        raw_flags = np.zeros(n, dtype=bool)
        zscores = np.full((n, width), np.nan)
        centers = np.full((n, width), np.nan)
        lower = np.full((n, width), np.nan)
        upper = np.full((n, width), np.nan)
        for row, t in enumerate(range(start, stop)):
            if not observed[t] or not np.isfinite(predictions[t]).all():
                self.votes.clear()
                continue
            unit = np.ones(width) if local_scales is None else local_scales[t]
            if not np.isfinite(unit).all() or np.any(unit <= 0):
                self.votes.clear()
                continue
            error = (values[t] - predictions[t]) / unit
            z = np.abs(error - self.center) / self.scale
            zscores[row] = z
            scores[row] = float(z.max())
            centers[row] = predictions[t] + self.center * unit
            half_width = self.threshold * self.scale * unit
            lower[row], upper[row] = centers[row] - half_width, centers[row] + half_width
            flags[row], raw_flags[row] = self._decision(scores[row])
            # Update ONLY after the decision; never use truth labels for updates.
            if self.adaptive and not raw_flags[row] and not flags[row]:
                self.history.append(error.copy())
                self.accepted_since_update += 1
                if self.accepted_since_update >= self.update_every:
                    self.center, updated_scale = robust_reference(np.asarray(self.history), self.floor,
                                                                  self.method, self.quantile)
                    self.scale = np.maximum(updated_scale, self.initial_scale*self.min_scale_fraction)
                    self.accepted_since_update = 0
        return {"score": scores, "flag": flags, "raw_flag": raw_flags, "z": zscores,
                "center": centers, "lower": lower, "upper": upper,
                "incident_profile": self.incident_profile}
