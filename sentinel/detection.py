"""Calibrated ranges and sequential, score-before-update decisions."""
from collections import deque
from dataclasses import dataclass
import numpy as np


def robust_reference(errors, floor):
    center = np.median(errors, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(errors - center), axis=0), floor)
    return center, scale


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

    @classmethod
    def calibrate(cls, errors, quantile, adaptive, config, floor, persistence):
        if len(errors) < config["min_calibration_rows"] or not np.isfinite(errors).all():
            raise ValueError("Insufficient finite, observed calibration errors")
        reference = errors[-config["reference_size"]:] if adaptive else errors
        center, scale = robust_reference(reference, floor)
        scores = np.max(np.abs(errors - center) / scale, axis=1)
        threshold = max(float(np.quantile(scores, quantile)), 1e-8)
        return cls(center, scale, threshold, adaptive, config["reference_size"],
                   config["reference_update_every"], floor, *persistence,
                   deque(reference.copy(), maxlen=config["reference_size"]),
                   deque(maxlen=persistence[1]))

    def replay(self, values, predictions, observed, start, stop, local_scales=None):
        """Mutate state only after scoring; no labels are accepted by this method."""
        n, width = stop - start, predictions.shape[1]
        scores = np.full(n, np.nan)
        flags = np.zeros(n, dtype=bool)
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
            raw_alert = bool(scores[row] > self.threshold)
            self.votes.append(raw_alert)
            flags[row] = sum(self.votes) >= self.votes_required
            if self.adaptive and not raw_alert and not flags[row]:
                self.history.append(error.copy())
                self.accepted_since_update += 1
                if self.accepted_since_update >= self.update_every:
                    self.center, self.scale = robust_reference(np.asarray(self.history), self.floor)
                    self.accepted_since_update = 0
        return {"score": scores, "flag": flags, "z": zscores, "center": centers,
                "lower": lower, "upper": upper}
