"""Past-only predictors with optional training-label filtering and seasonality."""
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def detect_period(values, allowed, maximum=512, minimum_correlation=0.6):
    """Find a repeating lag from TRAINING ONLY; return 1 if evidence is weak."""
    maximum = min(maximum, len(values) // 3)
    correlations = np.full(maximum + 1, np.nan)
    for lag in range(1, maximum + 1):
        valid = allowed[lag:] & allowed[:-lag]
        valid &= np.isfinite(values[lag:]) & np.isfinite(values[:-lag])
        if valid.sum() < max(24, 2 * lag):
            continue
        left, right = values[lag:][valid], values[:-lag][valid]
        if np.std(left) > 1e-12 and np.std(right) > 1e-12:
            correlations[lag] = np.corrcoef(left, right)[0, 1]
    candidates = []
    for lag in range(3, maximum):
        value = correlations[lag]
        if (np.isfinite(value) and value >= minimum_correlation
                and value >= correlations[lag-1] and value > correlations[lag+1]
                and value - correlations[max(1, lag//2)] >= 0.1):
            candidates.append(lag)
    return min(candidates) if candidates else 1


def noise_floor(values, allowed):
    """Training-derived tolerance for constant/nearly constant signals."""
    finite = allowed & np.isfinite(values)
    sample = values[finite]
    if not len(sample):
        raise ValueError("No usable training observations")
    center = np.median(sample)
    spread = 1.4826 * np.median(np.abs(sample - center))
    pairs = finite[1:] & finite[:-1]
    differences = np.abs(np.diff(values)[pairs])
    typical_change = np.median(differences) if len(differences) else 0.0
    return max(float(spread) * 1e-3, float(typical_change) * 0.1,
               float(np.median(np.abs(sample))) * 1e-8, 1e-9)


@dataclass
class Forecaster:
    kind: str
    window: int
    horizons: tuple
    alpha: float = 0.0
    estimator: object = None
    use_seasonality: bool = False
    period: int = 1
    training_examples: int = 0
    filtered_examples: int = 0

    @property
    def history_length(self):
        if self.kind == "seasonal":
            return self.period
        return max(self.window, 2*self.period if self.use_seasonality and self.period > 1 else 1)

    def _features(self, values):
        origins = np.arange(self.history_length - 1, len(values))
        if not len(origins):
            return origins, np.empty((0, self.window))
        x = sliding_window_view(values, self.window)[origins - self.window + 1]
        if self.use_seasonality and self.period > 1:
            x = np.column_stack([x, values[origins+1-self.period], values[origins+1-2*self.period]])
        return origins, x

    def fit(self, values, observed, train_end, normal_mask=None, config=None):
        # Slice BEFORE period discovery, scaling, windowing, or target building.
        values, observed = values[:train_end], observed[:train_end]
        allowed = observed.copy()
        if normal_mask is not None:
            allowed &= np.asarray(normal_mask[:train_end], dtype=bool)
        if config and (self.kind == "seasonal" or self.use_seasonality):
            self.period = (config["_training_period"] if "_training_period" in config else
                           detect_period(values, allowed, config["max_seasonal_period"],
                                         config["min_seasonal_correlation"]))
        if self.kind != "ridge":
            return self
        origins, x = self._features(values)
        inside = origins + max(self.horizons) < train_end
        origins, x = origins[inside], x[inside]
        targets = origins[:, None] + np.asarray(self.horizons)[None, :]
        y = values[targets]
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        valid &= observed[targets].all(axis=1)
        if normal_mask is not None:
            # Reject the ENTIRE input-to-last-target span, not just its endpoint.
            bad_prefix = np.r_[0, np.cumsum(~allowed)]
            start = origins - self.history_length + 1
            stop = origins + max(self.horizons) + 1
            valid &= bad_prefix[stop] == bad_prefix[start]
        self.training_examples = int(valid.sum())
        self.filtered_examples = int(len(valid) - valid.sum())
        if self.training_examples < max(16, x.shape[1]):
            raise ValueError("Too few clean training windows for this configuration")
        self.estimator = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha, solver="cholesky"))
        self.estimator.fit(x[valid], y[valid])
        return self

    def predict_aligned(self, values):
        result = np.full((len(values), len(self.horizons)), np.nan)
        if self.kind == "seasonal":
            if len(values) > self.period:
                result[self.period:, 0] = values[:-self.period]
            return result
        if len(values) < self.history_length:
            return result
        origins, x = self._features(values)
        valid = np.isfinite(x).all(axis=1)
        if not valid.any():
            return result
        if self.kind == "ridge":
            forecasts = np.asarray(self.estimator.predict(x[valid])).reshape(-1, len(self.horizons))
        elif self.kind == "mean":
            forecasts = x[valid].mean(axis=1, keepdims=True)
        elif self.kind == "median":
            forecasts = np.median(x[valid], axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown forecaster: {self.kind}")
        for column, horizon in enumerate(self.horizons):
            targets = origins[valid] + horizon
            usable = targets < len(values)
            result[targets[usable], column] = forecasts[usable, column]
        return result


def model_specs(config):
    yield Forecaster("mean", 3, (1,))
    yield Forecaster("seasonal", 1, (1,))
    for window in config["windows"]:
        yield Forecaster("median", window, (1,))
        for horizons in config["horizon_sets"]:
            for alpha in config["alphas"]:
                yield Forecaster("ridge", window, tuple(horizons), alpha,
                                 use_seasonality=config.get("seasonal_features", False))


def model_name(model):
    if model.kind == "mean":
        return "recent_mean"
    if model.kind == "seasonal":
        return "seasonal_naive"
    if model.kind == "median":
        return "rolling_median_mad"
    return "ridge_single" if len(model.horizons) == 1 else "ridge_multi"


def baseline_scales(values, observed, window):
    result = np.full((len(values), 1), np.nan)
    if len(values) <= window:
        return result
    windows = sliding_window_view(values, window)[:-1]
    centers = np.median(windows, axis=1)
    result[window:, 0] = 1.4826 * np.median(np.abs(windows - centers[:, None]), axis=1)
    return result
