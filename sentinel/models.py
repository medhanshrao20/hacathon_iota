"""Four small methods. Forecast alignment is explicit and future-safe."""
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Forecaster:
    kind: str
    window: int
    horizons: tuple
    alpha: float = 0.0
    estimator: object = None

    def fit(self, values, observed, train_end):
        if self.kind != "ridge":
            return self
        # Origin is the last input index. Every target must be inside training.
        origins = np.arange(self.window - 1, train_end - max(self.horizons))
        if len(origins) < self.window:
            raise ValueError("Insufficient training history for this window/horizon")
        all_windows = sliding_window_view(values, self.window)
        x = all_windows[origins - self.window + 1]
        target_indices = origins[:, None] + np.array(self.horizons)[None, :]
        y = values[target_indices]
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        valid &= observed[target_indices].all(axis=1)
        if valid.sum() < self.window:
            raise ValueError("Insufficient observed training targets")
        self.estimator = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha, solver="cholesky"))
        self.estimator.fit(x[valid], y[valid])
        return self

    def predict_aligned(self, values):
        """Column h at row T uses inputs ending at T-h, even in batch mode."""
        result = np.full((len(values), len(self.horizons)), np.nan)
        if len(values) < self.window:
            return result
        x = sliding_window_view(values, self.window)
        origins = np.arange(self.window - 1, len(values))
        valid = np.isfinite(x).all(axis=1)
        if not valid.any():
            return result
        if self.kind == "ridge":
            forecasts = self.estimator.predict(x[valid])
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
    for window in config["windows"]:
        yield Forecaster("median", window, (1,))
        for horizons in config["horizon_sets"]:
            for alpha in config["alphas"]:
                yield Forecaster("ridge", window, tuple(horizons), alpha)


def model_name(model):
    if model.kind == "mean":
        return "recent_mean"
    if model.kind == "median":
        return "rolling_median_mad"
    return "ridge_single" if len(model.horizons) == 1 else "ridge_multi"


def baseline_scales(values, observed, window):
    """Past-only raw-value MAD for the median detector, excluding current value."""
    result = np.full((len(values), 1), np.nan)
    if len(values) <= window:
        return result
    windows = sliding_window_view(values, window)[:-1]
    centers = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - centers[:, None]), axis=1)
    result[window:, 0] = 1.4826 * mad
    return result
