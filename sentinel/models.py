"""Past-only predictors with optional training-label filtering and seasonality."""
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


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
    pattern_scaler: object = None
    shape_neighbors: object = None
    state_neighbors: object = None
    raw_floor: float = 1e-9

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
        if self.kind == "boosted":
            raw = sliding_window_view(values, self.window)[origins-self.window+1]
            x = np.column_stack([x, window_state(raw, self.raw_floor)])
        return origins, x

    def fit(self, values, observed, train_end, normal_mask=None, config=None):
        # Slice BEFORE period discovery, scaling, windowing, or target building.
        values, observed = values[:train_end], observed[:train_end]
        allowed = observed.copy()
        if normal_mask is not None:
            allowed &= np.asarray(normal_mask[:train_end], dtype=bool)
        self.raw_floor = noise_floor(values, allowed)
        if config and (self.kind == "seasonal" or self.use_seasonality):
            self.period = (config["_training_period"] if "_training_period" in config else
                           detect_period(values, allowed, config["max_seasonal_period"],
                                         config["min_seasonal_correlation"]))
        if self.kind == "pattern":
            windows = sliding_window_view(values, self.window)
            valid = np.isfinite(windows).all(axis=1)
            if normal_mask is not None:
                bad = np.r_[0, np.cumsum(~allowed)]
                valid &= bad[self.window:] == bad[:-self.window]
            elif len(valid):
                # Missing contexts must not become reference patterns.
                bad = np.r_[0, np.cumsum(~observed)]
                valid &= bad[self.window:] == bad[:-self.window]
            reference = windows[valid]
            self.training_examples = int(len(reference))
            self.filtered_examples = int(len(windows)-len(reference))
            if len(reference) < 16:
                raise ValueError("Too few clean training reference patterns")
            limit = config.get("pattern_reference_size", 512) if config else 512
            if len(reference) > limit:
                reference = reference[np.linspace(0, len(reference)-1, limit, dtype=int)]
            shape = window_shape(reference, self.raw_floor)
            state = window_state(reference, self.raw_floor)
            self.pattern_scaler = RobustScaler().fit(state)
            self.shape_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=1).fit(shape)
            self.state_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=1).fit(
                self.pattern_scaler.transform(state))
            return self
        if self.kind not in ("ridge", "boosted"):
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
        if self.kind == "boosted":
            # Fixed rounds: sklearn's random early-stopping split is not used.
            self.estimator = make_pipeline(HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=config.get("boosting_iterations", 100) if config else 100,
                max_leaf_nodes=7, learning_rate=.05, min_samples_leaf=max(20, int(valid.sum()*.01)),
                l2_regularization=self.alpha, early_stopping=False, random_state=42))
            self.estimator.fit(x[valid], y[valid, 0])
        else:
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
        if self.kind in ("ridge", "boosted"):
            forecasts = np.asarray(self.estimator.predict(x[valid])).reshape(-1, len(self.horizons))
        elif self.kind == "mean":
            forecasts = x[valid].mean(axis=1, keepdims=True)
        elif self.kind == "median":
            forecasts = np.median(x[valid], axis=1, keepdims=True)
        elif self.kind == "pattern":
            # A real simple forecast accompanies the separate novelty channels.
            forecasts = x[valid, -3:].mean(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown forecaster: {self.kind}")
        for column, horizon in enumerate(self.horizons):
            targets = origins[valid] + horizon
            usable = targets < len(values)
            result[targets[usable], column] = forecasts[usable, column]
        return result

    @property
    def auxiliary_names(self):
        return ("shape_novelty", "state_novelty") if self.kind == "pattern" else ()

    def auxiliary_scores(self, values):
        """Current-window evidence, available only AFTER the current reading."""
        if self.kind != "pattern":
            return None
        result = np.full((len(values), 2), np.nan)
        if len(values) < self.window:
            return result
        windows = sliding_window_view(values, self.window)
        valid = np.isfinite(windows).all(axis=1)
        if valid.any():
            selected = windows[valid]
            shape = self.shape_neighbors.kneighbors(window_shape(selected, self.raw_floor), return_distance=True)[0][:, 0]
            state = self.state_neighbors.kneighbors(self.pattern_scaler.transform(
                window_state(selected, self.raw_floor)), return_distance=True)[0][:, 0]
            result[np.flatnonzero(valid)+self.window-1] = np.column_stack([shape, state])
        return result


def window_shape(windows, floor):
    centered = windows - windows.mean(axis=1, keepdims=True)
    normalized = centered / np.maximum(windows.std(axis=1, keepdims=True), floor)
    # Eight-bin PAA captures coarse shape without comparing full high-dimensional windows.
    return np.column_stack([part.mean(axis=1) for part in np.array_split(normalized, min(8, windows.shape[1]), axis=1)])


def window_state(windows, floor):
    width = windows.shape[1]
    difference = np.diff(windows, axis=1)
    time_axis = np.arange(width) - (width-1)/2
    slope = (windows @ time_axis) / max(float(time_axis @ time_axis), 1.)
    return np.column_stack([windows.mean(axis=1), windows.std(axis=1),
                            np.median(windows, axis=1), slope*width,
                            np.mean(np.abs(difference), axis=1),
                            np.mean(np.abs(difference) <= floor, axis=1),
                            windows[:, -max(2, width//4):].mean(axis=1)-windows.mean(axis=1)])


def model_specs(config):
    yield Forecaster("mean", 3, (1,))
    yield Forecaster("seasonal", 1, (1,))
    for window in config["windows"]:
        yield Forecaster("median", window, (1,))
        if "boosted" in config.get("extra_models", []):
            yield Forecaster("boosted", window, (1,), 10., use_seasonality=config.get("seasonal_features", False))
        if "pattern" in config.get("extra_models", []):
            yield Forecaster("pattern", window, (1,))
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
    if model.kind == "boosted":
        return "boosted_forecast"
    if model.kind == "pattern":
        return "pattern_hybrid"
    return "ridge_single" if len(model.horizons) == 1 else "ridge_multi"


def baseline_scales(values, observed, window):
    result = np.full((len(values), 1), np.nan)
    if len(values) <= window:
        return result
    windows = sliding_window_view(values, window)[:-1]
    centers = np.median(windows, axis=1)
    result[window:, 0] = 1.4826 * np.median(np.abs(windows - centers[:, None]), axis=1)
    return result
