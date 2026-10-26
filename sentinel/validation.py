"""Development-only folds, bounded search, and leave-source-group-out fallback."""
import copy
import hashlib
from itertools import product
import json
import numpy as np
import pandas as pd

from .data import event_ids, load_series, split_points
from .detection import Detector
from .evaluation import evaluate
from .models import Forecaster, baseline_scales, detect_period, model_name, model_specs, noise_floor
from .reporting import write_json


def chronological_folds(n, config, explicit=None):
    train_end, cal_end, test_start = split_points(n, config["split_fractions"], explicit)
    width = test_start - cal_end
    folds = [(train_end-shift*width, cal_end-shift*width, test_start-shift*width)
             for shift in reversed(range(config["cv_folds"]))]
    if any(not 0 < t < c < v <= test_start for t, c, v in folds):
        raise ValueError("CV folds do not fit before test; reduce cv_folds or validation-section width")
    return folds, test_start


def prepare_development(path, relative, windows, config, explicit=None):
    stamps, all_values, all_observed, audit = load_series(path, config["max_missing_fraction"])
    folds, test_start = chronological_folds(len(all_values), config, explicit)
    # Objects passed to search contain NO test observations or test label IDs.
    stamps = stamps.iloc[:test_start].reset_index(drop=True)
    values, observed = all_values[:test_start].copy(), all_observed[:test_start].copy()
    ids = event_ids(stamps, windows)
    events = set()
    event_folds = 0
    fold_references = []
    for train_end, cal_end, val_end in folds:
        normal = ids[:train_end] < 0 if config["use_training_labels"] else np.ones(train_end, bool)
        allowed = observed[:train_end] & normal
        floor = noise_floor(values[:train_end], allowed)
        period = detect_period(values[:train_end], allowed, config["max_seasonal_period"],
                               config["min_seasonal_correlation"])
        found = set(ids[cal_end:val_end].tolist()) - {-1}
        events.update(found)
        event_folds += bool(found)
        fold_references.append({"floor": floor, "period": period})
    audit.update(series=relative, folds=[{"train_end": t, "calibration_end": c, "validation_end": v}
                                       for t, c, v in folds], test_start=test_start,
                 selection_distinct_events=len(events), event_folds=event_folds,
                 use_training_labels=config["use_training_labels"],
                 test_status="Previously inspected public test; rerun is exploratory, not fresh evidence")
    group = relative.split('/')[0]
    group = config.get("source_groups", {}).get(group, group)
    return {"path": path, "relative": relative, "group": group,
            "stamps": stamps, "values": values, "observed": observed, "ids": ids,
            "folds": folds, "references": fold_references, "audit": audit,
            "test_start": test_start, "selection_events": len(events), "event_folds": event_folds}


def model_parameters(model):
    return {"kind": model.kind, "family": model_name(model), "window": model.window,
            "horizons": list(model.horizons), "alpha": model.alpha,
            "use_seasonality": model.use_seasonality}


def from_parameters(params):
    return Forecaster(params["kind"], params["window"], tuple(params["horizons"]), params["alpha"],
                      use_seasonality=params["use_seasonality"])


def prepare_errors(values, predictions, observed, model, floor):
    local = (np.maximum(baseline_scales(values, observed, model.window), floor)
             if model.kind == "median" else None)
    errors = values[:, None] - predictions
    if local is not None:
        errors = errors / local
    errors[~observed] = np.nan
    return errors, local


def fit_fold(stream, spec, fold_number, config):
    train_end, cal_end, val_end = stream["folds"][fold_number]
    values, observed = stream["values"][:val_end], stream["observed"][:val_end]
    normal = stream["ids"][:train_end] < 0 if config["use_training_labels"] else None
    reference = stream["references"][fold_number]
    model = copy.deepcopy(spec).fit(values[:train_end], observed[:train_end], train_end, normal,
                                   {**config, "_training_period": reference["period"]})
    predictions = model.predict_aligned(values)
    errors, local = prepare_errors(values, predictions, observed, model, reference["floor"])
    calibration = errors[train_end:cal_end]
    calibration = calibration[np.isfinite(calibration).all(axis=1)]
    if len(calibration) < config["min_calibration_rows"]:
        raise ValueError("Not enough mature, observed calibration errors")
    eligible = observed[cal_end:val_end] & np.isfinite(predictions[cal_end:val_end]).all(axis=1)
    if not eligible.any():
        raise ValueError("No eligible validation targets")
    mae = float(np.mean(np.abs(values[cal_end:val_end][eligible] - predictions[cal_end:val_end, 0][eligible])))
    return model, predictions, calibration, local, mae


def calibrate(calibration, local, floor, params, config):
    return Detector.calibrate(calibration, params["quantile"], params["adaptive"], config,
                              1e-3 if local is not None else floor, params["persistence"],
                              params["method"], params["incident_profile"])


def aggregate(rows):
    """Pool counts and fold-score moments without reading any test data."""
    tp = sum(r["detected_events"] for r in rows)
    fp = sum(r["false_alerts"] for r in rows)
    fn = sum(r["missed_events"] for r in rows)
    normal_points = sum(r["normal_points"] for r in rows)
    count = total = squares = 0
    for row in rows:
        if "event_fold_count" in row:
            count += row["event_fold_count"]
            total += row["event_fold_f1_sum"]
            squares += row["event_fold_f1_sq_sum"]
        elif row["true_events"]:
            count += 1
            total += row["event_f1"]
            squares += row["event_f1"]**2
    dispersion = float(np.sqrt(max(0, squares/count - (total/count)**2))) if count else 0.0
    return {"event_f1": 2*tp/max(1, 2*tp+fp+fn), "event_precision": tp/max(1, tp+fp),
            "event_recall": tp/max(1, tp+fn), "detected_events": tp, "false_alerts": fp,
            "missed_events": fn, "true_events": tp+fn, "normal_points": normal_points,
            "false_alerts_per_1000_normal_points": 1000*fp/max(1, normal_points),
            "event_fold_count": count, "event_fold_f1_sum": total,
            "event_fold_f1_sq_sum": squares, "event_fold_f1_std": dispersion}


def rank(summary, params, config):
    excess = max(0, summary["false_alerts_per_1000_normal_points"] - config["false_alert_budget_per_1000"])
    objective = summary["event_f1"] - .1*summary["event_fold_f1_std"] - .05*excess
    complexity = params["window"]*len(params["horizons"]) if params["kind"] == "ridge" else params["window"]
    return (objective, -summary["false_alerts_per_1000_normal_points"], summary["event_recall"],
            -complexity, params["alpha"])


def develop_series(stream, config, output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "audit.json", stream["audit"])
    trials, failures, params_by_id = [], [], {}
    for model_number, spec in enumerate(model_specs(config), 1):
        base = model_parameters(spec)
        for fold_number, (_, cal_end, val_end) in enumerate(stream["folds"]):
            try:
                model, predictions, calibration, local, mae = fit_fold(stream, spec, fold_number, config)
                options = product(config["adaptation"], config["threshold_quantiles"], config["persistence"],
                                  config["threshold_methods"], config["incident_profiles"])
                for adaptive, quantile, persistence, method, incident in options:
                    params = {**base, "adaptive": adaptive, "quantile": quantile,
                              "persistence": list(persistence), "method": method, "incident_profile": incident}
                    token = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
                    params_by_id[token] = params
                    detector = calibrate(calibration, local, stream["references"][fold_number]["floor"], params, config)
                    result = detector.replay(stream["values"], predictions, stream["observed"], cal_end, val_end, local)
                    metrics = evaluate(stream["stamps"].iloc[cal_end:val_end].reset_index(drop=True),
                                       stream["ids"][cal_end:val_end], result["flag"], np.isfinite(result["score"]))
                    trials.append({"candidate": token, "fold": fold_number, **params, **metrics,
                                   "forecast_mae": mae, "learned_period": model.period,
                                   "training_examples": model.training_examples,
                                   "filtered_training_examples": model.filtered_examples})
            except (ValueError, np.linalg.LinAlgError) as error:
                failures.append({**base, "fold": fold_number, "error": str(error)})
        if model_number % 5 == 0:
            pd.DataFrame(trials).to_csv(output / "experiment_log.csv", index=False)
            print(f"  {model_number} forecast specifications; {len(trials)} fold trials", flush=True)
    pd.DataFrame(trials).to_csv(output / "experiment_log.csv", index=False)
    write_json(output / "candidate_failures.json", failures)
    grouped = {}
    for row in trials:
        grouped.setdefault(row["candidate"], []).append(row)
    summaries = {key: {"parameters": params_by_id[key], "validation": aggregate(rows)}
                 for key, rows in grouped.items() if len(rows) == config["cv_folds"]}
    if not summaries:
        raise ValueError("No configuration completed every chronological fold")
    write_json(output / "cv_summary.json", summaries)
    stream["summaries"], stream["output"] = summaries, output
    return stream


def pooled_by_group(streams):
    pools = {}
    for target in streams:
        # Historical peers must also predate the target's test boundary in wall
        # clock time: do not borrow tuning labels from another stream's future.
        cutoff = target["stamps"].iloc[-1]
        others = [s for s in streams if s["group"] != target["group"]
                  and s["stamps"].iloc[-1] <= cutoff]
        grouped = {}
        for stream in others:
            for key, item in stream["summaries"].items():
                grouped.setdefault(key, []).append(item["validation"])
        pools[target["relative"]] = {key: aggregate(rows) for key, rows in grouped.items()}
    return pools


def preset_rank(params):
    """Predeclared conservative fallback, independent of validation labels."""
    return (not params["adaptive"], params["method"] == "empirical", params["quantile"],
            params["incident_profile"] == "guarded", params["persistence"] == [1, 1],
            -abs(params["window"]-48), -abs(max(params["horizons"])-24), params["alpha"])


def select_methods(stream, pools, config):
    own_information = (stream["selection_events"] >= config["min_selection_events"]
                       and stream["event_folds"] >= config["min_event_folds"])
    selected = {}
    for key, item in stream["summaries"].items():
        params = item["parameters"]
        pooled = pools.get(stream["relative"], {}).get(key)
        if own_information:
            metrics, source = item["validation"], "own_chronological_folds"
        elif pooled and pooled["true_events"] >= config["min_selection_events"] and pooled["event_fold_count"] >= config["min_event_folds"]:
            metrics, source = pooled, "other_source_groups_development_only"
        else:
            metrics, source = item["validation"], "predeclared_preset_insufficient_events"
        key_rank = rank(metrics, params, config) if source != "predeclared_preset_insufficient_events" else preset_rank(params)
        family = params["family"]
        # Prefer evidence-based candidates if a pooled candidate is unavailable.
        comparison = (source != "predeclared_preset_insufficient_events", key_rank)
        if family not in selected or comparison > selected[family]["rank"]:
            selected[family] = {"candidate": key, "parameters": params, "validation": item["validation"],
                                "selection_metrics": metrics, "selection_source": source, "rank": comparison}
    evidence = {family: item for family, item in selected.items()
                if item["selection_source"] != "predeclared_preset_insufficient_events"}
    winner = max(evidence, key=lambda family: evidence[family]["rank"]) if evidence else "recent_mean"
    return selected, winner
