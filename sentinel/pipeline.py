"""CLI orchestration. Test labels are never used to choose configurations."""
import argparse
import copy
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback

import joblib
import numpy as np
import pandas as pd

from .data import download_nab, event_ids, load_series, split_points
from .detection import Detector
from .evaluation import evaluate, selection_key
from .models import baseline_scales, model_name, model_specs
from .reporting import chart, prediction_frame, summary_report, write_json

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_config(config):
    fractions = config["split_fractions"]
    if len(fractions) != 4 or any(x <= 0 for x in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("Four positive split_fractions must sum to one")
    for key in ("windows", "alphas", "threshold_quantiles", "horizon_sets", "adaptation", "persistence"):
        if not config[key]:
            raise ValueError(f"Empty setting: {key}")
    if any(not isinstance(w, int) or w < 2 for w in config["windows"]):
        raise ValueError("windows must contain integers >= 2")
    if any(a <= 0 for a in config["alphas"]):
        raise ValueError("alphas must be positive")
    if any(not 0 < q < 1 for q in config["threshold_quantiles"]):
        raise ValueError("threshold_quantiles must be between zero and one")
    if [1] not in config["horizon_sets"]:
        raise ValueError("Include [1] in horizon_sets for the single-horizon comparison")
    if not any(len(h) > 1 for h in config["horizon_sets"]):
        raise ValueError("Include a multi-horizon set for the four-method comparison")
    for horizons in config["horizon_sets"]:
        if not horizons or len(set(horizons)) != len(horizons) or horizons[0] != 1 or any(
                not isinstance(h, int) or h < 1 for h in horizons):
            raise ValueError("Each horizon set must start with 1 and have unique positive integers")
    for required, width in config["persistence"]:
        if not 1 <= required <= width:
            raise ValueError("Persistence must satisfy 1 <= required <= width")
    if config["reference_size"] < 16 or config["reference_update_every"] < 1:
        raise ValueError("Invalid reference buffer settings")
    if config["min_calibration_rows"] < 16 or not 0 <= config["max_missing_fraction"] < 1:
        raise ValueError("Invalid calibration or missing-data settings")


def find_data(config, pattern=None, limit=None):
    root = project_path(config["data_dir"])
    labels_path = project_path(config["labels_file"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    paths = sorted(p for p in root.rglob("*.csv") if any(
        p.match(glob) for glob in config["include"]))
    if pattern:
        paths = [p for p in paths if fnmatch.fnmatch(p.relative_to(root).as_posix(), pattern)]
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError("No CSV files matched; check data_dir and --only")
    for path in paths:
        if path.relative_to(root).as_posix() not in labels:
            raise ValueError(f"Missing labels key for {path.name}; unknown labels are not normal labels")
    return root, paths, labels


def prepare_errors(values, predictions, observed, model, floor):
    local = None
    if model.kind == "median":
        local = np.maximum(baseline_scales(values, observed, model.window), floor)
    errors = values[:, None] - predictions
    if local is not None:
        errors = errors / local
    errors[~observed] = np.nan
    return errors, local


def search_series(path, relative, windows, config, output, explicit_split=None):
    stamps, values, observed, audit = load_series(path, config["max_missing_fraction"])
    ids = event_ids(stamps, windows)
    train_end, cal_end, val_end = split_points(len(values), config["split_fractions"], explicit_split)
    max_window = max(config["windows"])
    max_horizon = max(max(h) for h in config["horizon_sets"])
    common_start = max_window + max_horizon - 1
    if train_end <= common_start + max_window:
        raise ValueError("Series is too short for the configured common history; reduce windows/horizons")
    floor = max(float(np.nanstd(values[:train_end])) * 1e-6, 1e-9)
    audit.update({"series": relative, "train_end": train_end, "calibration_end": cal_end,
                  "validation_end": val_end, "test_end": len(values),
                  "training_anomaly_points_for_audit_only": int((ids[:train_end] >= 0).sum()),
                  "validation_events": int(len(set(ids[cal_end:val_end]) - {-1})),
                  "test_events": int(len(set(ids[val_end:]) - {-1})),
                  "warning": "No validation events: recall and anomaly ranking cannot be validated."
                  if not np.any(ids[cal_end:val_end] >= 0) else None})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "audit.json", audit)
    best, trials, failures = {}, [], []
    for model_number, model in enumerate(model_specs(config), 1):
        family = model_name(model)
        description = {"family": family, "window": model.window,
                       "horizons": list(model.horizons), "alpha": model.alpha}
        try:
            start_time = time.perf_counter()
            model.fit(values, observed, train_end)
            # Only training/calibration/validation values are exposed during search.
            predictions = model.predict_aligned(values[:val_end])
            forecast_seconds = time.perf_counter() - start_time
            errors, local = prepare_errors(values[:val_end], predictions, observed[:val_end], model, floor)
            valid_cal = np.isfinite(errors[train_end:cal_end]).all(axis=1)
            calibration = errors[train_end:cal_end][valid_cal]
            if len(calibration) < config["min_calibration_rows"]:
                raise ValueError("Too few calibration observations")
            valid_val = observed[cal_end:val_end] & np.isfinite(predictions[cal_end:val_end]).all(axis=1)
            if not valid_val.any():
                raise ValueError("No scoreable validation observations")
            mae = float(np.mean(np.abs(values[cal_end:val_end][valid_val] -
                                      predictions[cal_end:val_end, 0][valid_val])))
            for adaptive in config["adaptation"]:
                for quantile in config["threshold_quantiles"]:
                    for persistence in config["persistence"]:
                        detector = Detector.calibrate(calibration, quantile, adaptive, config,
                                                      1e-6 if local is not None else floor, persistence)
                        begin = time.perf_counter()
                        replay = detector.replay(values, predictions, observed, cal_end, val_end, local)
                        elapsed = time.perf_counter() - begin
                        metrics = evaluate(stamps.iloc[cal_end:val_end].reset_index(drop=True),
                                           ids[cal_end:val_end], replay["flag"], np.isfinite(replay["score"]))
                        params = {**description, "adaptive": adaptive, "quantile": quantile,
                                  "persistence": persistence, "threshold": detector.threshold}
                        trial = {**params, **metrics, "forecast_mae": mae,
                                 "fit_and_validation_forecast_seconds": forecast_seconds,
                                 "validation_replay_seconds": elapsed}
                        trials.append(trial)
                        complexity = model.window * len(model.horizons) if model.kind == "ridge" else model.window
                        key = selection_key(metrics, mae, complexity)
                        if family not in best or key > best[family]["key"]:
                            best[family] = {"key": key, "params": params, "metrics": metrics,
                                            "model": model, "detector": copy.deepcopy(detector),
                                            "floor": floor, "validation_replay": replay}
        except (ValueError, np.linalg.LinAlgError) as error:
            failures.append({**description, "error": str(error)})
        if model_number % 5 == 0:
            pd.DataFrame(trials).to_csv(output / "experiment_log.csv", index=False)
            print(f"  {model_number} forecast configurations checked; {len(trials)} detector trials", flush=True)
    pd.DataFrame(trials).to_csv(output / "experiment_log.csv", index=False)
    write_json(output / "candidate_failures.json", failures)
    if len(best) != 4:
        raise RuntimeError(f"Expected four method families, obtained {list(best)}; inspect candidate_failures.json")
    winner = max(best, key=lambda family: best[family]["key"])
    write_json(output / "selected.json", {"winner": winner, "selection": "validation only",
               "methods": {name: {"parameters": item["params"], "validation": item["metrics"]}
                           for name, item in best.items()}})
    records = []
    # Test data is first forecast/scored after validation selection is complete.
    for family, item in best.items():
        model, detector = item["model"], item["detector"]
        start_time = time.perf_counter()
        predictions = model.predict_aligned(values)
        _, local = prepare_errors(values, predictions, observed, model, floor)
        replay = detector.replay(values, predictions, observed, val_end, len(values), local)
        seconds = time.perf_counter() - start_time
        metrics = evaluate(stamps.iloc[val_end:].reset_index(drop=True), ids[val_end:],
                           replay["flag"], np.isfinite(replay["score"]))
        scored = np.isfinite(replay["score"])
        mae = float(np.mean(np.abs(values[val_end:][scored] - predictions[val_end:, 0][scored]))) if scored.any() else None
        record = {"series": relative, "family": family, "selected_on_validation": family == winner,
                  **metrics, "forecast_mae": mae, "test_processing_seconds": seconds,
                  "test_points_per_processing_second": len(replay["score"]) / max(seconds, 1e-9)}
        records.append(record)
        frame = prediction_frame(stamps, values, observed, predictions, replay, model, detector.threshold, val_end)
        frame.to_csv(output / f"test_{family}.csv", index=False)
        write_json(output / f"metrics_{family}.json", record)
        if family == winner:
            frame.to_csv(output / "predictions.csv", index=False)
            validation_frame = prediction_frame(stamps, values, observed, predictions,
                                                item["validation_replay"], model,
                                                detector.threshold, cal_end)
            validation_frame.to_csv(output / "validation_predictions.csv", index=False)
            chart(validation_frame, output / "validation_chart.html", f"{relative}: validation selection",
                  ids[cal_end:val_end], config["plot_max_points"])
            chart(frame, output / "chart.html", f"{relative}: {family} (untouched local test)",
                  ids[val_end:], config["plot_max_points"])
            keep = max_window + max_horizon
            joblib.dump({"model": model, "detector": detector, "raw_floor": floor,
                         "history_values": values[-keep:], "history_observed": observed[-keep:],
                         "history_timestamps": stamps.iloc[-keep:].astype(str).tolist(),
                         "parameters": item["params"], "series": relative,
                         "note": "State after local test replay; coefficients fitted on training only"},
                        output / "model.joblib")
    return records


def predict_file(args, config):
    if not args.artifact:
        raise ValueError("--predict requires --artifact (load only your trusted model.joblib)")
    # joblib/pickle is executable serialization: only load artifacts you trust.
    bundle = joblib.load(project_path(args.artifact))
    stamps, values, observed, audit = load_series(project_path(args.predict), config["max_missing_fraction"])
    old_stamps = pd.to_datetime(bundle["history_timestamps"], utc=True)
    if stamps.iloc[0] <= old_stamps[-1]:
        raise ValueError("Prediction CSV must begin after the saved state's last timestamp")
    joined = pd.Series(np.r_[bundle["history_values"], values]).ffill().to_numpy()
    joined_observed = np.r_[bundle["history_observed"], observed]
    offset = len(bundle["history_values"])
    model, detector = bundle["model"], bundle["detector"]
    predictions = model.predict_aligned(joined)
    _, local = prepare_errors(joined, predictions, joined_observed, model, bundle["raw_floor"])
    begin = time.perf_counter()
    replay = detector.replay(joined, predictions, joined_observed, offset, len(joined), local)
    audit["replay_seconds"] = time.perf_counter() - begin
    all_stamps = pd.Series(old_stamps.tolist() + stamps.tolist())
    frame = prediction_frame(all_stamps, joined, joined_observed, predictions, replay, model, detector.threshold, offset)
    output = new_output(config, "predict")
    frame.to_csv(output / "predictions.csv", index=False)
    chart(frame, output / "chart.html", "New data: saved model and frozen selection", max_points=config["plot_max_points"])
    keep = len(bundle["history_values"])
    bundle.update(history_values=joined[-keep:], history_observed=joined_observed[-keep:],
                  history_timestamps=all_stamps.iloc[-keep:].astype(str).tolist())
    joblib.dump(bundle, output / "model.joblib")
    write_json(output / "audit.json", audit)
    print(f"Predictions written to {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")


def new_output(config, prefix):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = project_path(config["output_dir"]) / f"{prefix}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def main():
    parser = argparse.ArgumentParser(description="Sentinel: four CPU methods, validation search, untouched local test")
    parser.add_argument("--config", default="config.json", help="Project-relative JSON configuration")
    parser.add_argument("--check-data", action="store_true", help="Validate files/splits only; no fitting")
    parser.add_argument("--download-data", action="store_true", help="Download pinned public NAB data only; no fitting")
    parser.add_argument("--only", help="Series glob, e.g. 'realKnownCause/*temperature*'")
    parser.add_argument("--limit", type=int, help="Limit number of files for a first manual run")
    parser.add_argument("--predict", help="Unlabeled CSV continuing a saved signal")
    parser.add_argument("--artifact", help="Trusted model.joblib from the same signal")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    np.random.seed(config["seed"])
    if args.download_data:
        manifest = download_nab(project_path("data/nab"))
        print(f"Downloaded {len(manifest['files'])} pinned files; no fitting performed")
        return
    if args.predict:
        predict_file(args, config)
        return
    root, paths, labels = find_data(config, args.only, args.limit)
    splits = (json.loads(project_path(config["split_manifest"]).read_text(encoding="utf-8"))
              if config.get("split_manifest") else None)
    if splits is not None and any(p.relative_to(root).as_posix() not in splits for p in paths):
        raise ValueError("split_manifest must contain every selected relative CSV key")
    if args.check_data:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            stamps, values, observed, audit = load_series(path, config["max_missing_fraction"])
            split_points(len(values), config["split_fractions"], splits[relative] if splits else None)
            event_ids(stamps, labels[relative])
            print(f"OK {relative}: {len(values)} rows; {audit['missing_rows']} missing; "
                  f"{audit['irregular_intervals']} irregular intervals")
        print("Data checks completed; no fitting performed")
        return
    output = new_output(config, "experiment")
    write_json(output / "config_used.json", config)
    write_json(output / "environment.json", {"python": platform.python_version(),
               "packages": {name: importlib.metadata.version(name) for name in
                            ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "plotly")}})
    records, failures = [], []
    for number, path in enumerate(paths, 1):
        relative = path.relative_to(root).as_posix()
        token = path.stem + "_" + hashlib.sha256(relative.encode()).hexdigest()[:8]
        print(f"[{number}/{len(paths)}] {relative}: fitting and validating candidates", flush=True)
        try:
            records.extend(search_series(path, relative, labels[relative], config, output / "series" / token,
                                         splits[relative] if splits else None))
        except Exception as error:
            failures.append({"series": relative, "error": str(error), "traceback": traceback.format_exc()})
            print(f"FAILED {relative}: {error}", flush=True)
        write_json(output / "failures.json", failures)
        if records:
            summary_report(output, records)
    if not records:
        raise RuntimeError(f"No completed series. Inspect {output / 'failures.json'}")
    print(f"Finished {len(records)//4}/{len(paths)} series. Report: {output / 'report.html'}")
    if failures:
        raise SystemExit("Some series failed; review failures.json before using results")
