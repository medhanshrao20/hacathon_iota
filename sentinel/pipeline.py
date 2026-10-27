"""Two passes: finish development selection for ALL streams before test replay."""
import argparse
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
from threadpoolctl import threadpool_limits

from .data import download_nab, event_ids, load_series
from .evaluation import evaluate
from .reporting import chart, prediction_frame, summary_report, write_json
from .validation import (calibrate, chronological_folds, develop_series, fit_fold, from_parameters,
                         pooled_by_group, prepare_development, prepare_errors, select_methods)

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_config(config):
    fractions = config["split_fractions"]
    if len(fractions) != 4 or any(x <= 0 for x in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("Four positive split fractions must sum to one")
    for key in ("windows", "alphas", "threshold_quantiles", "horizon_sets", "adaptation",
                "persistence", "threshold_methods", "incident_profiles"):
        if not config[key]:
            raise ValueError(f"Empty setting: {key}")
    if any(not isinstance(w, int) or w < 2 for w in config["windows"]):
        raise ValueError("windows must contain integers >= 2")
    if any(a <= 0 for a in config["alphas"]):
        raise ValueError("alphas must be positive")
    if any(not 0 < q < 1 for q in config["threshold_quantiles"]):
        raise ValueError("Quantiles must be between zero and one")
    if [1] not in config["horizon_sets"] or not any(len(h) > 1 for h in config["horizon_sets"]):
        raise ValueError("Include single and multi-horizon configurations")
    for horizons in config["horizon_sets"]:
        if not horizons or horizons[0] != 1 or len(set(horizons)) != len(horizons) or any(
                not isinstance(h, int) or h < 1 for h in horizons):
            raise ValueError("Horizon sets must start at 1 and contain unique positive integers")
    if any(not 1 <= required <= width for required, width in config["persistence"]):
        raise ValueError("Invalid persistence rule")
    if set(config["threshold_methods"]) - {"mad", "empirical"}:
        raise ValueError("Threshold methods are mad or empirical")
    if set(config["incident_profiles"]) - {"plain", "guarded"}:
        raise ValueError("Incident profiles are plain or guarded")
    if config["reference_size"] < 16 or config["reference_update_every"] < 1 or config["min_calibration_rows"] < 16:
        raise ValueError("Invalid reference settings")
    if not 0 <= config["max_missing_fraction"] < 1:
        raise ValueError("Invalid missing-data limit")
    if config["cv_folds"] < 2 or not 1 <= config["min_event_folds"] <= config["cv_folds"]:
        raise ValueError("At least two chronological folds are required")
    if config["min_selection_events"] < 1 or config["false_alert_budget_per_1000"] < 0:
        raise ValueError("Invalid selection requirements")
    if not isinstance(config["use_training_labels"], bool):
        raise ValueError("use_training_labels must explicitly be true or false")
    if config["max_seasonal_period"] < 3 or not 0 < config["min_seasonal_correlation"] <= 1:
        raise ValueError("Invalid seasonality settings")
    if not 0 < config["close_threshold_ratio"] < 1 or config["recovery_readings"] < 1 or config["cooldown_readings"] < 0:
        raise ValueError("Invalid incident recovery settings")
    if not 0 < config["adaptive_min_scale_fraction"] <= 1:
        raise ValueError("Invalid adaptive scale floor")
    if set(config.get("extra_models", [])) - {"boosted", "pattern"}:
        raise ValueError("extra_models supports boosted and pattern")
    if config.get("pattern_reference_size", 512) < 16 or config.get("boosting_iterations", 100) < 1:
        raise ValueError("Invalid new model sizes")


def find_data(config, pattern=None, limit=None):
    root = project_path(config["data_dir"])
    labels = json.loads(project_path(config["labels_file"]).read_text(encoding="utf-8"))
    paths = sorted(p for p in root.rglob("*.csv") if any(p.match(glob) for glob in config["include"]))
    if pattern:
        paths = [p for p in paths if fnmatch.fnmatch(p.relative_to(root).as_posix(), pattern)]
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError("No CSV files matched")
    for path in paths:
        if path.relative_to(root).as_posix() not in labels:
            raise ValueError(f"Missing labels key for {path.name}; unknown labels are not normal labels")
    return root, paths, labels


def frozen_selection(selected, winner):
    return {"winner": winner, "selection": "chronological development folds only; no test labels",
            "test_status": "Exploratory rerun of previously inspected public test",
            "methods": {name: {key: value for key, value in item.items() if key != "rank"}
                        for name, item in selected.items()}}


def evaluate_selected(stream, selected, winner, windows, config):
    """Selection is already frozen before this function is called."""
    output = stream["output"]
    records = []
    final_fold = len(stream["folds"]) - 1
    train_end, cal_end, test_start = stream["folds"][final_fold]
    for family, item in selected.items():
        params = item["parameters"]
        model, development_predictions, calibration, local, _, auxiliary = fit_fold(stream, from_parameters(params), final_fold, config)
        floor = stream["references"][final_fold]["floor"]
        detector = calibrate(calibration, local, floor, params, config)
        validation_replay = detector.replay(stream["values"], development_predictions,
                                             stream["observed"], cal_end, test_start, local, auxiliary)
        # Full observations and test label IDs enter only AFTER selection/fit.
        stamps, values, observed, audit = load_series(stream["path"], config["max_missing_fraction"])
        if audit["sha256"] != stream["audit"]["sha256"]:
            raise RuntimeError("Input changed between development and test; refusing mixed results")
        begin = time.perf_counter()
        predictions = model.predict_aligned(values)
        _, local, auxiliary = prepare_errors(values, predictions, observed, model, floor)
        replay = detector.replay(values, predictions, observed, test_start, len(values), local, auxiliary)
        seconds = time.perf_counter() - begin
        ids = event_ids(stamps, windows)
        metrics = evaluate(stamps.iloc[test_start:].reset_index(drop=True), ids[test_start:],
                           replay["flag"], np.isfinite(replay["score"]))
        eligible = np.isfinite(replay["score"])
        mae = float(np.mean(np.abs(values[test_start:][eligible]-predictions[test_start:, 0][eligible]))) if eligible.any() else None
        record = {"series": stream["relative"], "family": family, "selected_on_validation": family == winner,
                  "selection_source": item["selection_source"], **metrics, "forecast_mae": mae,
                  "test_processing_seconds": seconds,
                  "test_points_per_processing_second": len(replay["score"])/max(seconds, 1e-9)}
        records.append(record)
        frame = prediction_frame(stamps, values, observed, predictions, replay, model, detector.threshold, test_start)
        frame.to_csv(output / f"test_{family}.csv", index=False)
        write_json(output / f"metrics_{family}.json", record)
        if family == winner:
            frame.to_csv(output / "predictions.csv", index=False)
            validation_frame = prediction_frame(stream["stamps"], stream["values"], stream["observed"],
                                                 development_predictions, validation_replay, model,
                                                 detector.threshold, cal_end)
            validation_frame.to_csv(output / "validation_predictions.csv", index=False)
            chart(validation_frame, output / "validation_chart.html", f"{stream['relative']}: final CV fold",
                  stream["ids"][cal_end:test_start], config["plot_max_points"])
            chart(frame, output / "chart.html", f"{stream['relative']}: exploratory public test replay",
                  ids[test_start:], config["plot_max_points"])
            keep = model.history_length + max(model.horizons) + 1
            joblib.dump({"format_version": 3, "model": model, "detector": detector, "raw_floor": floor,
                         "history_values": values[-keep:], "history_observed": observed[-keep:],
                         "history_timestamps": stamps.iloc[-keep:].astype(str).tolist(),
                         "parameters": params, "series": stream["relative"],
                         "note": "Training-only coefficients; state after causal public test replay"},
                        output / "model.joblib")
    return records


def search_series(path, relative, windows, config, output, explicit_split=None):
    """Single-series convenience for synthetic tests; full CLI pools streams."""
    stream = prepare_development(path, relative, windows, config, explicit_split)
    develop_series(stream, config, output)
    selected, winner = select_methods(stream, {}, config)
    write_json(output / "selected.json", frozen_selection(selected, winner))
    return evaluate_selected(stream, selected, winner, windows, config)


def predict_file(args, config):
    if not args.artifact:
        raise ValueError("--predict requires a trusted --artifact model.joblib")
    bundle = joblib.load(project_path(args.artifact))
    if bundle.get("format_version") != 3:
        raise ValueError("Older model format: rerun training with the current implementation")
    stamps, values, observed, audit = load_series(project_path(args.predict), config["max_missing_fraction"])
    old_stamps = pd.to_datetime(bundle["history_timestamps"], utc=True)
    if stamps.iloc[0] <= old_stamps[-1]:
        raise ValueError("New timestamps must be later than the saved state")
    joined = pd.Series(np.r_[bundle["history_values"], values]).ffill().to_numpy()
    joined_observed = np.r_[bundle["history_observed"], observed]
    offset = len(bundle["history_values"])
    model, detector = bundle["model"], bundle["detector"]
    predictions = model.predict_aligned(joined)
    _, local, auxiliary = prepare_errors(joined, predictions, joined_observed, model, bundle["raw_floor"])
    replay = detector.replay(joined, predictions, joined_observed, offset, len(joined), local, auxiliary)
    all_stamps = pd.Series(old_stamps.tolist() + stamps.tolist())
    frame = prediction_frame(all_stamps, joined, joined_observed, predictions, replay, model, detector.threshold, offset)
    output = new_output(config, "predict")
    frame.to_csv(output / "predictions.csv", index=False)
    chart(frame, output / "chart.html", "New readings: saved model", max_points=config["plot_max_points"])
    keep = len(bundle["history_values"])
    bundle.update(history_values=joined[-keep:], history_observed=joined_observed[-keep:],
                  history_timestamps=all_stamps.iloc[-keep:].astype(str).tolist())
    joblib.dump(bundle, output / "model.joblib")
    write_json(output / "audit.json", audit)
    print(f"Predictions written to {output}")


def new_output(config, prefix):
    output = project_path(config["output_dir"]) / (prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    return output


def main():
    parser = argparse.ArgumentParser(description="Sentinel v3: nonlinear forecasts, pattern novelty, causal CV")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--check-data", action="store_true", help="Validate data and fold boundaries; no fitting")
    parser.add_argument("--download-data", action="store_true", help="Pinned data download only")
    parser.add_argument("--only", help="Relative CSV glob")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--selection-audit", help="Audit saved development folds only; no training or test replay")
    parser.add_argument("--predict", help="Unlabeled continuation CSV")
    parser.add_argument("--artifact", help="Trusted v3 model.joblib")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    # Bound BLAS/OpenMP parallelism; avoid oversubscribing small per-stream fits.
    thread_controller = threadpool_limits(limits=config.get("cpu_threads", 4))
    np.random.seed(config["seed"])
    if args.download_data:
        print(f"Downloaded {len(download_nab(project_path('data/nab'))['files'])} files; no training")
        return
    if args.predict:
        predict_file(args, config)
        return
    root, paths, labels = find_data(config, args.only, args.limit)
    splits = (json.loads(project_path(config["split_manifest"]).read_text(encoding="utf-8"))
              if config.get("split_manifest") else None)
    if splits is not None and any(p.relative_to(root).as_posix() not in splits for p in paths):
        raise ValueError("split_manifest must include every selected CSV")
    if args.check_data:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            stamps, values, _, audit = load_series(path, config["max_missing_fraction"])
            folds, test_start = chronological_folds(len(values), config, splits[relative] if splits else None)
            event_ids(stamps, labels[relative])
            print(f"OK {relative}: {len(values)} cleaned rows; folds {folds}; test starts {test_start}")
        print("Data checked; no fitting")
        return
    if args.selection_audit:
        from .selection_policy import audit_saved_development
        streams = [prepare_development(path, path.relative_to(root).as_posix(),
                    labels[path.relative_to(root).as_posix()], config,
                    splits[path.relative_to(root).as_posix()] if splits else None) for path in paths]
        audit = audit_saved_development(project_path(args.selection_audit), streams, config)
        directory = new_output(config, "selection_audit")
        write_json(directory / "selection_policy_development.json", audit)
        print(f"Development-only selection audit: {audit['chosen_policy']}; {directory}")
        return
    output = new_output(config, "experiment_v3")
    write_json(output / "config_used.json", config)
    write_json(output / "environment.json", {"python": platform.python_version(),
               "packages": {name: importlib.metadata.version(name) for name in
                            ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "plotly")}})
    sources = list((ROOT / "sentinel").glob("*.py")) + [ROOT / "run.py"]
    write_json(output / "source_hashes.json", {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sources})
    streams, failures, records = [], [], []
    for number, path in enumerate(paths, 1):
        relative = path.relative_to(root).as_posix()
        print(f"[Development {number}/{len(paths)}] {relative}", flush=True)
        try:
            stream = prepare_development(path, relative, labels[relative], config, splits[relative] if splits else None)
            token = path.stem + "_" + hashlib.sha256(relative.encode()).hexdigest()[:8]
            streams.append(develop_series(stream, config, output / "series" / token))
        except Exception as error:
            failures.append({"series": relative, "phase": "development", "error": str(error), "traceback": traceback.format_exc()})
            print(f"FAILED development: {error}", flush=True)
        write_json(output / "failures.json", failures)
    if not streams:
        raise RuntimeError(f"No completed development series. Inspect {output/'failures.json'}")
    pools = pooled_by_group(streams)
    frozen = []
    for stream in streams:
        selected, winner = select_methods(stream, pools, config)
        write_json(stream["output"] / "selected.json", frozen_selection(selected, winner))
        frozen.append((stream, selected, winner))
    write_json(output / "selection_frozen_before_test.json", {s["relative"]: frozen_selection(selected, winner)
                                                              for s, selected, winner in frozen})
    for number, (stream, selected, winner) in enumerate(frozen, 1):
        print(f"[Test replay {number}/{len(frozen)}] {stream['relative']} (winner already frozen: {winner})", flush=True)
        try:
            records.extend(evaluate_selected(stream, selected, winner, labels[stream["relative"]], config))
        except Exception as error:
            failures.append({"series": stream["relative"], "phase": "test", "error": str(error), "traceback": traceback.format_exc()})
            print(f"FAILED test replay: {error}", flush=True)
        write_json(output / "failures.json", failures)
        if records:
            summary_report(output, records)
    if not records:
        raise RuntimeError(f"No completed test series. Inspect {output/'failures.json'}")
    completed = len({record["series"] for record in records})
    print(f"Finished {completed}/{len(paths)} series. Report: {output/'report.html'}")
    if failures:
        raise SystemExit("Some series failed; inspect failures.json")
