"""v4: frozen v2 control, next-fold policy comparisons, stress tests, then test once."""
import copy
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from .control_v2 import select_v2
from .data import event_ids, load_series
from .experiment_metrics import choose_policy, metrics, paired_interval
from .models import Forecaster
from .policies import EvidenceDetector, fit_policy, policy_specs, replay_policy
from .reporting import chart, prediction_frame, summary_report, write_json
from .selection_policy import historical_pool, load_trials, prefixes
from .stress import stress_stream
from .validation import (develop_series, fit_fold, from_parameters, pooled_by_group,
                         prepare_development, prepare_errors)


def validate_settings(settings):
    if settings["alert_budget_per_10000"] < 0 or settings["cusum_drift"] <= 0:
        raise ValueError("Invalid alert budget or CUSUM drift allowance")
    if not settings["rate_quantiles"] or any(not 0 < q <= 1 for q in settings["rate_quantiles"]):
        raise ValueError("Calibration quantiles must be in (0, 1]")
    if not 0 < settings["cusum_reference_quantile"] < 1 or not 0 < settings["confirmation_percentile"] < 1:
        raise ValueError("Invalid evidence quantiles")
    if not 0 < settings["confidence_level"] < 1 or settings["bootstrap_repetitions"] < 100:
        raise ValueError("Use at least 100 bootstrap samples and a valid confidence level")
    if settings["minimum_f1_gain"] < 0 or settings["maximum_recall_drop"] < 0:
        raise ValueError("Invalid adoption margins")
    if min(settings["minimum_policy_streams"], settings["minimum_policy_events"], settings["runtime_windows"]) < 1:
        raise ValueError("Invalid minimum evidence or runtime sample count")
    if settings["secondary_window"] < 2 or settings["injection_context"] < 2:
        raise ValueError("Invalid pattern or injection context window")
    if not settings["injection_seeds"] or len(settings["injection_seeds"]) != len(settings["injection_lengths"]):
        raise ValueError("Injection seeds and durations must have matching nonempty lists")
    if any(n < 2 for n in settings["injection_lengths"]):
        raise ValueError("Injection durations must be at least two readings")


def fit_bundle(stream, params, fold, config, specs):
    train_end, cal_end, end = stream["folds"][fold]
    model, predictions, _, local, _, _ = fit_fold(stream, from_parameters(params), fold, config)
    values, observed = stream["values"][:end], stream["observed"][:end]
    floor = stream["references"][fold]["floor"]
    errors, _, _ = prepare_errors(values, predictions, observed, model, floor)
    secondary_model, secondary_scores, secondary_error = None, None, None
    if any(spec["confirm"] for spec in specs):
        normal = stream["ids"][:train_end] < 0 if config["use_training_labels"] else None
        try:
            secondary_model = Forecaster("pattern", config["v4"]["secondary_window"], (1,)).fit(
                values[:train_end], observed[:train_end], train_end, normal, config)
            secondary_scores = secondary_model.auxiliary_scores(values)
        except ValueError as error:
            secondary_model = None
            secondary_error = str(error)
    detectors, audits, failures = {}, {}, []
    for spec in specs:
        try:
            if spec["confirm"] and secondary_scores is None:
                raise ValueError(f"Secondary detector unavailable: {secondary_error}")
            secondary = secondary_scores[train_end:cal_end] if spec["confirm"] else None
            detector, audit = fit_policy(errors[train_end:cal_end], params,
                1e-3 if model.kind == "median" else floor, config, spec, secondary)
            detectors[spec["name"]], audits[spec["name"]] = detector, audit
        except ValueError as error:
            if spec["name"] == "control_v2":
                raise
            failures.append({"policy": spec["name"], "error": str(error)})
    return {"model": model, "secondary_model": secondary_model, "secondary_scores": secondary_scores,
            "predictions": predictions, "local": local, "floor": floor, "detectors": detectors,
            "calibration_audit": audits, "policy_failures": failures}


def runtime_sample(bundle, detector, values, observed, start, settings):
    """Actual one-reading calls including feature construction, prediction and state updates."""
    model, secondary_model = bundle["model"], bundle["secondary_model"]
    if not isinstance(detector, EvidenceDetector) or detector.confirmation_level is None:
        secondary_model = None
    state = copy.deepcopy(detector)
    keep = max(model.history_length+max(model.horizons)+1,
               secondary_model.history_length if secondary_model is not None else 0)
    samples = []
    for t in range(start, min(len(values), start+settings["runtime_windows"])):
        before = time.perf_counter()
        left = max(0, t-keep)
        segment, seen = values[left:t+1], observed[left:t+1]
        predictions = model.predict_aligned(segment)
        _, local, _ = prepare_errors(segment, predictions, seen, model, bundle["floor"])
        secondary = secondary_model.auxiliary_scores(segment) if secondary_model is not None else None
        replay_policy(state, segment, predictions, seen, len(segment)-1, len(segment), local, secondary)
        samples.append((time.perf_counter()-before)*1000)
    return {"runtime_sample_windows": len(samples), "per_window_p50_ms": float(np.median(samples)),
            "per_window_p95_ms": float(np.quantile(samples, .95)),
            "runtime_scope": "feature construction + forecast + scoring/state; excludes file I/O and fitting"}


def record_failure(failures, output, **item):
    failures.append(item)
    write_json(output / "failures.json", failures)


def run_controlled(config, data_root, paths, labels, splits, output, control_config):
    """No old test results are inputs. All adoption choices precede every test replay."""
    settings = config["v4"]
    validate_settings(settings)
    # The predictor grid and v2 selection rules are pinned; data locations remain portable overrides.
    control = {**control_config, "v4": settings, "extra_models": [],
               "cpu_threads": config["cpu_threads"], "seed": config["seed"],
               "pattern_reference_size": config.get("pattern_reference_size", 512)}
    for key in ("data_dir", "labels_file", "output_dir", "include", "split_manifest"):
        control[key] = config[key]
    specs = list(policy_specs())
    control_digest = hashlib.sha256(json.dumps(control_config, sort_keys=True).encode()).hexdigest()
    write_json(output / "control_config.json", control_config)
    write_json(output / "experiment_protocol.json", {
        "control_commit": "9a76ee2", "control": "v2 predictor grid and selection rule",
        "control_config_canonical_sha256": control_digest,
        "control_config_matches_original_v2": control_digest == "6a5d2a6ab833d578815233c538b30266ffb565a1e7e751db85fd7362612971af",
        "policies": specs, "adoption": "next-fold real events; historical peer cutoff; paired uncertainty",
        "calibration": "no calibration or test labels; split blocks for revised policies",
        "synthetic": "post-freeze stress tests only; no fitting or adoption on injections",
        "test_status": "Previously inspected public test; exploratory final report"})
    streams, failures, policy_rows = [], [], []
    write_json(output / "failures.json", failures)
    for number, path in enumerate(paths, 1):
        relative = path.relative_to(data_root).as_posix()
        print(f"[v2 control development {number}/{len(paths)}] {relative}", flush=True)
        stream = prepare_development(path, relative, labels[relative], control,
                                     splits[relative] if splits else None)
        token = path.stem+"_"+hashlib.sha256(relative.encode()).hexdigest()[:8]
        streams.append(develop_series(stream, control, output / "series" / token))
    trials = [load_trials(stream) for stream in streams]
    histories = [prefixes(stream, *trial) for stream, trial in zip(streams, trials)]
    for index, stream in enumerate(streams):
        print(f"[Next-fold policy development {index+1}/{len(streams)}] {stream['relative']}", flush=True)
        choices = []
        for fold in range(1, len(stream["folds"])):
            past = histories[index][fold-1]
            pool = historical_pool(past, past["available_until"], histories)
            selected, family = select_v2(past, pool, control)
            params = selected[family]["parameters"]
            bundle = fit_bundle(stream, params, fold, control, specs)
            _, start, stop = stream["folds"][fold]
            for name, detector in bundle["detectors"].items():
                replay = replay_policy(copy.deepcopy(detector), stream["values"], bundle["predictions"],
                    stream["observed"], start, stop, bundle["local"], bundle["secondary_scores"])
                row = {"series": stream["relative"], "group": stream["group"], "fold": fold,
                    "policy": name, "model_family": family,
                    "validation_end": str(stream["stamps"].iloc[stop-1]),
                    **metrics(stream["stamps"].iloc[start:stop].reset_index(drop=True), stream["ids"][start:stop], replay)}
                policy_rows.append(row)
            for failed in bundle["policy_failures"]:
                record_failure(failures, output, series=stream["relative"], phase="policy_candidate", fold=fold, **failed)
            choices.append({"fold": fold, "selection_available_until": str(past["available_until"]),
                            "parameters": params, "calibration": bundle["calibration_audit"]})
        write_json(stream["output"] / "next_fold_choices.json", choices)
        pd.DataFrame(policy_rows).to_csv(output / "development_policy_comparison.csv", index=False)
    pools, frozen = pooled_by_group(streams), {}
    for stream in streams:
        selected, family = select_v2(stream, pools, control)
        winner, audit = choose_policy(stream, policy_rows, specs, settings, config["seed"])
        frozen[stream["relative"]] = {"policy": winner, "parameters": selected[family]["parameters"],
                                     "selection_source": selected[family]["selection_source"], "adoption_audit": audit}
        write_json(stream["output"] / "selected.json", frozen[stream["relative"]])
    # This file must exist before injection results and before the first test reading is replayed.
    write_json(output / "selection_frozen_before_test.json", frozen)
    stress_rows, stress_skips, records, runtime_rows = [], [], [], []
    for number, stream in enumerate(streams, 1):
        relative = stream["relative"]
        selected = frozen[relative]
        print(f"[Frozen replay {number}/{len(streams)}] {relative}: {selected['policy']}", flush=True)
        bundle = fit_bundle(stream, selected["parameters"], len(stream["folds"])-1, control, specs)
        if selected["policy"] not in bundle["detectors"]:
            raise RuntimeError(f"Frozen policy cannot be fitted for {relative}; refusing a test-driven fallback")
        injected, skipped = stress_stream(stream, bundle, settings)
        stress_rows.extend(injected)
        if skipped:
            stress_skips.append({"series": relative, "reason": skipped})
        pd.DataFrame(stress_rows).to_csv(output / "synthetic_stress.csv", index=False)
        write_json(output / "synthetic_skips.json", stress_skips)
        stamps, values, observed, audit = load_series(stream["path"], control["max_missing_fraction"])
        if audit["sha256"] != stream["audit"]["sha256"]:
            raise RuntimeError("Data changed after development")
        start = stream["test_start"]
        ids = event_ids(stamps, labels[relative])
        predictions = bundle["model"].predict_aligned(values)
        _, local, _ = prepare_errors(values, predictions, observed, bundle["model"], bundle["floor"])
        secondary = bundle["secondary_model"].auxiliary_scores(values) if bundle["secondary_model"] is not None else None
        for name, initial in bundle["detectors"].items():
            detector = copy.deepcopy(initial)
            _, validation_start, _ = stream["folds"][-1]
            validation = replay_policy(detector, values, predictions, observed, validation_start, start, local, secondary)
            runtime = runtime_sample(bundle, detector, values, observed, start, settings)
            runtime_rows.append({"series": relative, "policy": name, **runtime})
            begin = time.perf_counter()
            replay = replay_policy(detector, values, predictions, observed, start, len(values), local, secondary)
            seconds = time.perf_counter()-begin
            row = {"series": relative, "group": stream["group"], "family": name,
                "policy": name, "selected_on_validation": False, "model_family": selected["parameters"]["family"],
                "selection_source": "frozen_next_fold_development", "batch_scoring_seconds": seconds,
                **runtime, **metrics(stamps.iloc[start:].reset_index(drop=True), ids[start:], replay)}
            records.append(row)
            frame = prediction_frame(stamps, values, observed, predictions, replay, bundle["model"], detector.threshold, start)
            frame.to_csv(stream["output"] / f"test_{name}.csv", index=False)
            write_json(stream["output"] / f"metrics_{name}.json", row)
            if name == selected["policy"]:
                records.append({**row, "family": "selected_system", "policy": "selected_system", "selected_on_validation": True})
                frame.to_csv(stream["output"] / "predictions.csv", index=False)
                chart(frame, stream["output"] / "chart.html", f"{relative}: frozen {name}", ids[start:], config["plot_max_points"])
                validation_frame = prediction_frame(stamps, values, observed, predictions, validation,
                    bundle["model"], detector.threshold, validation_start)
                validation_frame.to_csv(stream["output"] / "validation_predictions.csv", index=False)
                keep = max(bundle["model"].history_length+max(bundle["model"].horizons)+1,
                    bundle["secondary_model"].history_length if bundle["secondary_model"] is not None else 0)
                joblib.dump({"format_version": 4, "model": bundle["model"], "detector": detector,
                    "secondary_model": bundle["secondary_model"] if selected["policy"].startswith(("E_", "F_")) else None,
                    "raw_floor": bundle["floor"], "history_values": values[-keep:],
                    "history_observed": observed[-keep:], "history_timestamps": stamps.iloc[-keep:].astype(str).tolist(),
                    "parameters": selected["parameters"], "policy": name, "series": relative,
                    "note": "Frozen development choice; label-free state after causal test replay"}, stream["output"] / "model.joblib")
        summary_report(output, records)
        pd.DataFrame(runtime_rows).to_csv(output / "runtime_per_window.csv", index=False)
    test_intervals = {}
    for spec in specs:
        name = spec["name"]
        if name != "control_v2":
            test_intervals[name] = {unit: paired_interval(records, name, "control_v2", settings, config["seed"], unit)
                                    for unit in ("series", "group")}
    selected_rows = [{**r, "policy": "selected_system"} if r["family"] == "selected_system" else r for r in records]
    test_intervals["selected_system"] = {unit: paired_interval(selected_rows, "selected_system", "control_v2",
        settings, config["seed"], unit) for unit in ("series", "group")}
    write_json(output / "test_uncertainty_report_only.json", test_intervals)
    write_json(output / "run_status.json", {"completed_series": len(streams), "test_used_for_selection": False,
        "synthetic_used_for_selection": False, "candidate_failures": failures,
        "normal_exposure_note": "Nominal cadence caps gaps; no event-only bootstrap; no fresh independent test claim"})
    print(f"Finished v4: {len(streams)}/{len(paths)} streams. Report: {output/'report.html'}", flush=True)
