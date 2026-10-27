"""Event metrics and paired uncertainty; normal streams remain in every resample."""
import numpy as np
import pandas as pd

from .evaluation import evaluate
from .validation import aggregate


def metrics(stamps, ids, replay):
    eligible = np.isfinite(replay["score"])
    result = evaluate(stamps.reset_index(drop=True), ids, replay["flag"], eligible)
    differences = stamps.diff().dt.total_seconds().to_numpy()
    positive = differences[np.isfinite(differences) & (differences > 0)]
    cadence = float(np.median(positive)) if len(positive) else 0.
    exposure = np.where(np.isfinite(differences), np.clip(differences, 0, cadence), cadence)
    normal_seconds = float(exposure[eligible & (ids < 0)].sum())
    result.update(false_alerts_per_10000_normal_readings=10*result["false_alerts_per_1000_normal_points"],
                  normal_exposure_seconds=normal_seconds,
                  false_alerts_per_normal_day=(result["false_alerts"]*86400/normal_seconds if normal_seconds else None))
    flags = replay["flag"] & eligible
    starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    delays, found = [], set()
    for t in starts:
        event = int(ids[t])
        if event >= 0 and event not in found:
            found.add(event)
            first = np.flatnonzero(eligible & (ids == event))[0]
            delays.append((stamps.iloc[t]-stamps.iloc[first]).total_seconds())
    result.update(median_delay_seconds=float(np.median(delays)) if delays else None,
                  p95_delay_seconds=float(np.quantile(delays, .95)) if delays else None)
    return result


def paired_interval(rows, candidate, control, settings, seed, unit="series"):
    frame = pd.DataFrame(rows)
    columns = ["detected_events", "false_alerts", "missed_events"]
    left = frame[frame.policy.eq(candidate)].groupby(unit)[columns].sum()
    right = frame[frame.policy.eq(control)].groupby(unit)[columns].sum()
    if set(left.index) != set(right.index):
        return {"lower": None, "upper": None, "units": 0, "warning": "Unpaired policy records"}
    left, right = left.sort_index().to_numpy(), right.sort_index().to_numpy()
    n = len(left)
    if n < 2:
        return {"lower": None, "upper": None, "units": n, "warning": "Too few independent units"}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n, size=(settings["bootstrap_repetitions"], n))
    def f1(counts):
        tp, fp, fn = counts.T
        return 2*tp/np.maximum(1, 2*tp+fp+fn)
    delta = f1(left[samples].sum(axis=1))-f1(right[samples].sum(axis=1))
    tail = (1-settings["confidence_level"])/2
    lo, hi = np.quantile(delta, [tail, 1-tail])
    return {"lower": float(lo), "upper": float(hi), "units": n,
            "confidence_level": settings["confidence_level"], "resampling_unit": unit}


def choose_policy(target, rows, specs, settings, seed):
    cutoff = target["stamps"].iloc[-1]
    available = [r for r in rows if pd.Timestamp(r["validation_end"]) <= cutoff and
                 (r["series"] == target["relative"] or r["group"] != target["group"])]
    control = [r for r in available if r["policy"] == "control_v2"]
    keys = {(r["series"], r["fold"]) for r in control}
    base = aggregate(control)
    audits, accepted = {}, []
    for spec in specs:
        name = spec["name"]
        if name == "control_v2":
            continue
        candidate = [r for r in available if r["policy"] == name]
        paired = {(r["series"], r["fold"]) for r in candidate} == keys
        summary = aggregate(candidate)
        interval = paired_interval(available, name, "control_v2", settings, seed)
        groups = paired_interval(available, name, "control_v2", settings, seed, "group")
        reasons = []
        if not paired:
            reasons.append("Incomplete paired development folds")
        if len({r["series"] for r in control}) < settings["minimum_policy_streams"]:
            reasons.append("Insufficient historical streams")
        if base["true_events"] < settings["minimum_policy_events"]:
            reasons.append("Insufficient real development event opportunities")
        if summary["event_f1"]-base["event_f1"] < settings["minimum_f1_gain"]:
            reasons.append("F1 gain below declared margin")
        if summary["event_recall"] < base["event_recall"]-settings["maximum_recall_drop"]:
            reasons.append("Recall below allowed floor")
        if 10*summary["false_alerts_per_1000_normal_points"] > settings["alert_budget_per_10000"]:
            reasons.append("Development false-alert budget exceeded")
        if interval["lower"] is None or interval["lower"] <= 0:
            reasons.append("Paired stream interval does not establish a gain")
        if groups["lower"] is None or groups["lower"] <= 0:
            reasons.append("Source-group sensitivity does not establish a gain")
        parent = {"D_cusum": "C_spike_calibration", "E_confirmation": "B_calibration",
                  "F_confirmation_bypass": "C_spike_calibration", "G_single_exceedance": "B_calibration"}.get(name)
        incremental = None
        if parent:
            incremental = paired_interval(available, name, parent, settings, seed)
            parent_keys = {(r["series"], r["fold"]) for r in available if r["policy"] == parent}
            if parent_keys != keys:
                reasons.append("Simpler-parent paired folds are incomplete")
            if incremental["lower"] is None or incremental["lower"] <= 0:
                reasons.append("Increment over simpler parent is uncertain")
        audits[name] = {"development": summary, "stream_interval": interval, "source_group_interval": groups,
                        "incremental_interval": incremental, "rejection_reasons": reasons}
        if not reasons:
            accepted.append(name)
    simpler = [name for name in accepted if not name.startswith(("E_", "F_"))]
    eligible = simpler or accepted  # Verifiers are considered only when simpler policies have not passed.
    winner = max(eligible, key=lambda name: audits[name]["development"]["event_f1"]) if eligible else "control_v2"
    return winner, {"chosen_policy": winner, "control": base, "policies": audits,
        "historical_sources": sorted({(r["series"], r["validation_end"]) for r in control}),
        "rule": "Real development only; paired stream/source-group gains; keep v2 if uncertain"}
