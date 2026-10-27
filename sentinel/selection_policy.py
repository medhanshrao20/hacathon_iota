"""Choose a selection policy by replaying earlier-to-later development folds."""
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .validation import aggregate, select_methods


# Small declared policy set; none of these settings is chosen from test metrics.
POLICIES = {
    "original_transfer": {},
    "local_prior": {"selection_prior_events": .5, "local_false_alert_penalty": .2},
    "balanced_prior": {"selection_prior_events": 2., "local_false_alert_penalty": .2},
    "false_alert_control": {"local_false_alert_penalty": .5},
    "three_event_evidence": {"min_selection_events": 3, "selection_prior_events": 2.,
                             "local_false_alert_penalty": .2},
    "two_event_evidence": {"min_selection_events": 2, "min_event_folds": 1,
                           "selection_prior_events": 2., "local_false_alert_penalty": .2},
}
for family in ("recent_mean", "seasonal_naive", "rolling_median_mad", "ridge_single",
               "ridge_multi", "boosted_forecast", "pattern_hybrid"):
    POLICIES[f"fixed_{family}"] = {"fixed_family": family}


def load_trials(stream):
    rows = pd.read_csv(stream["output"] / "experiment_log.csv").to_dict("records")
    params, by_fold = {}, defaultdict(dict)
    for row in rows:
        key = row["candidate"]
        params[key] = {name: row[name] for name in
                       ("kind", "family", "window", "alpha", "use_seasonality", "adaptive",
                        "quantile", "method", "incident_profile")}
        params[key].update(horizons=ast.literal_eval(row["horizons"]),
                           persistence=ast.literal_eval(row["persistence"]))
        by_fold[int(row["fold"])][key] = row
    return params, by_fold


def prefixes(stream, params, by_fold):
    result = []
    for count in range(1, len(stream["folds"])+1):
        available = set.intersection(*(set(by_fold[i]) for i in range(count)))
        summaries = {key: {"parameters": params[key],
                           "validation": aggregate([by_fold[i][key] for i in range(count)])}
                     for key in params if key in available}
        events, event_folds = set(), 0
        for _, start, stop in stream["folds"][:count]:
            found = set(stream["ids"][start:stop]) - {-1}
            events.update(found)
            event_folds += bool(found)
        result.append({**stream, "summaries": summaries, "selection_events": len(events),
                       "event_folds": event_folds,
                       "available_until": stream["stamps"].iloc[stream["folds"][count-1][2]-1]})
    return result


def historical_pool(target, cutoff, histories):
    grouped = defaultdict(list)
    for history in histories:
        if history[0]["group"] == target["group"]:
            continue
        eligible = [past for past in history if past["available_until"] <= cutoff]
        if eligible:
            for key, item in eligible[-1]["summaries"].items():
                grouped[key].append(item["validation"])
    return {target["relative"]: {key: aggregate(rows) for key, rows in grouped.items()}}


def tune_selection_policy(streams, config):
    """Select on next-fold decisions; never read any test prediction or label."""
    trials = [load_trials(stream) for stream in streams]
    histories = [prefixes(stream, *trial) for stream, trial in zip(streams, trials)]
    scores = {name: [] for name in POLICIES}
    failures = {name: 0 for name in POLICIES}
    for index, stream in enumerate(streams):
        for held in range(1, len(stream["folds"])):
            past = histories[index][held-1]
            pool = historical_pool(past, past["available_until"], histories)
            for name, overrides in POLICIES.items():
                selected, winner = select_methods(past, pool, {**config, **overrides})
                winner = overrides.get("fixed_family", winner)
                if winner not in selected:
                    failures[name] += 1
                    continue
                chosen = selected[winner]["candidate"]
                row = trials[index][1][held].get(chosen)
                if row is None:
                    failures[name] += 1
                else:
                    scores[name].append(row)
    records = {}
    for name, rows in scores.items():
        metrics = aggregate(rows)
        objective = (metrics["event_f1"] - .1*metrics["event_fold_f1_std"]
                     - .05*max(0, metrics["false_alerts_per_1000_normal_points"]
                               - config["false_alert_budget_per_1000"]))
        records[name] = {"overrides": POLICIES[name], "development": metrics,
                         "failed_next_fold_fits": failures[name], "objective": objective}
    eligible = [name for name in records if not failures[name]]
    if not eligible:
        raise ValueError("No selection policy completed next-fold development replay")
    # Declared order favors the original policy on exact ties.
    winner = max(eligible, key=lambda name: records[name]["objective"])
    return {**config, **POLICIES[winner]}, {"chosen_policy": winner, "policies": records,
            "protocol": "Choose using fold 0 -> 1 and folds 0,1 -> 2; historical other-source peers only"}


def audit_saved_development(directory, streams, config):
    """Read only verified development artifacts; never read test outputs."""
    if json.loads((directory / "config_used.json").read_text()) != config:
        raise ValueError("Audit configuration differs from the saved development run")
    hashes = json.loads((directory / "source_hashes.json").read_text())
    root = Path(__file__).resolve().parents[1]
    for name in ("data", "models", "detection", "evaluation", "validation"):
        relative = f"sentinel/{name}.py"
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != hashes[relative]:
            raise ValueError(f"Scoring source changed: {relative}")
    for stream in streams:
        token = stream["path"].stem + "_" + hashlib.sha256(stream["relative"].encode()).hexdigest()[:8]
        stream["output"] = directory / "series" / token
        audit = json.loads((stream["output"] / "audit.json").read_text())
        if audit["sha256"] != stream["audit"]["sha256"] or audit["folds"] != stream["audit"]["folds"]:
            raise ValueError(f"Data or boundaries changed: {stream['relative']}")
    return tune_selection_policy(streams, config)[1]
