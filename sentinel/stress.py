"""Post-selection synthetic stress cases; never fed into fitting or adoption."""
import copy

import numpy as np

from .experiment_metrics import metrics
from .policies import replay_policy
from .validation import prepare_errors


def stress_stream(stream, bundle, settings):
    model, secondary_model = bundle["model"], bundle["secondary_model"]
    values, observed = stream["values"], stream["observed"]
    _, start, stop = stream["folds"][-1]
    context = settings["injection_context"]
    width = max(96, context+max(settings["injection_lengths"])+32)
    clean = observed & (stream["ids"] < 0)
    bad = np.r_[0, np.cumsum(~clean)]
    starts = [i for i in range(start, stop-width+1) if bad[i+width] == bad[i]]
    if not starts:
        return [], "No sufficiently long clean development block"
    block = starts[0]
    keep = max(model.history_length+max(model.horizons)+1,
               secondary_model.history_length if secondary_model is not None else 0)
    left = max(0, block-keep)
    offset = block-left
    raw = values[left:block+width].copy()
    local_observed = observed[left:block+width].copy()
    stamps = stream["stamps"].iloc[block:block+width].reset_index(drop=True)
    before = raw[:offset+context]
    scale = max(1.4826*np.median(np.abs(before-np.median(before))), bundle["floor"])
    nonnegative = np.min(values[:stream["folds"][-1][0]][observed[:stream["folds"][-1][0]]]) >= 0
    initial = {}
    for name, detector in bundle["detectors"].items():
        state = copy.deepcopy(detector)
        replay_policy(state, values, bundle["predictions"], observed, start, block,
                      bundle["local"], bundle["secondary_scores"])
        initial[name] = state
    rows = []
    for case_index, (seed, duration) in enumerate(zip(settings["injection_seeds"], settings["injection_lengths"])):
        for kind in ("spike", "dip", "flatline", "level", "variance", "drift"):
            if kind == "flatline" and np.std(before) <= bundle["floor"]:
                continue  # A stuck reading on an already-flat signal is not a distinct anomaly.
            rng = np.random.default_rng(seed)
            changed = raw.copy()
            length = 1 if kind in ("spike", "dip") else duration
            location = offset+context
            sl = slice(location, location+length)
            amplitude = (3.+3*case_index)*scale
            if kind == "flatline":
                changed[sl] = changed[location-1]
            elif kind == "variance":
                changed[sl] += rng.normal(0, amplitude, length)
            elif kind == "drift":
                changed[sl] += np.linspace(amplitude/length, amplitude, length)
            else:
                changed[sl] += -amplitude if kind == "dip" else amplitude
            if nonnegative:
                changed[sl] = np.maximum(0, changed[sl])
            predictions = model.predict_aligned(changed)
            _, local, _ = prepare_errors(changed, predictions, local_observed, model, bundle["floor"])
            secondary = secondary_model.auxiliary_scores(changed) if secondary_model is not None else None
            ids = np.full(width, -1, dtype=int)
            ids[context:context+length] = 0
            for name, state in initial.items():
                result = replay_policy(copy.deepcopy(state), changed, predictions, local_observed,
                                       offset, len(changed), local, secondary)
                rows.append({"series": stream["relative"], "policy": name, "kind": kind,
                    "seed": seed, "duration_readings": length, "amplitude": float(amplitude),
                    "partition": "held_out" if case_index or kind in ("flatline", "variance") else "diagnostic",
                    "used_for_selection": False, **metrics(stamps, ids, result)})
    return rows, None
