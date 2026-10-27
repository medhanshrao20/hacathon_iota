"""Frozen v2 selection rule from 9a76ee2; no v3 shrinkage or local-rate penalty."""
from .validation import preset_rank, rank


def select_v2(stream, pools, config):
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
        comparison = (source != "predeclared_preset_insufficient_events", key_rank)
        if family not in selected or comparison > selected[family]["rank"]:
            selected[family] = {"candidate": key, "parameters": params, "validation": item["validation"],
                                "selection_metrics": metrics, "selection_source": source, "rank": comparison}
    evidence = {family: item for family, item in selected.items()
                if item["selection_source"] != "predeclared_preset_insufficient_events"}
    winner = max(evidence, key=lambda family: evidence[family]["rank"]) if evidence else "recent_mean"
    return selected, winner
