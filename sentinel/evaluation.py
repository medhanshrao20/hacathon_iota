"""Explicit local event metric; replace matching here for organizer rules."""
import numpy as np


def evaluate(timestamps, event_ids, flags, eligible):
    """One alert per consecutive positive episode; onset must enter a true window.

    At most one TP per window. Later alert episodes in a detected window are
    duplicates (counted as false alerts). Boundary-crossing events are clipped
    to the evaluated section; delay uses the first eligible event timestamp.
    Missing/unscorable readings break alert episodes. No point adjustment.
    """
    flags = np.asarray(flags, dtype=bool) & eligible
    onset = flags & ~np.r_[False, flags[:-1]]
    truth = event_ids >= 0
    events = set(event_ids[eligible & truth].tolist())
    detected, false_alerts, delays = set(), 0, []
    normal_alerts, duplicate_alerts = 0, 0
    for t in np.flatnonzero(onset):
        event = int(event_ids[t])
        if event in events and event not in detected:
            detected.add(event)
            first = np.flatnonzero(eligible & (event_ids == event))[0]
            delays.append(float((timestamps.iloc[t] - timestamps.iloc[first]).total_seconds()))
        else:
            false_alerts += 1
            if event in detected:
                duplicate_alerts += 1
            else:
                normal_alerts += 1
    tp, fn = len(detected), len(events - detected)
    precision = tp / (tp + false_alerts) if tp + false_alerts else 0.0
    recall = tp / len(events) if events else 0.0
    f1 = 2 * tp / (2 * tp + false_alerts + fn) if 2 * tp + false_alerts + fn else 0.0
    point_tp = int((flags & truth & eligible).sum())
    point_fp = int((flags & ~truth & eligible).sum())
    point_fn = int((~flags & truth & eligible).sum())
    point_tn = int((~flags & ~truth & eligible).sum())
    return {"event_f1": f1, "event_precision": precision, "event_recall": recall,
            "true_events": len(events), "detected_events": tp, "false_alerts": false_alerts,
            "missed_events": fn, "alert_episodes": int(onset.sum()),
            "mean_delay_seconds": float(np.mean(delays)) if delays else None,
            "point_f1": 2 * point_tp / max(1, 2 * point_tp + point_fp + point_fn),
            "point_accuracy": (point_tp + point_tn) / max(1, int(eligible.sum())),
            "true_positive_points": point_tp, "true_negative_points": point_tn,
            "false_negative_points": point_fn,
            "false_positive_points": point_fp, "eligible_points": int(eligible.sum()),
            "normal_alert_episodes": normal_alerts, "duplicate_alert_episodes": duplicate_alerts,
            "normal_points": int((eligible & ~truth).sum()),
            "false_alerts_per_1000_normal_points": 1000 * false_alerts / max(1, int((eligible & ~truth).sum())),
            "false_positive_rate": point_fp / max(1, int((eligible & ~truth).sum()))}
