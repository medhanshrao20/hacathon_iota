"""Portable CSVs, self-contained charts, and an honest result summary."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd


def write_json(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False), encoding="utf-8")


def prediction_frame(timestamps, values, observed, predictions, replay, model, threshold, start):
    n = len(replay["score"])
    scored = np.isfinite(replay["score"])
    frame = pd.DataFrame({"timestamp": timestamps.iloc[start:start+n].astype(str).to_numpy(),
                          "value": np.where(observed[start:start+n], values[start:start+n], np.nan),
                          "observed": observed[start:start+n], "anomaly_score": replay["score"],
                          "threshold": threshold,
                          "raw_threshold_exceeded": replay.get("raw_flag", replay["flag"]),
                          "incident_profile": replay.get("incident_profile", "plain"),
                          "label": np.where(scored, np.where(replay["flag"], "ANOMALOUS", "NORMAL"),
                                            "UNSCORABLE")})
    for column, horizon in enumerate(model.horizons):
        frame[f"forecast_h{horizon}"] = predictions[start:start+n, column]
        frame[f"expected_h{horizon}"] = replay["center"][:, column]
        frame[f"lower_h{horizon}"] = replay["lower"][:, column]
        frame[f"upper_h{horizon}"] = replay["upper"][:, column]
        frame[f"error_h{horizon}"] = frame["value"] - frame[f"forecast_h{horizon}"]
        frame[f"squared_error_h{horizon}"] = frame[f"error_h{horizon}"] ** 2
        frame[f"score_h{horizon}"] = replay["z"][:, column]
    frame["threshold_exceeded_without_incident"] = frame["raw_threshold_exceeded"] & ~replay["flag"]
    for column, name in enumerate(model.auxiliary_names):
        frame[name] = replay["auxiliary_scores"][:, column]
        frame[f"score_{name}"] = replay["z"][:, len(model.horizons) + column]
    evidence = []
    for i in range(n):
        if not scored[i]:
            evidence.append("Missing observation or insufficient forecast history")
            continue
        dominant = int(np.argmax(replay["z"][i]))
        if dominant >= len(model.horizons):
            name = model.auxiliary_names[dominant - len(model.horizons)]
            evidence.append(
                f"{name}: distance from training normal patterns "
                f"{frame[name].iloc[i]:.6g}; normalized score {replay['score'][i]:.4g} "
                f"versus threshold {threshold:.4g}. "
                "Label includes incident rules; novelty does not establish a root cause."
            )
            continue
        horizon = model.horizons[dominant]
        evidence.append(
            f"h={horizon}: observed {values[start+i]:.6g}; expected "
            f"{replay['center'][i, dominant]:.6g}; range "
            f"[{replay['lower'][i, dominant]:.6g}, {replay['upper'][i, dominant]:.6g}]; "
            f"score {replay['score'][i]:.4g} versus threshold {threshold:.4g}. "
            "Label includes persistence, incident recovery and cooldown rules; this is evidence, not a root cause."
        )
    frame["evidence"] = evidence
    return frame


def chart(frame, path, title, event_labels=None, max_points=6000):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Retain every alert even when background traces are downsampled.
    step = max(1, len(frame) // max_points)
    indices = np.unique(np.r_[np.arange(0, len(frame), step),
                              np.flatnonzero(frame["label"].eq("ANOMALOUS"))])
    data = frame.iloc[indices]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    forecast_columns = [c for c in frame if c.startswith("expected_h")]
    horizon = forecast_columns[0].split("_h")[1]
    for column, name, color in [("value", "Actual", "#16324f"),
                                (f"expected_h{horizon}", "Expected (short horizon)", "#00897b"),
                                (f"lower_h{horizon}", "Lower limit", "#8bc8bd"),
                                (f"upper_h{horizon}", "Upper limit", "#8bc8bd")]:
        figure.add_trace(go.Scatter(x=data.timestamp, y=data[column], name=name,
                                   line={"color": color, "width": 1}), row=1, col=1)
    alerts = frame[frame.label.eq("ANOMALOUS")]
    figure.add_trace(go.Scatter(x=alerts.timestamp, y=alerts.value, mode="markers", name="Alerts",
                               text=alerts.evidence, marker={"color": "#d84343", "size": 6}), row=1, col=1)
    if event_labels is not None:
        truth = frame.loc[np.asarray(event_labels) >= 0]
        figure.add_trace(go.Scatter(x=truth.timestamp, y=truth.value, mode="markers", name="Known event window",
                                   marker={"color": "#efb535", "size": 3, "opacity": 0.5}), row=1, col=1)
    figure.add_trace(go.Scatter(x=data.timestamp, y=data.anomaly_score, name="Combined score"), row=2, col=1)
    figure.add_trace(go.Scatter(x=data.timestamp, y=data.threshold, name="Threshold"), row=2, col=1)
    figure.update_layout(title=title, template="plotly_white", height=760, hovermode="x unified")
    figure.write_html(path, include_plotlyjs=True, full_html=True)


def summary_report(directory, records):
    frame = pd.DataFrame(records)
    frame.to_csv(directory / "comparison.csv", index=False)
    rows = []
    for family, group in frame.groupby("family"):
        tp, fp, fn = (int(group[c].sum()) for c in ("detected_events", "false_alerts", "missed_events"))
        rows.append({"family": family, "series": len(group),
                     "micro_event_f1": 2 * tp / max(1, 2 * tp + fp + fn),
                     "macro_event_f1_all_series": float(group.event_f1.mean()),
                     "detected_events": tp, "false_alerts": fp, "missed_events": fn})
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(directory / "aggregate.csv", index=False)
    links = "\n".join(f'<li><a href="{html.escape(p.relative_to(directory).as_posix())}">'
                      f'{html.escape(p.parent.name)}</a></li>' for p in sorted(directory.glob("series/*/chart.html")))
    page = f'''<!doctype html><meta charset="utf-8"><title>Sentinel results</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#17324d}}
table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ddd}}a{{color:#00796b}}</style>
<h1>Sentinel experiment report</h1>
<p>Each series winner was selected on chronological development folds or other-source-group development data.
All selections were frozen before test replay. The public test was inspected during v1 development;
These repeat-test results are exploratory, not a fresh independent benchmark. Test results did not choose the winner.
These are local event metrics, not organizer-certified scores. Predictions are streaming-causal;
forecast computation is batched, so reported throughput is not a per-window latency benchmark.</p>
<h2>Test comparison</h2>{aggregate.to_html(index=False)}
<h2>Detailed results</h2>{frame.to_html(index=False)}
<h2>Selected-method charts</h2><ul>{links}</ul>
<p>See experiment_log.csv, selected.json, audit.json and saved model bundles for reproducibility.</p>'''
    (directory / "report.html").write_text(page, encoding="utf-8")
