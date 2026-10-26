"""Strict time-series loading, explicit splits, and reproducible NAB downloads."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import pandas as pd

NAB_COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"


def fetch(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Sentinel-NAB"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download_nab(destination):
    """Download data, labels and license, never upstream executables or results."""
    destination = Path(destination)
    tree = json.loads(fetch(
        f"https://api.github.com/repos/numenta/NAB/git/trees/{NAB_COMMIT}?recursive=1"
    ))
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned an incomplete tree")
    paths = [item["path"] for item in tree["tree"] if item["type"] == "blob"
             and (item["path"].startswith(("data/", "labels/"))
                  or item["path"] == "LICENSE.txt")]

    def download(relative):
        url = f"https://raw.githubusercontent.com/numenta/NAB/{NAB_COMMIT}/{relative}"
        content = fetch(url)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite modified file: {target}")
        target.write_bytes(content)
        return {"path": relative, "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content), "url": url}

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(download, paths))
    manifest = {"repository": "https://github.com/numenta/NAB", "commit": NAB_COMMIT,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(), "files": records}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_series(path, max_missing_fraction=0.05):
    frame = pd.read_csv(path)
    if not {"timestamp", "value"}.issubset(frame.columns):
        raise ValueError("CSV must have timestamp,value columns")
    if len(frame) == 0:
        raise ValueError("Empty CSV")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if timestamps.isna().any():
        raise ValueError("Timestamps must be valid")
    input_rows = len(frame)
    duplicate_rows = int(timestamps.duplicated().sum())
    reordered = not timestamps.is_monotonic_increasing
    # Public NAB contains repeated/out-of-order timestamps. Normalize stored
    # history BEFORE splitting/replay; preserve raw downloads and audit changes.
    frame = frame.assign(timestamp=timestamps).sort_values("timestamp", kind="stable")
    frame = frame.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    timestamps = frame["timestamp"]
    numeric = pd.to_numeric(frame["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    observed = numeric.notna().to_numpy()
    missing_fraction = float(1 - observed.mean())
    if missing_fraction > max_missing_fraction:
        raise ValueError(f"Missing fraction {missing_fraction:.3f} exceeds configured limit")
    # Forward filling is causal. Leading missing values remain missing.
    values = numeric.ffill().to_numpy(dtype=float)
    intervals = timestamps.diff().dropna().dt.total_seconds()
    audit = {"input_rows": input_rows, "rows": len(frame), "duplicate_rows_removed": duplicate_rows,
             "timestamps_reordered": reordered, "timestamp_policy": "stable_sort_keep_first",
             "missing_rows": int((~observed).sum()),
             "median_interval_seconds": float(intervals.median()) if len(intervals) else None,
             "irregular_intervals": int((intervals != intervals.median()).sum()) if len(intervals) else 0,
             "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    return timestamps, values, observed, audit


def split_points(n, fractions, explicit=None):
    if explicit is not None:
        points = [int(explicit[key]) for key in ("train_end", "calibration_end", "validation_end")]
    else:
        points = [int(n * value) for value in np.cumsum(fractions)[:3]]
    if not (0 < points[0] < points[1] < points[2] < n):
        raise ValueError("Split boundaries must satisfy 0 < train < calibration < validation < rows")
    return points


def event_ids(timestamps, windows):
    """Return original event IDs; -1 denotes normal. Endpoints are inclusive."""
    ids = np.full(len(timestamps), -1, dtype=int)
    for event, (start, end) in enumerate(windows):
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
        if start > end:
            raise ValueError("Anomaly window has start after end")
        mask = ((timestamps >= start) & (timestamps <= end)).to_numpy()
        if np.any(ids[mask] != -1):
            raise ValueError("Overlapping anomaly windows are not supported; clarify label schema")
        ids[mask] = event
    return ids
