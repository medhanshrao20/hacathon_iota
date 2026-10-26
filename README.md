# Sentinel: PS2 time-series anomaly detection

A portable CPU project for the Utkrishta 24-hour hackathon. It compares four
method families, chooses configurations on validation, and then evaluates an
untouched chronological local test. The nine synthetic regression tests pass,
including a small end-to-end four-method run. No successful full NAB evaluation
or performance scores are claimed yet.

## Run on another PC

Copy this entire folder, including `data/`, to the other PC. Python 3.11 or 3.12
is recommended. No GPU, Colab, API key, database, Node, or internet connection
is needed for an experiment once Python dependencies are installed.

Open a terminal in this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run.py --check-data
.venv\Scripts\python -m unittest discover -s tests -v
```

Start with one series, then run the full search:

```powershell
.venv\Scripts\python run.py --only "realKnownCause/machine_temperature_system_failure.csv"
.venv\Scripts\python run.py
```

On macOS/Linux use `.venv/bin/python` instead. There are no machine-specific
paths in configuration or source. Relative paths resolve from this project
folder, not the terminal's working directory. `run.py` is the only application
entry point. The tests are optional independent synthetic checks.

## Methods

1. **Recent mean / Last-3:** predict from the previous three readings. No fitting.
2. **Rolling median/MAD:** use the prior window's median and robust variation.
   No learned model. Current readings never enter their own reference window.
3. **Single-horizon Ridge:** a trained one-step lagged regression.
4. **Multi-horizon Ridge:** trained direct regressions for multiple horizons.
   Store conceptually separate forecasts and score each when its target arrives.

Every method gets calibration and validation; only Ridge needs regression
training. The multi-horizon design is inspired by Zhu et al. (ICLR 2026), not
claimed as a new algorithm. See `METHODOLOGY.md` and `references/README.md`.

## Data already included

The public NAB CSVs, labels, descriptions and upstream license are under
`data/nab/`. `data/nab/manifest.json` records a pinned upstream commit and a
SHA-256 hash for each downloaded file. `DATA_VALIDATION.json` records the
preparation checks. This is the **public development corpus**, not the
organizer's private starter pack or hidden test.

To fetch the same pinned data again, without training:

```powershell
.venv\Scripts\python run.py --download-data
```

The downloader refuses to replace a file whose contents differ. Do not put
organizer data inside `data/nab/`; preserve the pinned public snapshot.

For organizer data, place CSVs in `data/organizer/data/` and a window-label JSON
in `data/organizer/labels/combined_windows.json`. Change `data_dir`, `labels_file`
and optionally `split_manifest` in `config.json`. CSVs require `timestamp,value`.
JSON keys must be CSV paths relative to `data_dir`, for example:

```json
{
  "machine.csv": [["2026-01-01 10:00:00", "2026-01-01 10:30:00"]],
  "normal_machine.csv": []
}
```

Missing label keys are an error: unknown labels are never treated as normal.
If their schema differs, convert it explicitly before evaluating. The
organizer's exact metric and submission schema were not supplied; the local
metric is documented below and canonical output is not advertised as their
official submission format.

## Parameter selection

`config.json` contains the complete search. Default lookbacks are 24/48/96;
Ridge alphas are 0.1/1/10/100; horizon sets are [1], [1,8,24], [1,16,64].
Five calibration quantiles, fixed/adaptive references and two persistence
rules are compared. This is a meaningful search, not a promise of best scores.

Default chronology: 50% training, 20% calibration, 15% validation, 15% local
test. Use organizer boundaries if given. An optional split-manifest JSON maps
every selected series key to exclusive zero-based row boundaries:

```json
{
  "machine.csv": {
    "train_end": 5000,
    "calibration_end": 7000,
    "validation_end": 8500
  }
}
```

The remaining rows are test rows. These are counts after timestamp normalization,
not raw CSV line numbers or timestamps. No
automatic refit on calibration/validation/test occurs. Thresholds use calibration
errors, and labels are used only for validation selection and final evaluation.
Each series has its own model and settings. A missing-anomaly validation section
is explicitly warned about; such a split cannot establish anomaly recall.

## Outputs after a manual run

Each invocation creates a new timestamped directory under `results/`:

```text
results/experiment_.../
  config_used.json        Exact experiment configuration
  environment.json        Actual installed package versions
  failures.json           Failed series and tracebacks; never silently hidden
  comparison.csv          Four methods' local test metrics per series
  aggregate.csv           Aggregate comparison (micro and macro event F1)
  report.html             Offline report with links to signal charts
  series/<signal>/
    audit.json            Hash, timestamp cleanup, missing values, split/event counts
    experiment_log.csv    Validation trial parameters and metrics
    candidate_failures.json
    selected.json         Validation winners and their configurations
    test_<method>.csv     Detailed local test predictions for each family
    metrics_<method>.json
    predictions.csv      Winner selected on validation, NOT on test
    validation_predictions.csv
    validation_chart.html
    chart.html            Offline interactive chart for that winner
    model.joblib          Trained model plus causal continuation state
```

Open `report.html` in a browser. Charts contain embedded Plotly and work offline.
Charts may downsample background lines; prediction CSVs retain all readings.
Progress is printed after every five forecast configurations. Completed-series
reports are refreshed as the run proceeds. An interrupted series must be rerun;
there is no implicit resume or reuse of old search results.

## Continue on new, unlabeled readings

Use the winner's trusted saved bundle from the same signal:

```powershell
.venv\Scripts\python run.py --predict data/new_readings.csv --artifact results/experiment_RUN/series/SIGNAL/model.joblib
```

Replace `RUN` and `SIGNAL` with real output names. New timestamps must be later
than the saved state's last timestamp. No retraining or label access occurs.
Outputs and updated state go to a new `results/predict_.../` directory.
Only load model bundles you created or trust: joblib uses executable pickle.
Dependencies must also match the environment that created the model.

## Metric and interpretation limits

Some public NAB files contain duplicated or out-of-order timestamps. The loader
stably sorts by timestamp and keeps the first occurrence of each timestamp before
splitting or chronological replay. Raw downloaded CSVs remain unchanged. Removed
row counts and reorder flags are recorded. Explicit split row boundaries must
refer to this normalized sequence. This offline cleanup is not an assertion that
an online system can reorder already-processed late arrivals.

The local metric counts one alert at each consecutive positive episode's onset.
An onset must lie inside a labeled window to detect it. One true positive per
window; duplicate alert episodes are false alerts. Missing rows break episodes.
Windows crossing split boundaries are clipped to the evaluation section; delay
is measured from the first eligible labeled reading in that section. This is
**not** a claim of exact NAB-score, paper-metric, or organizer-metric parity.
Replace matching in `sentinel/evaluation.py` when official rules arrive.

No label-based point adjustment, oracle test threshold, or test-based winner
selection is used. Aggregate test comparisons are descriptive: do not choose a
new winner from them and continue claiming the same test is untouched.

Prediction errors and scores do not prove physical failure. Ranges are empirical,
not guaranteed confidence intervals. Short/long horizon combination can miss
very long or predictable anomalies. Rejected-error adaptation can remain stuck
after a legitimate permanent level change. Horizons count observations; irregular
intervals are reported, not silently resampled. Prediction throughput is batch
processing throughput, not a measured online per-window latency guarantee.

## Code map

| File | Responsibility |
|---|---|
| `run.py` | Single application entry point |
| `sentinel/data.py` | Loading, splits, labels, pinned downloads |
| `sentinel/models.py` | Statistical methods and Ridge forecast alignment |
| `sentinel/detection.py` | Calibration, sequential decisions and adaptation |
| `sentinel/evaluation.py` | Explicit local event evaluation |
| `sentinel/pipeline.py` | Search, selection, test and saved-model inference |
| `sentinel/reporting.py` | Prediction explanations, charts and reports |

Initial preparation used syntax/data checks only. A later first NAB attempt
failed due to squeezed single-target Ridge predictions. That shape bug is fixed;
all nine synthetic tests now pass. The full NAB experiment has not been restarted
by the assistant and still needs a fresh manual run.
