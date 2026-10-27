
# Create venv
python -m venv .venv
# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# (Or on Linux / macOS: source .venv/bin/activate)

# Install base launcher dependencies
pip install -r paper_reproduction/requirements.txt
# Set up the isolated environments for all model families (MOMENT, Chronos, TimesFM, Time-MoE)
python paper_reproduction/run.py --setup







# Historical v3 guide

Use run.py --config config_v3.json for this workflow. The default is now v4.

# Sentinel v3: causal time-series anomaly detection

CPU-only PS2 project with one application entry point, `run.py`. Includes the
58 public NAB series. Existing v1/v2 results are preserved. The v2 selected system
achieved 35.44% local event-F1 (14 detected events, 43 false alerts, 8 missed).
The v3 run adds nonlinear forecasting, pattern novelty and locally checked selection.

**Measured result: the attempted upgrade did not improve selected-system F1:**
35.44% (v2) → 34.15% (v3), with 14/22 events detected in both runs and false
alerts increasing from 43 to 46. All 58 streams finished in 47.8 minutes.
Treat this as an experimental branch. See `V3_RESULTS.md`; the v2 implementation
is preserved at commit `9a76ee2`. Do not claim these additions are a better model.

## Run

Open a terminal in the project folder. For a fresh PC use Python 3.11/3.12:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

For an existing installation no dependency reinstall is needed. Run:

```powershell
.venv\Scripts\python run.py --check-data
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python run.py
```

On macOS/Linux use `.venv/bin/python`. All relative paths resolve from the project
folder. No GPU, API keys, database or internet is needed after dependency setup.
Start with one series if desired:

```powershell
.venv\Scripts\python run.py --only "realKnownCause/machine_temperature_system_failure.csv"
```

`--only` / `--limit` restrict the development pool too. A single-stream run may
have too few events and no unrelated source group, so it uses the declared preset.
Its settings may therefore differ from a full-corpus run. This is intentional.

## What changed

| Change | Purpose |
|---|---|
| Three expanding chronological folds | Avoid selecting from one sparse validation period |
| Minimum event evidence and stable-fold ranking | Reduce fragile parameter selection |
| Other-source-group fallback with timestamp cutoff | Borrow historical development evidence without target/source/future leakage |
| Training-only seasonality discovery | Add seasonal-naive predictions and seasonal Ridge lag features |
| MAD and empirical-error calibration | Compare threshold approaches without assuming Gaussian errors |
| Training-derived tolerance and bounded scale shrinkage | Handle nearly constant signals and unstable normalization |
| Incident recovery, hysteresis and cooldown | Reduce repeated alerts; compare against plain decisions |
| Optional known-training-anomaly exclusion | Avoid fitting on contaminated input/target spans |
| Frozen selection file before any test replay | Audit that test labels did not choose models |

Seven method families are compared: recent mean (Last-3), rolling median/MAD,
seasonal naive, single-horizon Ridge, multi-horizon Ridge, histogram gradient
boosting and a normal-reference pattern hybrid. If no credible
period is found, seasonal naive explicitly falls back to lag-1 persistence.
All require calibration. Ridge and boosting learn regression coefficients/trees;
the pattern hybrid fits a scaler and nearest-normal-window reference index.
The new models are candidates, not guaranteed winners.

Boosting uses 100 fixed rounds, 7 leaves, learning rate .05, L2=10 and an absolute
error loss. Random early stopping is disabled. The pattern hybrid compares
eight-bin normalized shapes and seven level/variation features against at most
512 clean training windows. Shape/state distances are separate upper-only score
channels; its displayed forecast remains the actual Last-3 forecast.

## Training-label mode: read this before organizer submission

The public-data default is **`"use_training_labels": true`**. It removes any
Ridge/boosting training example whose complete input-to-last-target span intersects a
known TRAINING anomaly or missing observation. Pattern reference windows receive
the same full-window cleanup. Period discovery and the training
noise estimate also exclude known training anomalies. This is supervised use of
historical training annotations, not a claim of fully unsupervised learning.

If organizers do not provide/permit training annotations, set it to **false**.
No training anomaly labels then affect the model, period discovery or noise floor.
In both modes, calibration labels do not set thresholds; test labels never enter
fitting, calibration, online updates or selection. Validation labels are used for
model selection, as explicitly documented.

## Data and chronological boundaries

`data/nab/` holds pinned raw data, labels, description and upstream license;
`manifest.json` contains original URLs and hashes. `DATA_VALIDATION.json` records
the original read-only preparation audit. Use:

```powershell
.venv\Scripts\python run.py --download-data
```

to redownload the same commit without training. Modified raw files are not
overwritten. Public NAB has 49 duplicate timestamp rows. In-memory stable sorting
and first-occurrence deduplication yields 365,509 rows; raw downloads stay intact.
Audit counts, missing values, irregular intervals and data hashes are recorded.
There is no future-based interpolation or backward fill.

Default final split boundaries remain 50% train / 20% calibration / 15% final
validation / 15% test. CV expands backward from these boundaries:

| Fold | Fit on | Calibrate on | Validate on |
|---|---|---|---|
| 1 | 0–20% | 20–40% | 40–55% |
| 2 | 0–35% | 35–55% | 55–70% |
| 3 | 0–50% | 50–70% | 70–85% |

Each fold has a newly fitted scaler and model. Targets cannot cross its training
boundary. Later folds may train on earlier-fold observations/labels once they
are historical; earlier folds are never refitted using later observations.

Selection uses only completed folds. At least five distinct development events
across at least two folds are required for per-stream tuning. Otherwise borrow
development summaries from OTHER source groups whose development timestamps
are not later than the target's last pre-test timestamp. If evidence is still
insufficient, use a preset independent of validation labels. With a valid peer
pool but sparse local events, blend local development evidence with that prior
(local weight min(.5, events/(events+5))) and penalize excessive local false
alerts. Even event-free local folds can reject a transfer that alarms too often.
Put related streams
in the same parent folder or map folder aliases through `source_groups`.
Both artificial NAB categories are grouped together by default.

The **entire corpus finishes development and freezes selections before any test
replay**. Final models use fold 3's training boundary; they are not refit on
calibration/validation/test. Error-reference adaptation is past-only and label-free.

## Configuration

`config.json` is the full experiment specification. Defaults:

- Lookbacks 24/96; Ridge alpha 1/10/100.
- Horizons [1], [1,8,24], [1,16,64].
- Two threshold methods; quantiles .99/.999; fixed/adaptive references.
- 1-of-1 or 2-of-3 opening votes; plain or guarded incident handling.
- Guarded mode closes after 3 readings below 60% of the opening threshold,
  followed by an 8-reading cooldown. Those settings are fixed, not tuned on test.
- Adaptive buffer 200 accepted errors, update every 16 accepted readings,
  scale cannot shrink below half its initial value or its training-derived floor.
- Selection penalizes inconsistent folds and false alert rates above the
  operational target of 1 per 1,000 normal readings. It is a target, not a promise.

There are 26 predictor specifications × 32 detector settings = 832 candidates,
each evaluated on three folds when feasible (2,496 fold trials per stream).
Failed candidates are logged and cannot win based on an incomplete fold set.
These defaults are not advertised as best measured parameters. Do not enlarge
the search merely to fit a few labeled events.

Organizer data must replace public development data explicitly. Set `data_dir`
and `labels_file`; CSVs need `timestamp,value`. Labels use relative CSV keys:

```json
{"machine.csv": [["2026-01-01 10:00:00", "2026-01-01 10:30:00"]], "normal.csv": []}
```

Missing keys are errors. For organizer boundaries, set `split_manifest` to a JSON
mapping each key to exclusive normalized row counts `train_end`,
`calibration_end`, `validation_end`. The latter is the test start. Earlier CV
folds are derived by shifting backward by the last validation-section width.
If they do not fit, change the CV/split settings rather than using future data.

## Outputs

Every invocation creates `results/experiment_v3_<timestamp>/` without overwriting
old runs. During phase 1 there are no new test results yet; this is expected.

- `report.html`, `comparison.csv`, `aggregate.csv`: test-replay comparisons.
- `selection_frozen_before_test.json`: all selected configurations before testing.
- `config_used.json`, `environment.json`, `source_hashes.json`: reproducibility.
- `failures.json`: explicit stream failures and their phase.
- `series/<signal>/experiment_log.csv`: every CV fold trial.
- `cv_summary.json`, `candidate_failures.json`, `audit.json`, `selected.json`.
- `test_<method>.csv` / `metrics_<method>.json`: each available method's outputs.
- `predictions.csv`, `chart.html`: validation-selected winner's public test replay.
- `validation_predictions.csv`, `validation_chart.html`: winner on the final CV fold.
- `model.joblib`: v3 model and continuation state; only load trusted bundles.

Prediction CSVs include forecast, corrected center, expected lower/upper bounds,
signed/squared errors per horizon, score, raw exceedance, incident label and
numerical evidence. `threshold_exceeded_without_incident` discloses readings
suppressed by persistence/cooldown. A low alarm count is not itself proof of a
better detector: recovery/cooldown can miss separate nearby events.

To continue the same signal on later unlabeled readings:

```powershell
.venv\Scripts\python run.py --predict data/new_readings.csv --artifact results/experiment_v3_RUN/series/SIGNAL/model.joblib
```

Replace RUN/SIGNAL. New timestamps must be after the saved state. Older bundles
are rejected; v3 writes fresh predictions and state in a new directory.

## Honest evaluation

The old public test was already inspected and informed these improvements.
**A rerun on it is exploratory, not a new independent test and not proof of
zero overfitting.** Final independent evidence needs the organizer's hidden test
or another genuinely untouched dataset. Do not optimize against the new replay
score and describe it as an untouched evaluation.

The provisional local event metric is unchanged: one alert per consecutive
positive episode onset, one TP per true window, duplicate episodes count as
false alerts. Missing rows break episodes; boundary-crossing windows are clipped.
No point adjustment or test-label oracle threshold is used. Replace matching and
the output schema with the organizer's rules once supplied; this is not the
original NAB score or a claimed exact paper reproduction.

23 synthetic tests passed during upgrade, including CLI, saved inference,
test-value/label perturbation, training-label isolation, fold boundaries and
source/time-isolated pooling. All 58 series passed read-only data/fold checks.
New checks cover both added models and compare accelerated replay against an
independent scalar oracle. Existing data and results have not been deleted.
See `METHODOLOGY.md`, `BASELINE.md` and `references/`.

Audit model-selection alternatives without fitting or reading any test outputs:

```powershell
.venv\Scripts\python run.py --selection-audit results/experiment_v3_RUN
```

This checks thirteen declared policies by choosing from earlier development
folds and evaluating on the next fold, with historical other-source peers only.
It writes a separate `selection_audit_...` folder and does not change models.
