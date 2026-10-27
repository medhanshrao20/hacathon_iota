# Sentinel v4: controlled anomaly-detection experiments

## Latest reported results and presentation

- [Ten-slide Sentinel presentation](submission/Sentinel_Unbreakabale_Loop.pptx)
- [Imported benchmark results and interpretation](paper_reproduction/reported_runs/paper_nab_20261027T090825_560662Z/README.md)
- [Complete-method comparison](paper_reproduction/reported_runs/paper_nab_20261027T090825_560662Z/summary.csv)

Best mean offline event-F1: **73.64%** (centered variance,64 readings).
Best VUS-PR: **45.82%** (variance,256 readings). These are full-series benchmark
results with evaluation-selected thresholds, not hidden-test streaming scores.
Twelve methods completed58 streams; a further configuration stopped after18.
The presentation reports operational precision, recall, delay and runtime
separately. Add team member names on slide1 before submission.

CPU-only PS2 project. One application entry point: `run.py`. All 58 pinned public
NAB streams are included. The new experiment workflow is implemented and checked,
but **the full real-data v4 run has not been started**. The user will run it.

## Run

Open a terminal in the project folder. If the existing environment is installed:

```powershell
.venv\Scripts\python run.py
```

On a new PC, use Python 3.11/3.12:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run.py --check-data
.venv\Scripts\python run.py
```

On Linux/macOS replace `.venv\Scripts\python` with `.venv/bin/python`.
No GPU, API key or internet is required after dependencies are installed.
Paths in configuration are relative to the project folder.

To check the code on small synthetic data only:

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

The 29 regression tests passed, including a complete two-stream synthetic v4
experiment, test-suffix perturbation, and saved confirmation-model inference.
See `V4_PREPARATION.json` for preparation checks. These checks do not establish
a real-data F1 improvement or guarantee zero overfitting.

## Experiment design

The control uses the v2 configuration and selection rule from commit `9a76ee2`.
Its predictor grid is preserved in `controls/v2_config.json`. No old test scores
are loaded to choose models or policies. The v3 workflow remains available:

```powershell
.venv\Scripts\python run.py --config config_v3.json
```

The v4 workflow:

1. Rebuild v2 development results on the original three chronological folds.
2. Choose a forecaster using earlier folds and historical other-source peers,
   then evaluate policies on the next fold: fold 0 -> 1 and folds 0/1 -> 2.
3. Choose a policy using real development events, paired uncertainty, a recall
   floor and an alert-rate target. Related sources and future peers are excluded.
4. Save every stream's selection before any stress result or test replay.
5. Run synthetic stress cases separately; they never train models or choose policies.
6. Report the frozen policies on the public test, with measured one-reading runtime.

| Policy | Change from control |
|---|---|
| `control_v2` | Unchanged v2 scoring and incident decisions |
| `A_spike` | Immediate trigger above the calibration maximum; ordinary votes retained |
| `B_calibration` | Per-channel empirical scaling plus whole-policy rate calibration |
| `C_spike_calibration` | A + B; completes the 2x2 experiment |
| `D_cusum` | C + two-sided accumulation before the point threshold |
| `E_confirmation` | B + independently calibrated pattern confirmation percentile |
| `F_confirmation_bypass` | C + confirmation; extreme spikes can bypass confirmation |
| `G_single_exceedance` | B with 1-of-1 opening; explicitly not independent corroboration |

All channels share one incident state. No overlapping new onsets are created
inside an open incident. There is no escalation-reopen or changed event matcher.
Confirmation candidates are adopted only if simpler candidates have not passed.
CUSUM and verifier candidates also need evidence of a gain over their simpler parent.

## Calibration and safeguards

Predictors and learned features fit only the fold's training prefix. The public
default excludes known historical training anomalies; this is supervised use of
training annotations. Set `use_training_labels` in the control config to false
if organizer rules prohibit it; record that this changes the original control.

Revised policies split out-of-training calibration into reference and later
rate-tuning blocks. Labels are not accepted by the calibration function. Missing
readings retain their positions and break votes/CUSUM continuity. Unlabeled
calibration is not certified normal: its episode rate is a conservative operating
proxy, not a guaranteed false-alarm rate. True false-alert rates are measured on
real development labels. No test labels enter fitting, calibration or selection.

The alert budget is soft: no alarm quota blocks later readings. Whole-policy
calibration includes spike, CUSUM and confirmation decisions. Spike levels come
from calibration data, not a multiplier fitted to the eight known test misses.

Default adoption requires a development F1 gain of at least .02, no recall drop,
at most 10 false episodes per 10,000 normal readings, enough historical evidence,
and positive lower paired-bootstrap bounds at 95% confidence for both stream
and source-group resampling. Normal streams and their false alerts remain in
the resamples. These are declared conservative gates, not statistical guarantees.
**If evidence is insufficient or uncertain, the selected system retains v2.**

Shared settings are under `v4` in `config.json`; the baseline predictor grid is
under `controls/v2_config.json`. Freeze settings before comparing final test scores.
The legacy search fields in the root configuration do not replace that pinned
control grid. The protocol records whether the control configuration still
matches the original v2 fingerprint. Finite calibration blocks cannot guarantee
precise 99.9th-percentile tail rates; calibration row counts and rate trials are saved.

## Outputs

Each run creates `results/experiment_v4_<UTC timestamp>/`. Old results are preserved.

- `report.html`, `comparison.csv`, `aggregate.csv`: eight fixed policy comparisons
  plus the development-selected system; includes event precision/recall/F1 and delay.
- `development_policy_comparison.csv`: next-fold real-event evidence.
- `selection_frozen_before_test.json`: policy, baseline parameters, adoption gates,
  paired intervals and the historical evidence available for each stream.
- `synthetic_stress.csv`, `synthetic_skips.json`: separate injection diagnostics.
- `runtime_per_window.csv`: P50/P95 latency from real one-reading calls including
  feature construction, forecasts and state updates; excludes fitting and file I/O.
- `test_uncertainty_report_only.json`: descriptive paired test uncertainty; never
  used to change the frozen choices.
- `failures.json`: explicit candidate-policy failures; missing paired evidence
  cannot win selection. Fatal errors stop the run rather than silently dropping streams.
- `series/<signal>/`: control CV logs, calibration decisions, per-policy predictions,
  selected charts and `model.joblib` with continuation state.

New prediction evidence includes spike cutoff, CUSUM statistic/cutoff, secondary
percentile/cutoff and the decision channel. Disabled channel cutoffs are blank/NaN.
Expected value ranges describe point forecast errors; a CUSUM anomaly can occur
inside those ranges and is explained separately.

Continue later readings from a trusted selected model:

```powershell
.venv\Scripts\python run.py --predict data/new_readings.csv --artifact results/experiment_v4_RUN/series/SIGNAL/model.joblib
```

Replace RUN/SIGNAL. New timestamps must follow the saved history. No labels or
retraining are needed for continuation. Only load trusted joblib artifacts.

`--only` and `--limit` are available, but restrict the historical peer pool too.
A small subset may retain v2 because it lacks enough independent event evidence.

## Measured history and remaining limits

| Previous selected system | Event-F1 | Detected / actual | False alerts |
|---|---:|---:|---:|
| v1 | 16.00% | 12 / 22 | 116 |
| v2 | 35.44% | 14 / 22 | 43 |
| v3 | 34.15% | 14 / 22 | 46 |

v3 did not improve F1. v4 has no measured full-run score yet. See `BASELINE.md`,
`V3_RESULTS.md`, `V4_METHODOLOGY.md`, and the historical `README_V3.md`.
The public test has already been inspected: future replays are exploratory.
Hidden organizer data and its exact matching/output schema are still needed for
independent evaluation. Synthetic faults are simulations, not extra real events.
Bootstrap intervals are sensitive with few source groups. No root cause, unseen
generalization or F1 improvement is guaranteed. Tail fitting, transforms,
escalation reopening and EWMA are intentionally deferred.
