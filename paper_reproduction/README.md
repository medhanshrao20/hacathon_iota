# Zhu et al. (ICLR 2026): published methods on our NAB data

This is separate from Sentinel v1-v4. It does not modify their models, thresholds,
selection, data or results. No Sentinel module is imported here.

**This is a reproduction of the available published methods on NAB, not a
bit-for-bit reproduction of the entire paper's results.** The paper uses 350
TSB-AD-U evaluation series. We use our existing 58 NAB series. The authors did
not publish complete package locks, model checkpoint revisions, or the internal
MOMENT modifications used for some appendix ablations. Changing those conditions
prevents an honest claim of identical results or identical full-paper execution.

## Run from the hackathon project folder

Use Python 3.11. The existing project environment already has pandas/numpy for
the launcher. On a new PC, install the parent `requirements.txt` first.

```powershell
# Install separate model environments. This does not run inference on NAB.
.\.venv\Scripts\python paper_reproduction/run.py --setup

# Run the main published executable methods on all 58 streams.
.\.venv\Scripts\python paper_reproduction/run.py
```

Model weights download automatically on the first inference. The main suite
uses real foundation models, including 500M-parameter TimesFM and large MOMENT;
it is substantially heavier than Sentinel. The original scripts choose CUDA if
available, otherwise CPU, and retain their original batch sizes. Multiple GB of
packages/weights and potentially very long CPU runs should be expected. No
foundation model has been run here yet; import/setup or model compatibility
failures will stop the run with a log instead of substituting another model.

For the paper's CPU statistical baselines only:

```powershell
.\.venv\Scripts\python paper_reproduction/run.py --suite baselines --setup
.\.venv\Scripts\python paper_reproduction/run.py --suite baselines
```

Useful commands:

```powershell
.\.venv\Scripts\python paper_reproduction/run.py --check-data
.\.venv\Scripts\python paper_reproduction/run.py --plan
# Optional small run, not a full benchmark:
.\.venv\Scripts\python paper_reproduction/run.py --only speed_6005
# Resume using the actual relative run directory printed by the launcher:
.\.venv\Scripts\python paper_reproduction/run.py --resume paper_reproduction/runs/paper_nab_TIMESTAMP
```

New results go to `paper_reproduction/runs/paper_nab_<UTC timestamp>/`.
Existing Sentinel and paper results are preserved. Resume skips only jobs that
finished successfully; a failed model job is retried. Logs identify failures.
Source hashes are checked before running. Each run records the experiment plan,
input hashes, library versions, individual metrics and completed jobs.

## What is reproduced

The main suite has **48 configurations** before per-horizon and embedding
detector expansion. Every configuration is evaluated on every selected stream.
These settings are fixed before scoring and not selected using NAB labels.

| Method | Settings | Implementation source |
|---|---|---|
| Last-w squared difference | w=1,3 | Original `get_scores` and helpers |
| Centered-w squared difference | w=3,5 | Original `get_scores` and helpers |
| Centered moving variance | w=64,96,256,512,1024 | Original `get_scores` and helpers |
| Baseline ensemble | Var-96 and Last-3 | Original full-series min-max; maximum |
| MOMENT zero-shot reconstruction | base/large; w=64,256,512; large also1024 | Original `moment_zs.py` |
| Chronos-Bolt zero-shot forecast | small/base; w=64,256,512 | Original `chronos_zs.py`, AutoGluon API |
| TimesFM zero-shot forecast | 1.0-200M/2.0-500M; w=64,256,512 | Original `times_fm.py` |
| Time-MoE zero-shot forecast | 50M/200M; w=64,256,512 | Original `time-moe.py` |
| Long-horizon Chronos/TimesFM | both sizes; w=64,256,512; h=1,2,4,8,16,32,64,128,256,512 | Original prediction scripts and score-loop AST |
| Short/long ensemble | h=1 and64 | Figure8a notebook min-max then maximum |
| MOMENT embeddings | base, w=64, batch16, 768 dimensions | Original Figure8b notebook function ASTs |
| Embedding detectors | ECOD, IForest(seed0), LOF, COPOD, KNN default; KNN k=1,3,5,10,20,50 | Same notebook fit/decision-function procedure |

The embeddings and detectors are fit on the same evaluated series, as in the
paper's proof-of-concept notebook. This is **transductive offline fitting**, not
a clean held-out supervised training experiment. The original notebook's edge
padding/overwrite is retained. No claim that KNN20 was selected independently
of the original paper's case-study labels is made.

Original CLI modules execute argparse and hard-code a TSB directory. The adapter
loads only their unmodified function ASTs to change the dataset routing. For
neural inference it runs the unmodified scripts themselves. It does not replace
neural models with the simple baselines or with Sentinel's Ridge models.

## Data and evaluation differences to understand

The prepared input contains `Data,Label`, as required by the authors. The 58 NAB
streams contain 365,509 readings after stable timestamp sorting and keeping the
first duplicate, matching our existing NAB normalization. Original NAB CSVs
remain untouched. Missing values are dropped, following the paper's scripts;
timestamps are saved separately and are not model features. Full NAB anomaly
windows supply labels. `data/manifest.json` records the conversion and hashes.

Evaluation covers **entire series**, not our previous final15% test sections.
The total true-event count is therefore not the previous22. There is no online
training/validation/test selection stage in this zero-shot benchmark.

Metrics use the actual TSB-AD `get_metrics` implementation with its original
defaults (`slidingWindow=100`, `version=opt`, `thre=250`). Results include VUS-PR,
VUS-ROC, AUC-PR/ROC and the five F metrics returned by that function. As its code
states, when no predictions are passed, threshold-dependent metrics use an
**oracle threshold chosen using evaluation labels**. Its event-based F1 is not
our onset-based streaming event-F1. Figure8b originally uses dtaianomaly for
VUS-PR; this adapter consistently reports TSB-AD metrics for all methods instead.

The reproduction preserves the paper's offline behavior:

- Centered windows inspect future observations.
- Early backfilled scores can depend on future observations even for Last-w.
- Full-series min-max uses later scores to normalize earlier scores.
- Horizon-h block error is assigned to its first timestamp, although the last
  required observation arrives h-1 readings later.
- Embedding detectors use the complete evaluated embedding sequence.

**These outputs must not be presented as leakage-free immediate streaming
predictions, or directly compared with v2's35.44% deployment-style event-F1.**
No new persistence, calibration, CUSUM, adaptive scaling or alert logic is added.
The paper does not provide a deployable binary threshold-selection policy; this
reproduction does not invent one.

NAB includes all-normal streams, unlike many anomaly benchmark series. Metrics
requiring both classes are explicitly undefined for those streams. Their scores
are still saved; they are not treated as successful detections or silently
assigned perfect scores. Original min-max of a constant array gives NaN; it is
recorded as undefined without changing the authors' formula. `summary.csv` is a
macro mean over valid series and includes `total_series` and `valid_series` so
these exclusions are visible. It is not Sentinel's micro event-F1 aggregation.
Series shorter than a published context window are explicitly marked undefined
for that configuration; no shorter replacement window is used.

## What cannot be claimed as reproduced

- TSPulse: the paper reuses another publication's reported numbers rather than
  supplying a TSPulse inference implementation here. Those TSB results cannot
  be transferred to NAB. No invented TSPulse result is generated.
- Triangulation: this needs a separate labeled tuning dataset and per-dataset
  model selection. Our provided58 streams form the evaluation set. No independent
  NAB tuning set was supplied for this reproduction; it is not manufactured from
  evaluation labels. Original CLI supports tuning and is preserved for inspection.
- Appendix normalization ablations: original scripts are preserved, but they are
  not included in `full`. The paper explicitly requires internal MOMENT library
  modifications that are not included upstream. Running those scripts against an
  ordinary pip MOMENT installation would not reproduce that ablation.
- Supervised foundation-model fine-tuning is a future direction in the paper,
  not a specified training algorithm/checkpoint to reproduce.
- Published TSB rankings/statistical tables/non-TSFM comparisons require the
  original benchmark and external intermediate results. They are not NAB results.
- Bitwise numerical parity across unpinned libraries, checkpoint revisions and
  devices is not established. The exact main-suite foundation-model inference
  has not been tested on this machine; the run will expose compatibility errors.

## Files to inspect after your run

- `summary.csv`: macro metrics and valid-series counts per method.
- `metrics.csv`: individual series/method metrics and undefined reasons.
- `reproductions/intermediate/`: original predictions, embeddings and scores.
- `protocol.json`: fixed plan, source provenance and evaluation caveats.
- `logs/`: subprocess output and exact resolved environment versions.
- `status.json`, `completed.json`: completion/failure and resumable progress.
- `timing.jsonl`: offline job time including model loading/evaluation; **not**
  per-window streaming latency.

## Verification and credit

`test_paper.py` checks original score formulas, future-dependence boundaries,
exact long-horizon score placement/padding, data conversion, source hashes, and
a synthetic end-to-end metric/report/resume cycle. No NAB inference is performed
by these checks.

Paper: Zhu, Carpentier and Verbeke, *When Foundation Models are One-Liners:
Limitations and Future Directions for Time Series Anomaly Detection*, ICLR2026.
[Paper](https://openreview.net/forum?id=H27kvyG4qf) and
[authors' implementation](https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/publications/iclr2026_timeseriesfoundationmodelsad/).
Original code commit `dcbbd9fbeaabfb27ad084ffa4351a2418ea1dab9` (MIT license).
Vendored TSB-AD evaluator commit `303dffbfc11612995b17b06633f8c9d0ea0e38e5`
([source](https://github.com/TheDatumOrg/TSB-AD), Apache2.0 license).
The paper did not pin an evaluator revision; ours is explicit for repeatability.
`UPSTREAM_MANIFEST.json` hashes every preserved file. Both licenses are retained.
The metrics and MOMENT dependency imports were checked locally without loading
pretrained weights; their resolved compatibility locks are included. Those locks
are our tested environments, not the authors' unpublished environment locks.
Other model environments are installed/checked by `--setup` on your machine.
The adaptation/launcher is our engineering work; the algorithms are credited to
their authors. Do not present the paper's methods as our original contribution.
