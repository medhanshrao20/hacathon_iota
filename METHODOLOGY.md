# Sentinel v2/v3 forecasting methodology

The current default workflow is the controlled v4 experiment described in
V4_METHODOLOGY.md. This file preserves the shared forecasting definitions and
historical v3 selection design; v4 uses the frozen v2 selector instead of v3's
local-evidence shrinkage. See README.md for current commands.

## Objective and provenance

Predict expected readings and flag unusual deviations with numerical evidence.
The Last-3 baseline and short/long-horizon motivation are credited to Zhu,
Carpentier and Verbeke (ICLR 2026). Ridge, robust statistics and seasonal naive
are established methods. This is an engineering adaptation, not a claim of a
new algorithm. Source links and selected licensed upstream files are in
`references/`. No foundation-model weights or decomposition are required.

## 1. Data preparation

Parse timestamps, stable-sort and retain first occurrence per timestamp before
splitting. Preserve raw downloaded bytes. Reject invalid timestamps and streams
above the missing-value limit. Forward-fill context only; never backfill. Missing
actual targets are excluded from training/scoring. Leading missing context stays
unavailable. Irregular intervals are reported, not silently resampled. Horizons
are observation counts, not guaranteed elapsed durations. Normalized row counts
define every explicit split boundary.

## 2. Development isolation

With final boundaries (train_end, calibration_end, test_start), define validation
width d = test_start - calibration_end. For shifts 2, 1, 0 create folds:

  (train_end - shift*d, calibration_end - shift*d, test_start - shift*d)

Each fold is strictly ordered and ends before test_start. No shuffle, random
split or bidirectional feature construction occurs. The object passed into
development search contains only values, observation masks and label IDs before
test_start. Scalers, seasonality estimates, noise floors and model targets are
constructed from the appropriate fold's training prefix only.

Default folds are 0–20/20–40/40–55%, 0–35/35–55/55–70%, and
0–50/50–70/70–85% for train/calibrate/validate. Previous validation observations
can enter a later fold's training set because they are then historical. They
cannot change earlier predictions. Final models use only the last fold's
training prefix. No refit on all development data is silently performed.

## 3. Optional supervised training cleanup

When use_training_labels=true, form the normal mask ONLY from that fold's
training label IDs. Exclude Ridge/boosting windows if any point in their complete
input-to-last-target interval is missing or known anomalous. Prefix sums make
this check exact and efficient. Exclude those training anomalies from period
discovery and noise estimation too. Filtering cannot access calibration,
validation or test labels. Record accepted and rejected training-window counts.

The public default enables this supervised historical cleanup. Set it false
when organizer training annotations are unavailable/disallowed. The disabled
mode does not consult training labels for fitting or feature discovery.
Calibration and live reference updates never consult anomaly labels.

## 4. Predictors

Recent mean: forecast y[T] = mean(y[T-3:T]); report squared error as well as the
calibrated score. Rolling median uses the median of a trailing W window, and
normalizes its signed error by 1.4826*MAD of that past-only raw window.

Seasonal naive: detect a credible repeating lag p on training pairs only. Scan
lags up to min(512, training_length/3), requiring enough allowed pairs, correlation
at least .6, a local peak, and at least .1 improvement over the half-period lag.
Choose the shortest qualifying cycle to avoid an arbitrary multiple. If none
qualifies, p=1 and the seasonal method is explicitly persistence. It predicts
y[T-p], with no fit on later values.

Ridge: fit StandardScaler and L2 regression with intercept on lagged readings.
If training-only period discovery succeeds, append two seasonal historical lag
features to the lookback. Direct target sets are [1], [1,8,24], [1,16,64]. For
origin t, all inputs end at t and every target t+h must be inside training.
At current time T score predictions made at T-h. Batch implementation preserves
this alignment and is checked by causal perturbation tests. Regression alpha
is 1/10/100; windows are 24/96. Ridge is a candidate, not presumed best.

Histogram gradient boosting: predict the next reading from the same past-only
lags plus window mean, standard deviation, median, slope, mean absolute change,
flat-change fraction and recent level contrast. Fit absolute-error loss for
100 fixed iterations, learning rate .05, at most 7 leaves, L2=10 and minimum
leaf size max(20, 1% of accepted training windows). Disable random early stopping;
there is no hidden shuffled validation split. These conservative parameters are
declared defaults, not a claim of globally optimal values.

Pattern hybrid: keep at most 512 evenly spaced clean training windows. Build
two nearest-neighbor indices: eight-bin PAA of per-window z-normalized values,
and seven window-state features scaled by a training-only RobustScaler. At T,
query the window ending at T, after observing T, for shape and state distances.
This is anomaly evidence, not a forecast. The companion numerical forecast is
Last-3 using only values before T. Distances are separate score channels: only
distance ABOVE its calibration median contributes, so a more familiar window
cannot trigger a novelty alarm. Distance scale floors are .001. No labels or
future windows enter the reference index. The index is capped for CPU/RAM use.

## 5. Error calibration and expected ranges

Use out-of-training errors in the calibration prefix. Each horizon has separate
location/scale. Let b[h] be median signed error and D[h]=abs(error-b[h]). Compare:

  MAD scale = max(1.4826*median(D), .1*quantile(D,.9), floor)
  Empirical scale = max(quantile(D,q), floor)

q is .99 or .999. The training-derived raw floor combines .001 times robust
training spread, .1 times median normal adjacent change, a small relative level
tolerance and 1e-9. This is a numerical/noise heuristic, not known sensor accuracy.
The median detector divides by its past-only raw MAD first and uses a dimensionless
floor of .001 for calibration. No future-derived variance floor is used.

  z[T,h] = abs(error[T,h]-b[h])/scale[h]
  score[T] = max_h z[T,h]
  threshold = max(quantile(calibration combined scores,q), 1)

The threshold lower bound prevents an all-constant calibration reference from
creating an arbitrarily narrow range. Expected center is forecast+b*unit;
half-width is threshold*scale*unit. unit=1 except for the median detector's
raw local MAD. Ranges are empirical; neither Gaussian probability nor formal
coverage is claimed. Calibration anomalies can still contaminate estimates.

## 6. Incident decisions and past-only adaptation

Compare plain voting with a guarded incident state. Opening is either 1-of-1
or 2-of-3 raw exceedances. An open guarded incident remains open through short
dips. Close only after three observations below .6*threshold, then suppress
reopening for eight observed readings. No true-event IDs enter this state.
These fixed rules may merge or miss nearby real events; evaluate that tradeoff.
Missing targets are unscoreable and clear voting history, not fabricated normal
readings. Scoring flags across missing rows remain separate metric episodes.

Compare fixed references with adaptation. Adaptive references initialize from
the last 200 valid calibration error vectors. Score and decide BEFORE updating.
Append only errors with no raw exceedance and no active incident. Recompute
every 16 accepted readings. Scale cannot drop below .5 of initial scale or
the training floor. Neither coefficients nor thresholds are tuned using live
truth. False normal acceptance can contaminate references; legitimate permanent
changes can freeze adaptation. No unvalidated reset is hidden in the code.

## 7. Stable selection and sparse-event fallback

Only candidates completing every fold are eligible. Pool fold TP/FP/FN and
normal-point counts; measure dispersion of F1 across folds containing events.
The explicit ranking objective is:

  pooled event F1 - .1*event-fold F1 standard deviation
    - .05*max(0, false_alerts_per_1000_normal_points - budget)

budget defaults to 1.0. Tie-break with fewer false alerts, higher recall, lower
model complexity and stronger Ridge regularization. This objective and budget
are declared heuristics, not a guarantee of the lowest test false-alert rate.

Tune per stream only with at least five DISTINCT labeled development events
and events represented in at least two folds. Otherwise pool validation summaries
from OTHER source groups and only peers whose last development timestamp is
no later than the target's last pre-test timestamp. Both artificial categories
share a source group. Users must group other related streams appropriately.
No target/source-related stream or future peer-label period enters the pool.
With a valid peer pool and fewer local events, blend the peer ranking objective
with the local objective using weight min(.5, local_events/(local_events+5)).
Subtract a further .05 times the excess local false-alert rate above the budget;
even event-free local folds provide evidence against noisy transferred settings.
Record the local evidence weight and source. This does not use test alarms.
The pool uses only candidates complete on all peer folds. It never reads old
test metrics, prior result folders or test-selected parameters.

If no eligible pool has sufficient event evidence, use the deterministic preset:
fixed reference, empirical threshold, highest configured quantile, guarded
incidents, 1-of-1 vote, window nearest 48 and stronger regularization. With no
evidence to choose a family, use recent mean. Do not manufacture a validation
winner on an event-free period. Record selection_source and selection_metrics.

All streams finish development first. Save every selected configuration in
selection_frozen_before_test.json before any stream's test replay begins.
This prevents another stream's already-computed test result influencing fallback.

## 8. Evaluation, state and reporting

The local event-matching rule remains the v1 rule: episode onset inside a labeled
window earns one TP; duplicates count as false alerts; no point adjustment.
Report normal versus duplicate alert episodes separately. Clip windows at section
boundaries and disclose censored delay origins. This provisional metric is not
the original NAB score or a certified organizer event-F1 implementation.
Align both matching and prediction schema when organizer specifications arrive.

Reports distinguish current raw exceedance from incident state and suppressed
reopening. Save numerical forecasts, ranges and evidence. Save only the winning
v3 bundle, including fitted model, detector state and sufficient lag history for
same-stream continuation. New-data inference requires later timestamps, no labels
and no refit. Older bundles are rejected. Throughput is batch processing speed,
not a certified real-time per-window latency benchmark.

## 9. What the safeguards establish

Synthetic tests verify train-boundary target isolation, train-only scaling and
period discovery, full-span anomaly filtering, unchanged CV/selection after
test-value/label perturbations, target-group and future-peer exclusion, chunked
incident replay equivalence, CLI selection freezing, and saved inference.
Read-only data/fold checks cover all 58 series. Added tests check both new model
families, one-sided novelty evidence and vectorized replay against an independent
scalar reference. The 23 passing tests establish implementation checks; only a
completed real-data run can establish a measured F1 comparison.

No implementation can guarantee zero overfitting. The public test has already
been inspected and influenced the design, so rerun results are exploratory.
The code prevents direct test use during fitting/calibration/selection; an
independent generalization claim requires genuinely untouched data such as the
organizer's hidden test. More tuning on the same public replay does not create
fresh evidence. Existing v1 results stay available as the comparison baseline.

## 10. Read-only selection-policy audit

The optional --selection-audit command validates saved configuration, data hashes,
fold boundaries and scoring-source hashes, then reads only development logs.
For each stream, choose a candidate using fold 0 and evaluate on fold 1;
then choose using folds 0/1 and evaluate on fold 2. Borrow peer summaries only
when their validation end predates the target's current selection boundary,
and exclude the whole target source group. Candidate tie order follows the
original experiment log. Compare six declared transfer/local-evidence policies
and seven fixed-family policies. Rank their pooled next-fold metrics using the
same F1/dispersion/false-alert objective. This audit does not alter normal runs,
fit new models or load test predictions. It is development evidence, not an
independent generalization claim.
