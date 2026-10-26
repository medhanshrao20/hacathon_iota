# Methodology and experiment protocol

## Goal

For each observed time step, output a numerical anomaly score, NORMAL/ANOMALOUS
label and numerical explanation. Missing or insufficient-history rows are
UNSCORABLE rather than fabricated normal predictions. Optimize the organizer's
event metric once supplied; use the explicit local event metric in the meantime.

## 1. Data preparation

- Input: a separate `timestamp,value` CSV for each univariate stream.
- Parse timestamps as UTC and reject invalid dates. Public NAB has duplicate and
  out-of-order timestamps: stable-sort stored history and keep the first row per
  timestamp before splitting/replay. Preserve raw CSVs and audit removed/reordered
  rows. Split boundaries refer to normalized rows. This is offline history
  normalization, not online late-arrival handling.
- Convert invalid/nonfinite values to missing and report their count.
- Forward-fill values for model context only. Never backfill or interpolate
  using future observations. Missing actual targets are not scored or trained on.
- Reject streams above the configured missing fraction. Leading missing context
  remains unavailable until a complete lag window exists.
- Keep the original sampling order. Report irregular intervals. A forecast
  horizon means an observation count, not an assumed wall-clock duration.
- Keep data provenance, raw hashes and exclusive row split boundaries.

## 2. Chronological roles

Default sections are training 50%, calibration 20%, validation 15%, test 15%.
These are development defaults, not rules from the hackathon or paper. An
organizer boundary manifest overrides the defaults. Predictions for later
sections can use earlier observed context; parameters never use future labels.

Ridge and its StandardScaler fit only training examples. For origin t and
forecast horizon h, require t+h < train_end. Fit multi-output models only on
origins for which all targets are observed and remain inside training. Training
data can contain anomalies: no hidden-test or validation labels are used to
remove them. Calibration contains out-of-training forecasts, not fitted errors.

## 3. Four method families

### Recent mean / Last-3

Forecast y[T] from mean(y[T-3:T]). Its raw squared error is the paper's causal
Last-3 baseline away from warm-up boundaries. The project adds calibration,
thresholding, persistence and optional error-reference adaptation. This is not
an exact reproduction of the paper's complete metric pipeline.

### Rolling median/MAD

Forecast y[T] from median(y[T-W:T]). Compute local variation as
1.4826 * median(abs(history - median(history))). The current observation is
excluded. Divide signed error by this past-only local scale, then calibrate
its center/scale like the other methods. The raw scale has a training-derived
floor to make constant histories well-defined. This detector is not copied
from the paper; it is a robust statistical comparator.

### Single-horizon Ridge

Input X[t] = [y[t-W+1], ..., y[t]]. Target is y[t+1]. Fit StandardScaler and
L2-regularized Ridge regression, with intercept, using scikit-learn's Cholesky
solver. No neural network, API, GPU or pretrained weights are needed.

### Multi-horizon Ridge

Use the same inputs with direct targets at horizons [1,8,24] or [1,16,64].
The regressions are direct: no recursive use of guessed intermediate values.
At time T, compare y[T] with forecasts made at T-h for each horizon h.
Batch matrix computation is mathematically equivalent to this forecast
alignment; future observations are not part of a row's input.

This implements a causal, endpoint-error adaptation of the short/long-horizon
combination in Zhu et al., section 6.1. Their experiments use foundation
models and future-block MSE assigned back to a block's start. Here we use
Ridge, mature forecasts, robust calibration and the current timestamp.
Therefore a result is not claimed as a reproduction of their reported score.

## 4. Calibration

For each horizon, collect signed errors from the calibration section. Only
observed targets with all forecast columns available are eligible.

  bias[h] = median(error[h])
  scale[h] = max(1.4826 * MAD(error[h]), floor)
  z[T,h] = abs(error[T,h] - bias[h]) / scale[h]
  score[T] = max_h z[T,h]

For the rolling median method, error above means signed error already divided
by the past-only raw MAD. Each method therefore has its own calibrated score;
score magnitudes are not probabilities and cannot be compared as confidence.

The initial threshold is a quantile of the combined calibration scores, not
an assumed universal z=3. Candidate quantiles are .95/.975/.99/.995/.999.
Calibration labels are not used. Calibration anomalies can contaminate
thresholds; robust location/scale only mitigates this, not guarantees immunity.

At each horizon display:

  expected = forecast + bias * local_unit
  lower = expected - threshold * scale * local_unit
  upper = expected + threshold * scale * local_unit

local_unit is one for forecasting methods and the raw local MAD for the median
method. These are empirical detection ranges, not physical min/max or formal
coverage guarantees.

## 5. Sequential decisions and adaptation

For each eligible reading: compute scores using the existing reference, append
the raw threshold-exceedance vote, then issue the persistence decision.
Try 1-of-1 and 2-of-3 votes. Persistence can delay/miss isolated anomalies;
its false-alarm benefit must be validated. It can remain active briefly after
the current score falls, which is disclosed in output evidence.

Compare fixed references with adaptive references. Adaptive references begin
with the last 200 valid calibration errors. After scoring, append the current
errors only when neither the raw exceedance nor the persistence label is
anomalous. Recalculate robust reference statistics every 16 accepted readings.
The buffer contains at most 200 accepted vectors. Threshold quantile multiplier
and Ridge coefficients are not retrained during evaluation.

This gate prevents immediate contamination but can freeze adaptation during a
legitimate permanent change. No unvalidated auto-reset or test-label correction
is hidden in the implementation. Missing readings clear persistence votes and
do not update reference statistics.

## 6. Search and selection

- W in [24,48,96]. Ridge alpha in [.1,1,10,100].
- Horizon sets [1], [1,8,24], [1,16,64].
- Fixed/adaptive reference; five threshold quantiles; two persistence rules.
- No selected parameters are described as best before validation is run.
- Select the best configuration per family on validation only. Rank by local
  event F1, then fewer false alert episodes, higher recall, lower delay, lower
  false-positive point rate, lower one-step MAE and lower complexity.
- Select the winning family with the same validation ranking. Test results
  are computed only after selection and never select the saved winner.
- If validation contains no true events, warn explicitly: ranking can assess
  false alerts and forecasting fit, but cannot establish anomaly sensitivity.
- Each stream has independent selection; this is appropriate for known streams
  with historical labels, not a claim of zero-shot generalization to new sensors.
- Preserve every trial and candidate failure. A missing family is a visible
  failed series, not silently omitted from a fair four-method comparison.

## 7. Local evaluation

The implementation deliberately spells out a provisional metric instead of
assuming "event F1" has one universal definition:

- An alert is the onset of a contiguous positive episode.
- Onset inside an eligible labeled event detects that event once.
- Additional alert episodes inside that event count as false alerts.
- Onsets outside windows are false alerts; undetected windows are false negatives.
- A permanent positive prediction cannot receive credit for every event it spans.
- Missing/unscoreable rows break episodes and are excluded from denominators.
- Windows crossing a section boundary are clipped; reported delay starts at the
  first eligible labeled reading inside the section. These are censored local
  delays, not necessarily delay from the original physical anomaly onset.
- No point adjustment or ground-truth expansion is applied.

Report event precision/recall/F1, event counts, false alert episodes, mean detected
event delay, point F1, false-positive point rate and one-step forecast MAE.
Aggregate micro event F1 uses summed TP/FP/FN; macro includes zero-event series
and is labeled accordingly. Processing throughput includes batched forecast
computation and is not certified online-window latency. Search timings are
recorded separately.

Replace this matching policy with official organizer rules before interpreting
scores as competition results. The canonical prediction schema also needs an
organizer-specific conversion once their required schema is available.

## 8. Reporting and deployment

Write per-horizon raw forecast, corrected center, lower/upper range, signed and
squared errors, normalized score, combined score, threshold, decision and a
numerical explanation. Charts show the short-horizon range, all alerts and known
test windows. Per-horizon ranges remain available in CSVs.

Save the validation-selected model, detector state after test replay, and enough
observed history to continue the same signal. New-data inference requires no
labels, does not refit, and rejects timestamps overlapping saved history. Its
updated state is saved separately to avoid overwriting the original artifact.

## 9. Credit and limitations

Credit Zhu, Carpentier and Verbeke (ICLR 2026) for the evaluated one-liner baseline
and multiple-horizon motivation. Credit Numenta and original contributors for
NAB. Use scikit-learn's Ridge implementation. The pipeline is an engineering
implementation/adaptation, not a claimed new forecasting or detection algorithm.

The paper does not prove Ridge is best on NAB. More history or more horizons may
hurt. Normal repeated peaks can look unusual; slow or predictable faults can be
missed. All horizons can eventually incorporate a persistent anomaly. Fixed
chronological sections may have few events. No best-result or winning guarantee
is possible before running and inspecting validation and held-out evaluation.
