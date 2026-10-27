# Controlled v4 methodology

## Control and temporal ordering

The v2 predictor grid is pinned in controls/v2_config.json and its selector is
preserved in sentinel/control_v2.py. The shared scoring implementation remains
the checked causal implementation; optional policies live in a subclass and do
not alter the original Detector. The unchanged baseline selector matched all 58
saved v2 development choices during read-only preparation.

Original folds remain 0-20/20-40/40-55%, 0-35/35-55/55-70% and
0-50/50-70/70-85% for training/calibration/validation; test is 85-100%.
The usual v2 search is completed for all streams. To assess a policy, choose its
forecaster from fold 0 only and score on fold 1, then choose from folds 0/1 and
score on fold 2. A peer contributes only its completed historical folds whose
last timestamp is no later than the target's selection boundary. Exclude the
target's entire source group. Ties follow original candidate log order.

At final adoption time, policy evidence is restricted to the target's own
development results plus other-source results that predate its test boundary.
One global policy chosen from all wall-clock dates would leak future peer data
into earlier targets; the implementation therefore uses an as-of-time cohort.

## Optional decision channels

Forecasting windows, targets, training-label cleanup and periods retain the
v2 definitions in METHODOLOGY.md. There are no new large regression models.

Revised calibration divides chronological out-of-training errors into reference
and later tuning blocks, each with at least min_calibration_rows observed errors.
The reference uses per-channel empirical deviation scales. Candidate combined
point thresholds come from declared tuning quantiles .99/.995/.999. Replay each
complete policy on the later block and minimize rate excess over the declared
budget, then choose the most sensitive tied threshold. A threshold is a soft
operating target, not a quota. All calibrations are label-free. Calibration may
contain faults; reported calibration rates are not certified false-alarm rates.

The spike level is the maximum finite calibration score, bounded below by the
chosen point threshold. There is no fixed 2x multiplier or test-peak tuning.
Control/A retain original calibration; B/C implement the separate calibration
factor. A does not consume the calibration suffix again as extra online history.
Revised policies warm their state by replaying the later calibration block once.

CUSUM uses a separate robust center/scale from the reference block. For signed
standardized forecast errors z, both directions use:

    Cplus[t]  = max(0, Cplus[t-1]  + z[t] - k)
    Cminus[t] = max(0, Cminus[t-1] - z[t] - k)

The default drift allowance k=.5 is declared, not claimed optimal. The alarm
level is derived from the reference CUSUM path's declared quantile, with a
numerical lower bound of one. Accumulation precedes the point threshold, so a
sequence of weak errors can trigger it. State resets naturally at zero, across
missing readings and after recovery. During an open incident it does not create
additional onsets. CUSUM-driven incidents require quiet standardized errors as
well as the normal recovery test before closing. Original cooldown still applies
to CUSUM; only severe spikes can bypass it. All adaptation follows the final
combined decision, preventing a CUSUM alert from being accepted as normal history.

The optional secondary detector fits a bounded normal-window shape/state index
on the training prefix only. Its distances use a separate reference empirical
CDF and confirmation percentile (.99 by default), not half another threshold.
The complete combined policy is calibrated; independence between channels is
not assumed. The G variant explicitly uses 1-of-1 opening. No 1-of-2 rule is
misrepresented as independent confirmation.

## Adoption and uncertainty

Control, A/B/C and G are the first comparisons; D adds only CUSUM to C. E/F add
only confirmation to B/C. For every candidate require complete paired folds,
enough historical streams/events, a minimum .02 F1 gain, no recall loss, and
the development alert budget. Resample complete paired stream counts including
zero-event streams. Also resample source groups as a dependence sensitivity check.
Default intervals use 500 resamples, seed 42 and 95% confidence. Both lower
bounds must be positive. D/E/F/G also need a positive paired incremental interval
against their simpler parent. If simpler accepted policies exist, verifiers are
not adopted. Keep v2 when any required evidence is missing or uncertain.

These gates are conservative heuristics, not a proof of no overfitting. Rare
events, related streams and few source groups limit confidence. Future public
test results cannot change choices made by these gates.

## Synthetic evaluation and reporting

Save all final policies/parameters before generating synthetic results or test
replay. Inject only into copies of clean development blocks after fitting. Each
case recomputes forecasts from the perturbed past; reusing clean future forecasts
would create an unrealistically easy detection task. Use the real prior detector
state at the block start, never state from the end of the validation window.

Stress types are spikes, dips, flatlines, level shifts, variance changes and
gradual drift. Seeds 17/83 and durations 4/16 are declared. The later seed and
flatline/variance types are marked held-out stress partitions; no stress case
is used for fitting, threshold tuning or policy adoption. Nonnegative training
signals constrain injected values to remain nonnegative. Flatline cases on
already-flat signals and streams without long clean blocks are explicitly skipped.
Synthetic scores remain in a separate file and never inflate the real event count.

The event matcher is unchanged. Report false episodes per 10,000 normal readings
and per normal-exposure day; nominal cadence caps gaps, excluding unobserved
exposure. Detection delays remain clipped at evaluation boundaries. Runtime is
measured with one-reading feature/forecast/scoring/state calls, P50 and P95,
separately from batch throughput and file I/O. Test bootstrap intervals are
descriptive output only. Tail fits, transforms, EWMA and escalation are deferred.

## Credit and technical basis

The Last-3 and multi-horizon motivation remains credited to Zhu et al. (ICLR2026).
All added rules are combinations of established statistical techniques.
CUSUM background: https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm
This project does not claim a new CUSUM method or exact reproduction of the paper.
