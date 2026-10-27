# Preparation status

## v4 preparation - full run left for the user

Implemented the frozen v2 control, the A/B/C factorial comparison, two-sided
CUSUM, separate confirmation policies and an explicit single-exceedance variant.
Adoption uses earlier-to-later real development folds, historical source cutoffs,
paired stream/source-group uncertainty, a recall floor and a soft alert budget.
Uncertain changes retain v2. Synthetic injections run after selection freezes and
never train or choose models. One-reading latency is measured separately.

All 29 regression tests passed (59.128 seconds). The latest v4-specific checks
also passed after adding saved-verifier inference and channel evidence. Read-only
checks passed for all 58 series and 70 downloaded hashes. Every calibration half
has at least 112 observed rows. The preserved v2 selector's AST and configuration
match commit 9a76ee2; cached development choices matched on all 58 streams.

No full real-data v4 fitting, policy evaluation, test replay or measured F1 was
started during preparation. Existing results remain intact. Run run.py manually;
new outputs go to results/experiment_v4_<timestamp>/. See V4_PREPARATION.json,
README.md and V4_METHODOLOGY.md. Older status sections below are historical.

## v3 current upgrade

Added bounded histogram gradient boosting, training-normal shape/state novelty
and sparse-event selection shrinkage with local false-alert checks. Accelerated
score arithmetic preserves sequential decision/update order, verified against
an independent scalar oracle. All 20 tests initially passed in 20.416 seconds;
a further sparse-transfer regression also passed. All 58 data/fold checks passed.

The user-authorized v3 run completed all 58 streams in 47.8 minutes, with
144,288 fold trials and zero stream failures. Fifteen candidate-fold fits were
rejected for insufficient clean training windows. Event-F1 decreased from
.354430 to .341463: still 14/22 detected events, 46 false alerts and 8 misses.
This is an unsuccessful performance experiment, not a claimed improvement.
Previous runs remain intact. See V3_RESULTS.md.

The final suite passed all 23 tests (24.957 seconds). A subsequent development-only
audit compared 13 selection policies using earlier-to-later folds and still
favored the existing policy. It did not change the production model choices.
Model/scoring/selection source hashes match the full run. Only the read-only
audit CLI was added afterward. Older preparation-only entries below are history.

v2 finished successfully on all 58 series: selected event-F1 .354430,
14/22 true events detected, 43 false/duplicate alerts, 8 missed events.

## v2 upgrade status

The upgrade implements three chronological folds, source/time-isolated pooling,
seasonal naive and seasonal Ridge inputs, empirical/MAD thresholds, incident
hysteresis/recovery/cooldown, and optional training-label-only window cleanup.
All 17 synthetic regression tests passed (21.929 seconds), including two-stream
CLI selection freezing and unlabeled continuation. Read-only checks passed for
all 58 series. No full v2 real-data experiment was started by the assistant.
Existing results remain unchanged. See BASELINE.md and the revised README.
No zero-overfitting or unseen-test performance guarantee is claimed.

## Latest verification after first-run failure

The first user-started NAB run failed on all series because scikit-learn's
single-target Ridge predictions were one-dimensional. Forecast output now
explicitly retains the horizon column. All 9 synthetic regression tests pass,
including the four-method pipeline and saved-model loading (4.809 seconds).
The full NAB experiment was not restarted. No completed real-data evaluation
scores are available. Original preparation-only checks below are historical.

## Original preparation checks

- Application entry point: `run.py`.
- Public NAB snapshot downloaded at pinned commit
  `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140`.
- 58 series; 365,558 raw readings; 365,509 after explicit timestamp normalization.
- 49 repeated timestamp rows are removed in memory; raw downloads are unchanged.
- All 70 downloaded file hashes verified. Every labeled anomaly window has
  readings in the normalized data. Default split lengths are feasible.
- Application/test Python files parse and compile without executing their code.
- No machine-specific paths in application Python or config.
- Authors' GitLab repository inspected; selected original reference files and
  their license preserved with source links and commit.
- **No training, model inference, threshold search, model evaluation, runtime
  tests or end-to-end application execution performed during preparation.**
- No trained checkpoints, measured model scores or claimed best parameters yet.
- Dependency installation and runtime checks must be performed on the target PC.
- The organizer's private dataset, hidden test, exact evaluator and output schema
  were not available; public data and documented local metrics are development
  substitutes. Align them with the starter pack before final submission.

See `DATA_VALIDATION.json` for preparation checks, `README.md` for commands,
and `METHODOLOGY.md` for formulas and the experiment protocol.
