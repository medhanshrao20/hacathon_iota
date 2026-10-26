# Preparation status

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
