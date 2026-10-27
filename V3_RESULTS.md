# Sentinel v3 measured comparison

**This attempted upgrade did not improve selected-system event-F1. Preserve v2 as the current comparison reference.**

All 58 NAB streams completed in 47.8 minutes. Scores use the same local event metric as v2.

| Metric | Previous v2 | New v3 |
|---|---:|---:|
| Event F1 | 35.44% | 34.15% |
| Precision | 24.56% | 23.33% |
| Recall | 63.64% | 63.64% |
| Correctly detected events | 14 | 14 |
| False / duplicate alerts | 43 | 46 |
| Missed events | 8 | 8 |
| Total proposed alerts | 57 | 60 |

## Models

Added a shallow gradient-boosted next-reading forecast and a nearest-normal-window hybrid that checks shape, level, variation and flatlines. Existing statistical and Ridge models remain candidates. Selection now checks local development false alerts when borrowing evidence from other source groups.

| Family | Detected | False alerts | Missed | Event F1 |
|---|---:|---:|---:|---:|
| boosted_forecast | 11 | 31 | 11 | 34.38% |
| pattern_hybrid | 16 | 54 | 6 | 34.78% |
| recent_mean | 14 | 38 | 8 | 37.84% |
| ridge_multi | 12 | 34 | 10 | 35.29% |
| ridge_single | 11 | 35 | 11 | 32.35% |
| rolling_median_mad | 11 | 48 | 11 | 27.16% |
| seasonal_naive | 15 | 47 | 7 | 35.71% |

The selected system uses a family chosen on development data for each stream. The family table is a retrospective comparison, not a reason to switch winners using test scores.

## Verification and limitations

23 tests passed. All 58 data/fold checks passed. 144,288 chronological fold trials were recorded. 15 candidate-fold failures were explicitly rejected. No stream failed. Model, scoring, metric and selection source hashes match the full run. A read-only selection audit command was added afterward; it does not change normal runs. All selections were saved before any test predictions.

## Remaining gaps

False alerts remain the main problem. The pattern hybrid found 16/22 events but also raised 54 false alerts. Adding model capacity did not solve sparse-event selection or transfer between different signal types. Next work needs more labeled incidents from comparable sources and an untouched evaluation set. Do not choose thresholds or switch the deployed family just because one public test score looks better.

A follow-up audit compared 13 declared selection policies using only earlier-to-later development folds. It selected `original_transfer` and did not justify changing the model choices. See `SELECTION_POLICY_DEVELOPMENT.json` for all development scores. No additional test replay was performed.

Training uses only each fold’s historical training prefix. Known training anomaly annotations are enabled; this is not a fully unsupervised claim. Calibration does not use labels. Test labels only evaluate frozen predictions. The public test was previously inspected, so these results are exploratory; hidden organizer data is still needed to assess generalization. Passing tests does not guarantee zero overfitting.

Point accuracy: 86.47%; point F1: 20.17%. These measure individual readings; event-F1 measures finding incidents without excessive alerts.

## Saved files

Project: `project folder`
New results: `results/experiment_v3_20261027T010048_066974Z/`
Open `report.html` there for charts, selected parameters and each method’s predictions. `V3_SELECTED_PARAMETERS.json` preserves each stream’s development-selected settings. Previous result folders remain intact. `Sentinel_PS2_Portable.zip` contains portable code, data and measured summaries; it excludes environment folders and large result charts.
