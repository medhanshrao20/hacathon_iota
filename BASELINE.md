# Preserved public-test baselines

Original completed run: `results/experiment_20261026T170123_524711Z`.
The run's existing files remain unchanged and ignored by Git. Do not delete them
to make a v2 comparison. These numbers use our provisional local event metric.

| Method | Detected events | False/duplicate alerts | Event F1 |
|---|---:|---:|---:|
| Recent mean | 14 / 22 | 122 | .177215 |
| Single-horizon Ridge | 15 / 22 | 146 | .163934 |
| Multi-horizon Ridge | 17 / 22 | 172 | .161137 |
| Rolling median/MAD | 12 / 22 | 246 | .085714 |
| Per-stream validation-selected methods | 12 / 22 | 116 | .160000 |

The selected system raised 128 alert episodes: 12 first correct detections,
97 alerts outside labeled windows and 19 duplicate alerts. It missed ten events.
31 series had no validation events; 41 had no events in the test section.

v2 adds seasonality, stronger validation and calibrated incident handling.
The same public test is already known: v2 results are exploratory comparisons,
not a new independent test. Independent confirmation needs hidden/unseen data.

## Completed v2 baseline

Run: `results/experiment_v2_20261026T190642_888274Z`; all 58 streams succeeded.

| Method | Detected events | False/duplicate alerts | Event F1 |
|---|---:|---:|---:|
| Selected system | 14 / 22 | 43 | .354430 |
| Recent mean | 14 / 22 | 38 | .378378 |
| Single-horizon Ridge | 11 / 22 | 33 | .333333 |
| Multi-horizon Ridge | 12 / 22 | 47 | .296296 |
| Rolling median/MAD | 10 / 22 | 167 | .100503 |
| Seasonal naive | 15 / 22 | 57 | .319149 |

Selected precision .245614, recall .636364. The 57 proposed alert episodes
include 14 correct first detections, 36 outside-window alerts and 7 duplicates.
Eight events were missed. This baseline is preserved for comparison with v3.
