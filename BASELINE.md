# Preserved v1 public-test baseline

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
