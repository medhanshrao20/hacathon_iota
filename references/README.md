# Paper/code inspection and attribution

## Primary paper

Xiaokun Zhu, Louis Carpentier and Mathias Verbeke (2026).
**When Foundation Models are One-Liners: Limitations and Future Directions for
Time Series Anomaly Detection.** ICLR 2026.

- Paper: https://openreview.net/forum?id=H27kvyG4qf
- Official proceedings: https://proceedings.iclr.cc/paper_files/paper/2026/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html
- Authors' implementation: https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/publications/iclr2026_timeseriesfoundationmodelsad/
- Inspected commit: `dcbbd9fbeaabfb27ad084ffa4351a2418ea1dab9`
- Large intermediate results: https://doi.org/10.48804/MCVI4N

The authors' own code is on **GitLab**, not a newly discovered GitHub mirror.
The paper also links these GitHub projects:

| Link | Relevance here |
|---|---|
| https://github.com/TheDatumOrg/TSB-AD | Benchmark and metric reference; paper uses TSB-AD-U, not our exact NAB task |
| https://github.com/moment-timeseries-foundation-model/moment | Optional embedding research; not required for this CPU build |
| https://github.com/google-research/timesfm | Foundation forecasting models; not needed for the initial experiment |
| https://github.com/amazon-science/chronos-forecasting | Same: optional comparison, not a required dependency |
| https://github.com/Time-MoE/Time-MoE | Same |
| https://github.com/ibm-granite/granite-tsfm | TSPulse-related code; not imported here |

## Useful files inspected

Selected upstream files are preserved unchanged in `paper_code/`, with the
authors' MIT license. They are **reference material only** and are never
imported/executed by Sentinel. Their CLI dependencies are not installed.

- `evaluate_baselines.py`: confirms the Last-3 causal mean-error formula. Its
  centered variance/Centered baseline and initial backfills are unsuitable for
  immediate streaming decisions. Sentinel uses past-only calculations.
- `long_horizon_predictions_to_scores.py`: computes future-block MSE and assigns
  it back to the block start. Sentinel scores mature forecasts at the actual
  arrival timestamp instead; this is a streaming adaptation, not score parity.
- `evaluate_baselines_ensemble.py`: reference for combining baseline scores;
  Sentinel calibrates its combined threshold without test labels.
- `README.md`: upstream reproduction instructions and bibliography.

We do not download the 11.5 GB intermediate archive or foundation weights: those
are not needed to implement the useful causal ideas or run this project.

## Dataset credit

Numenta Anomaly Benchmark: https://github.com/numenta/NAB
Dataset descriptions and license are retained under `data/nab/`. The pinned
download manifest records original URLs, commit and per-file hashes.

Ahmad, Lavin, Purdy and Agha (2017), *Unsupervised real-time anomaly detection
for streaming data*, Neurocomputing. https://doi.org/10.1016/j.neucom.2017.04.070

## Presentation wording

"We implement statistical baselines and a streaming multi-horizon forecasting
detector inspired by Zhu et al. (ICLR 2026), using Ridge regression, calibrated
error ranges and interpretable alerts. We evaluate these methods on NAB."

Do not describe the cited baseline, horizon ensemble idea or Ridge regression
as an original algorithm. Do not describe this project as an exact reproduction
of the paper's foundation-model results.

## v3 implementation references

The added tree and nearest-neighbor methods use established scikit-learn
implementations; they are not claimed as methods introduced by the paper.

- [Histogram gradient boosting regressor](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html):
  CPU nonlinear forecasting; fixed rounds with early_stopping=False.
- [NearestNeighbors](https://scikit-learn.org/1.6/modules/generated/sklearn.neighbors.NearestNeighbors.html):
  distance to a bounded, training-only reference of normal windows.

The hybrid's shape/variation channels are our engineering combination of
standard features and nearest-neighbor distances. They are not an exact matrix
profile implementation or a foundation-model reproduction.
