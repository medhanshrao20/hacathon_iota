# Benchmark result import

The imported run contains 12 complete methods on 58 NAB streams and 18 finished
streams for a further MOMENT-512 configuration. The next stream was interrupted;
this is not a completed 48-configuration suite. Partial results are retained in
the individual JSON records and excluded from the supplied summary.

Best mean offline event-F1: **73.64%**, variance window64.
Best mean VUS-PR: **45.82%**, variance window256.
Summary metrics average 52 evaluable streams; six all-normal streams have
undefined two-class benchmark metrics. The numeric score arrays are retained
for all streams, including normal streams.

These are full-series, offline benchmark results. Centered variance uses future
observations; threshold-dependent benchmark F1 uses evaluation-label-selected
thresholds. It is not held-out operational streaming F1 and does not establish
false-alert performance on normal streams. No foundation-model fine-tuning was
performed. The best observed setting was identified after inspecting results.

Separate streaming context: 14 of 22 events detected, 43 false alerts,
event-F1 35.44%, precision 24.56%,
recall 63.64%. It uses a different evaluator and test
scope. Public test sections were previously inspected; this is exploratory
evidence, not an untouched organizer test result.

`IMPORT_MANIFEST.json` records the archive SHA256 and hashes of copied artifacts.
Original summaries, all714 individual metric records and score arrays are
preserved. Machine-specific logs and duplicate executables are excluded from
this published bundle; the raw import is retained locally. `timing.jsonl` records
offline whole-stream job time, not streaming per-window latency.

The presentation uses the required ten-slide Challenge2 structure and separates
offline results from operational metrics. It does not claim a new statistical
algorithm or a hidden-test score.
