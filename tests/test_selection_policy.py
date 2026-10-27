"""Selection policy audits must use only available historical fold evidence."""
import copy
import unittest

import numpy as np
import pandas as pd

from sentinel.selection_policy import historical_pool, prefixes


class SelectionPolicy(unittest.TestCase):
    def test_prefix_cannot_read_later_fold_results_or_labels(self):
        stream = dict(group="A", relative="A/a.csv", ids=np.full(100, -1),
                      stamps=pd.Series(pd.date_range("2026-01-01", periods=100, freq="min")),
                      folds=[(20,40,55),(35,55,70),(50,70,85)])
        row = dict(detected_events=1, false_alerts=2, missed_events=0, normal_points=100,
                   true_events=1, event_f1=.5)
        by_fold = {i:{"z":copy.deepcopy(row), "a":copy.deepcopy(row)} for i in range(3)}
        params = {"z":{}, "a":{}}
        first = prefixes(stream, params, by_fold)[0]
        self.assertEqual(list(first["summaries"]), ["z", "a"])
        by_fold[1]["z"]["false_alerts"] = 9999
        by_fold[2]["z"]["detected_events"] = 9999
        stream["ids"][55:] = 5
        other = prefixes(stream, params, by_fold)[0]
        self.assertEqual(first["summaries"], other["summaries"])
        self.assertEqual(other["selection_events"], 0)
        self.assertEqual(other["available_until"], stream["stamps"].iloc[54])

    def test_pool_rejects_related_sources_and_future_prefixes(self):
        metrics = dict(detected_events=1, false_alerts=0, missed_events=0,
                       normal_points=100, true_events=1, event_f1=1.)
        past = dict(group="B", summaries={"x":{"validation":metrics}},
                    available_until=pd.Timestamp("2026-01-01"))
        future = copy.deepcopy(past)
        future["available_until"] = pd.Timestamp("2027-01-01")
        future["summaries"]["x"]["validation"]["detected_events"] = 999
        related = copy.deepcopy(future)
        related.update(group="A", available_until=pd.Timestamp("2025-01-01"))
        target = dict(group="A", relative="A/target.csv")
        pool = historical_pool(target, pd.Timestamp("2026-06-01"), [[past,future],[related]])
        self.assertEqual(pool["A/target.csv"]["x"]["detected_events"], 1)
