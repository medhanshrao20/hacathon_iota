"""Synthetic checks only. Never scores or fits a NAB stream."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run
from evaluate import baseline_scores, ensemble, evaluate, horizon_scores, measure


class PaperChecks(unittest.TestCase):
    def test_original_functions_and_boundaries(self):
        x = pd.Series(np.arange(120, dtype=float)**1.3)
        for w in (1, 3):
            upstream = ((x.to_numpy()-x.rolling(w+1).mean().bfill().ffill().to_numpy())*(w+1)/w)**2
            np.testing.assert_array_equal(baseline_scores(x, 'last', w), upstream)
        np.testing.assert_array_equal(baseline_scores(x, 'var', 64), x.rolling(64, center=True).var().bfill().ffill())
        changed = x.copy()
        changed.iloc[60:] += 100
        np.testing.assert_array_equal(baseline_scores(x, 'last', 3)[4:60], baseline_scores(changed, 'last', 3)[4:60])
        self.assertFalse(np.array_equal(baseline_scores(x, 'centered', 5)[:60], baseline_scores(changed, 'centered', 5)[:60]))
        np.testing.assert_array_equal(ensemble(np.array([0., 1., 4.]), np.array([3., 2., 0.])), [1., 2/3, 1.])
        self.assertTrue(np.isnan(ensemble(np.ones(4), np.ones(4))).all())

    def test_original_horizon_indexing(self):
        x = np.arange(120, dtype=float)
        predictions = np.zeros((104, 512))
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            horizon_scores(x, predictions, 16, folder)
            for h in (1, 2, 4, 8, 16, 32, 64):
                expected = np.array([np.mean(x[16+i:16+i+h]**2) for i in range(120-16-h+1)])
                expected = np.pad(expected, (16, h-1), constant_values=(expected[0], expected[-1]))
                np.testing.assert_array_equal(np.load(folder / f'{h}.npy'), expected)

    def test_data_and_protocol_plan(self):
        self.assertEqual(len(run.experiment_plan('baselines')), 10)
        self.assertEqual(len(run.experiment_plan('full')), 48)
        run.verify_sources()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nab = root / 'nab'
            (nab / 'data/source').mkdir(parents=True)
            (nab / 'labels').mkdir()
            pd.DataFrame({'timestamp': ['2026-01-02', '2026-01-01', '2026-01-02'],
                          'value': [2, 1, 99]}).to_csv(nab / 'data/source/sample.csv', index=False)
            run.save(nab / 'labels/combined_windows.json', {'source/sample.csv': [['2026-01-02', '2026-01-02']]})
            with patch.object(run, 'ROOT', root):
                rows = run.prepare(nab)
            data = pd.read_csv(root / 'data/TSB-AD-U/source__sample.csv')
            self.assertEqual(rows[0]['removed_rows'], 1)
            self.assertEqual(data.Data.tolist(), [1, 2])
            self.assertEqual(data.Label.tolist(), [0, 1])
            short = {'kind': 'baseline', 'id': 'var_1024', 'name': 'var', 'window': 1024, 'env': 'metrics'}
            evaluate(root, short, 'source__sample.csv')
            saved = json.loads((root / 'metrics/var_1024__source__sample.csv.json').read_text())
            self.assertEqual(saved['status'], 'undefined_too_short_for_published_window')

    def test_synthetic_cli_roundtrip(self):
        # Exercise subprocess scoring + original TSB metrics, logs and resumption.
        original_root = run.ROOT
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('upstream', 'third_party'):
                shutil.copytree(original_root / name, root / name)
            for name in ('UPSTREAM_MANIFEST.json', 'evaluate.py', 'run.py'):
                shutil.copy2(original_root / name, root / name)
            nab = root / 'nab'
            (nab / 'data/source').mkdir(parents=True)
            (nab / 'labels').mkdir()
            dates = pd.date_range('2026-01-01', periods=320, freq='min', tz='UTC')
            data = np.sin(np.arange(320)/12)
            data[160:170] += 8
            pd.DataFrame({'timestamp': dates, 'value': data}).to_csv(nab / 'data/source/synthetic.csv', index=False)
            pd.DataFrame({'timestamp': dates, 'value': np.sin(np.arange(320)/12)}).to_csv(
                nab / 'data/source/normal.csv', index=False)
            run.save(nab / 'labels/combined_windows.json', {
                'source/synthetic.csv': [[str(dates[160]), str(dates[169])]], 'source/normal.csv': []})
            job = {'kind': 'baseline', 'id': 'last_3', 'name': 'last', 'window': 3, 'env': 'metrics'}
            args = argparse.Namespace(suite='baselines', setup=False, prepare=False, check_data=False,
                                      plan=False, resume=None, nab_dir=nab, only=None)
            with patch.object(run, 'ROOT', root), patch.object(run, 'experiment_plan', return_value=[job]), \
                    patch.object(run, 'python_for', return_value=Path(sys.executable)):
                run.run(args)
                folder = next((root / 'runs').iterdir())
                self.assertEqual(json.loads((folder / 'status.json').read_text())['status'], 'completed')
                metrics = pd.read_csv(folder / 'metrics.csv')
                self.assertEqual(set(metrics.status), {'ok', 'undefined_single_class_labels'})
                self.assertTrue(metrics['VUS-PR'].dropna().between(0, 1).all())
                summary = pd.read_csv(folder / 'summary.csv')
                self.assertEqual(summary.total_series.tolist(), [2])
                self.assertEqual(summary.valid_series.tolist(), [1])
                args.resume = folder
                run.run(args)
                self.assertEqual(len((folder / 'timing.jsonl').read_text().splitlines()), 2)
            empty, status = measure(np.zeros(20), np.zeros(20))
            self.assertEqual(status, 'undefined_single_class_labels')
            self.assertTrue(all(v is None for v in empty.values()))


if __name__ == '__main__':
    unittest.main()
