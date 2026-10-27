"""Dataset/CLI adapter; executes the original score functions without rewriting them."""
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from run import METRICS, ROOT, required_rows, save


def original_functions(filename, names):
    """Exclude the upstream module's argparse/TSB-dataset top-level side effects."""
    path = ROOT / 'upstream' / 'scripts' / filename
    tree = ast.parse(path.read_text(encoding='utf8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != set(names):
        raise ValueError(f'Upstream functions missing in {filename}')
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def baseline_scores(data, name, window):
    namespace = original_functions('evaluate_baselines.py',
                                   ['moving_window_variance', 'squared_difference', 'get_scores'])
    return namespace['get_scores'](data, name, window)


def ensemble(first, second):
    normalize = original_functions('evaluate_baselines_ensemble.py', ['min_max'])['min_max']
    # Preserve upstream 0/0 on constant arrays; report undefined, never invent a score.
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.maximum(normalize(first), normalize(second))


def horizon_scores(data, predictions, window, destination):
    path = ROOT / 'upstream/scripts/long_horizon_predictions_to_scores.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    outer = next(n for n in tree.body if isinstance(n, ast.For))
    loop = next(n for n in outer.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'horizon')
    namespace = {'np': np, 'os': __import__('os'), 'data': data, 'combined_array': predictions,
                 'win_size': window, 'horizons': 2**np.arange(10), 'path_scores': str(destination)}
    destination.mkdir(parents=True, exist_ok=True)
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), 'exec'), namespace)


def measure(scores, labels):
    result = dict.fromkeys(METRICS)
    if len(scores) != len(labels):
        raise ValueError('Score/label length mismatch')
    if not np.isfinite(scores).all():
        return result, 'undefined_nonfinite_scores_in_original_formula'
    if len(np.unique(labels)) < 2:
        return result, 'undefined_single_class_labels'
    from TSB_AD.evaluation.metrics import get_metrics
    measured = get_metrics(scores, labels)
    for key in METRICS:
        value = float(measured[key])
        result[key] = value if np.isfinite(value) else None
    return result, 'ok' if all(v is not None for v in result.values()) else 'partially_undefined_original_metrics'


def evaluate(folder, job, file):
    frame = pd.read_csv(folder / 'data/TSB-AD-U' / file).dropna()
    data = frame.Data
    labels = frame.Label.astype(int).to_numpy()
    stem = Path(file).stem
    output = folder / 'reproductions/intermediate' / job['id'] / 'scores'
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    if len(data) < required_rows(job):
        save(folder / 'metrics' / f'{job["id"]}__{file}.json', {
            'file_name': file, 'method': job['id'], 'status': 'undefined_too_short_for_published_window',
            'rows': len(data), 'evaluation': 'offline_full_series_oracle_benchmark', **dict.fromkeys(METRICS)})
        return
    if job['kind'] == 'baseline':
        scores = baseline_scores(data, job['name'], job['window'])
        path = output / f'{stem}.npy'
        np.save(path, scores)
        paths.append((job['id'], path))
    elif job['kind'] == 'baseline_ensemble':
        scores = ensemble(baseline_scores(data, 'var', 96), baseline_scores(data, 'last', 3))
        path = output / f'{stem}.npy'
        np.save(path, scores)
        paths.append((job['id'], path))
    elif job['kind'] == 'long_horizon':
        prediction_dir = output.parent / 'predictions' / stem
        chunks = sorted(prediction_dir.glob('*.npy'), key=lambda p: int(p.stem))
        if not chunks:
            raise FileNotFoundError(f'No original predictions in {prediction_dir}')
        predictions = np.concatenate([np.load(p) for p in chunks], axis=0)
        horizon_scores(data.to_numpy(), predictions, job['window'], output / stem)
        paths = [(f'{job["id"]}_h{h}', output / stem / f'{h}.npy')
                 for h in 2**np.arange(10) if (output / stem / f'{h}.npy').exists()]
        if (output / stem / '64.npy').exists():
            combined = ensemble(np.load(output / stem / '1.npy'), np.load(output / stem / '64.npy'))
            path = output / stem / 'ensemble_1_64.npy'
            np.save(path, combined)
            paths.append((job['id'] + '_max_h1_h64', path))
    elif job['kind'] == 'embedding':
        paths = [(p.parent.parent.name, p) for p in sorted(
            (folder / 'reproductions/intermediate').glob(f'moment_embedding_64_*/scores/{stem}.npy'))]
        if not paths:
            raise FileNotFoundError('No embedding detector scores')
    else:
        paths = [(job['id'], output / f'{stem}.npy')]
    for method, path in paths:
        scores = np.load(path)
        metrics, status = measure(scores, labels)
        save(folder / 'metrics' / f'{method}__{file}.json', {
            'file_name': file, 'method': method, 'status': status, 'rows': len(data),
            'evaluation': 'offline_full_series_oracle_benchmark', **metrics})


if __name__ == '__main__':
    evaluate(Path(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3])
