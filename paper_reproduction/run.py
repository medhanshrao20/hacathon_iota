"""Run the paper's published code on NAB, independently of Sentinel."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import venv

ROOT = Path(__file__).resolve().parent
METRICS = ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 'Standard-F1',
           'PA-F1', 'Event-based-F1', 'R-based-F1', 'Affiliation-F']
IMPORT_CHECKS = {
    'metrics': 'import numpy,pandas,scipy,sklearn',
    'moment': 'from momentfm import MOMENTPipeline; from pyod.models.knn import KNN',
    'chronos': 'from autogluon.timeseries import TimeSeriesPredictor,TimeSeriesDataFrame',
    'timesfm': 'from timesfm import TimesFm,TimesFmHparams,TimesFmCheckpoint',
    'timemoe': 'from transformers import AutoModelForCausalLM; import accelerate,torch',
}


def configure_storage():
    """Keep setup scratch files and downloaded weights beside this project."""
    cache = ROOT / '.cache'
    paths = {
        'TEMP': cache / 'tmp', 'TMP': cache / 'tmp', 'TMPDIR': cache / 'tmp',
        'PIP_CACHE_DIR': cache / 'pip',
        'HF_HOME': cache / 'huggingface',
        'HF_HUB_CACHE': cache / 'huggingface' / 'hub',
        'HUGGINGFACE_HUB_CACHE': cache / 'huggingface' / 'hub',
        'TRANSFORMERS_CACHE': cache / 'huggingface' / 'hub',
        'TORCH_HOME': cache / 'torch', 'XDG_CACHE_HOME': cache / 'other',
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    # tempfile may already have cached the Windows system temp directory.
    tempfile.tempdir = str(paths['TEMP'])


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def verify_sources():
    records = json.loads((ROOT / 'UPSTREAM_MANIFEST.json').read_text(encoding='utf8'))
    for record in records:
        actual = hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest()
        if actual != record['sha256']:
            raise ValueError(f"Upstream source was changed: {record['path']}")
    return records


def python_for(family):
    home = ROOT / '.envs' / family
    return home / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def setup(families):
    configure_storage()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Use Python 3.11, as in the paper and legacy TimesFM API.')
    for family in families:
        executable = python_for(family)
        if not executable.exists():
            venv.EnvBuilder(with_pip=True).create(executable.parent.parent)
        lock = ROOT / 'requirements' / f'{family}.lock.txt'
        requirements = lock if lock.exists() else ROOT / 'requirements' / f'{family}.txt'
        subprocess.run([str(executable), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
        subprocess.run([str(executable), '-c', IMPORT_CHECKS[family]], check=True)
        # Record resolved versions; authors did not publish complete environment locks.
        freeze = subprocess.check_output([str(executable), '-m', 'pip', 'freeze'], text=True)
        (executable.parent.parent / 'resolved_requirements.txt').write_text(freeze, encoding='utf8')


def prepare(nab):
    """Data/schema adaptation only. No models, scores or test-driven choices."""
    import numpy as np
    import pandas as pd

    windows = json.loads((nab / 'labels/combined_windows.json').read_text(encoding='utf8'))
    output = ROOT / 'data' / 'TSB-AD-U'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((nab / 'data').rglob('*.csv')):
        key = path.relative_to(nab / 'data').as_posix()
        frame = pd.read_csv(path)
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True, errors='raise')
        n_raw = len(frame)
        frame = frame.sort_values('timestamp', kind='stable').drop_duplicates('timestamp', keep='first')
        frame['value'] = pd.to_numeric(frame['value'], errors='coerce').replace([np.inf, -np.inf], np.nan)
        # Upstream scripts use dropna(), not forward-fill or interpolation.
        frame = frame.dropna(subset=['timestamp', 'value']).reset_index(drop=True)
        if key not in windows:
            raise ValueError(f'No label entry for {key}; refusing to invent normal labels')
        label = np.zeros(len(frame), dtype=int)
        for start, end in windows[key]:
            start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
            label[((frame.timestamp >= start) & (frame.timestamp <= end)).to_numpy()] = 1
        name = key.replace('/', '__')
        converted = output / name
        pd.DataFrame({'Data': frame.value, 'Label': label}).to_csv(converted, index=False)
        # Sidecar timestamps must not enter the upstream numeric model input.
        pd.DataFrame({'timestamp': frame.timestamp}).to_csv(output.parent / f'{name}.timestamps.csv', index=False)
        records.append({'file_name': name, 'nab_path': key, 'rows': len(frame), 'raw_rows': n_raw,
                        'removed_rows': n_raw-len(frame), 'positive_readings': int(label.sum()),
                        'label_windows': len(windows[key]),
                        'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'converted_sha256': hashlib.sha256(converted.read_bytes()).hexdigest()})
    if not records:
        raise ValueError(f'No NAB CSVs found in {nab / "data"}')
    lists = ROOT / 'data' / 'File_List'
    lists.mkdir(exist_ok=True)
    pd.DataFrame({'file_name': [r['file_name'] for r in records]}).to_csv(lists / 'TSB-AD-U-Eva.csv', index=False)
    save(ROOT / 'data' / 'manifest.json', {
        'dataset': 'NAB, not the paper TSB-AD-U benchmark',
        'scope': 'Entire series; not Sentinel last-15%-test split',
        'timestamp_policy': 'stable_sort_keep_first, as in our existing NAB experiments',
        'missing_policy': 'dropna, as in the authors scripts',
        'labels_sha256': hashlib.sha256((nab / 'labels/combined_windows.json').read_bytes()).hexdigest(),
        'files': records})
    return records


def experiment_plan(suite):
    jobs = []
    baselines = [('last', 1), ('last', 3), ('centered', 3), ('centered', 5)]
    baselines += [('var', w) for w in (64, 96, 256, 512, 1024)]
    for name, window in baselines:
        jobs.append({'kind': 'baseline', 'id': f'{name}_{window}', 'name': name, 'window': window, 'env': 'metrics'})
    jobs.append({'kind': 'baseline_ensemble', 'id': 'var_96_last_3', 'env': 'metrics'})
    if suite == 'baselines':
        return jobs
    for family, script, prefixes in [
        ('moment', 'moment_zs.py', ('moment_base', 'moment_large')),
        ('chronos', 'chronos_zs.py', ('bolt_small', 'bolt_base')),
        ('timesfm', 'times_fm.py', ('timesfm-1.0-200m', 'timesfm-2.0-500m')),
        ('timemoe', 'time-moe.py', ('TimeMoE-50M', 'TimeMoE-200M'))]:
        for size, prefix in zip(('s', 'l'), prefixes):
            for window in ((64, 256, 512, 1024) if family == 'moment' and size == 'l' else (64, 256, 512)):
                jobs.append({'kind': 'model', 'id': f'{prefix}_{window}', 'script': script,
                             'size': size, 'window': window, 'env': family})
    for family, script, prefixes in [
        ('chronos', 'chronos_long_horizon.py', ('bolt_small', 'bolt_base')),
        ('timesfm', 'times_fm_long_horizon.py', ('timesfm-1.0-200m', 'timesfm-2.0-500m'))]:
        for size, prefix in zip(('s', 'l'), prefixes):
            for window in (64, 256, 512):
                jobs.append({'kind': 'long_horizon', 'id': f'{prefix}_long_horizon_{window}',
                             'model_name': prefix, 'script': script, 'size': size,
                             'window': window, 'env': family})
    jobs.append({'kind': 'embedding', 'id': 'moment_base_embedding_64', 'env': 'moment', 'window': 64})
    return jobs


def required_rows(job):
    minimum = job.get('window', 96)
    if job['kind'] == 'long_horizon' or (job['kind'] == 'model' and job['env'] != 'moment'):
        minimum += 1
    return minimum


def execute(command, log, env):
    with log.open('a', encoding='utf8') as stream:
        stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)


def run(args):
    import pandas as pd

    verify_sources()
    plan = experiment_plan(args.suite)
    needed = sorted({'metrics'} | {j['env'] for j in plan})
    if args.setup:
        setup(needed)
        print('Environments installed. No model inference or NAB evaluation performed.')
        return
    if args.prepare or args.check_data:
        if args.nab_dir.exists():
            records = prepare(args.nab_dir)
        elif (ROOT / 'data/manifest.json').exists() and (ROOT / 'data/TSB-AD-U').exists():
            records = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf8'))['files']
        else:
            raise FileNotFoundError(f'NAB directory not found at {args.nab_dir} and pre-converted data not found at {ROOT / "data/TSB-AD-U"}')
        print(f'Prepared {len(records)} series, {sum(r["rows"] for r in records):,} readings. No models run.')
        return
    if args.plan:
        print(json.dumps(plan, indent=2))
        print(f'{len(plan)} configurations; original model batch sizes; required environments: {needed}')
        return
    missing = [f for f in needed if not python_for(f).exists()]
    if missing:
        raise RuntimeError(f'Missing separate environments {missing}. First run: python paper_reproduction/run.py --setup --suite {args.suite}')
    if args.resume:
        folder = args.resume.resolve()
        protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf8'))
        if protocol['suite'] != args.suite:
            raise ValueError('Resume suite differs from the frozen plan')
        if protocol['plan'] != plan:
            raise ValueError('Experiment plan changed since the run was frozen')
        files = pd.read_csv(folder / 'data/File_List/TSB-AD-U-Eva.csv').file_name.tolist()
    else:
        if args.nab_dir.exists():
            records = prepare(args.nab_dir)
        elif (ROOT / 'data/manifest.json').exists() and (ROOT / 'data/TSB-AD-U').exists():
            records = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf8'))['files']
        else:
            raise FileNotFoundError(f'NAB directory not found at {args.nab_dir} and pre-converted data not found at {ROOT / "data/TSB-AD-U"}')
        files = [r['file_name'] for r in records]
        if args.only:
            files = [f for f in files if args.only in f]
        if not files:
            raise ValueError('--only matched no series')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        folder = ROOT / 'runs' / f'paper_nab_{stamp}'
        folder.mkdir(parents=True, exist_ok=False)
        for name in ('scripts', 'tsfms'):
            shutil.copytree(ROOT / 'upstream' / name, folder / name)
        target = folder / 'data/TSB-AD-U'
        target.mkdir(parents=True)
        for file in files:
            shutil.copy2(ROOT / 'data/TSB-AD-U' / file, target / file)
        lists = folder / 'data/File_List'
        lists.mkdir()
        pd.DataFrame({'file_name': files}).to_csv(lists / 'TSB-AD-U-Eva.csv', index=False)
        shutil.copy2(ROOT / 'data/manifest.json', folder / 'data/manifest.json')
        save(folder / 'protocol.json', {'suite': args.suite, 'plan': plan, 'series': files,
             'upstream_files': verify_sources(), 'offline_future_observations_used': True,
             'oracle_label_thresholds_in_TSB_metrics': True,
             'comparable_to_Sentinel_streaming_F1': False,
             'scope': 'main published executable methods on full NAB series; see README for exclusions'})
    logs = folder / 'logs'
    logs.mkdir(exist_ok=True)
    complete = folder / 'completed.json'
    done = json.loads(complete.read_text()) if complete.exists() else []
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'third_party') + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONIOENCODING'] = 'utf-8'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    print(f'Results: {folder}', flush=True)
    try:
        for family in needed:
            # Detect API/dependency problems before starting any real experiment.
            execute([str(python_for(family)), '-c', IMPORT_CHECKS[family]],
                    logs / f'environment_{family}.log', env)
            freeze = subprocess.check_output([str(python_for(family)), '-m', 'pip', 'freeze'], text=True)
            (logs / f'requirements_{family}.txt').write_text(freeze, encoding='utf8')
        for i, job in enumerate(plan, 1):
            print(f'[{i}/{len(plan)}] {job["id"]}', flush=True)
            for file in files:
                key = job['id'] + '/' + file
                if key in done:
                    continue
                started = time.perf_counter()
                log = logs / f'{job["id"]}__{file}.log'
                rows = len(pd.read_csv(folder / 'data/TSB-AD-U' / file).dropna())
                if job['kind'] in ('model', 'long_horizon') and rows >= required_rows(job):
                    execute([str(python_for(job['env'])), str(folder / 'tsfms' / job['script']),
                             job['size'], str(job['window']), file], log, env)
                elif job['kind'] == 'embedding' and rows >= required_rows(job):
                    execute([str(python_for('moment')), str(ROOT / 'embedding.py'), str(folder), file], log, env)
                execute([str(python_for('metrics')), str(ROOT / 'evaluate.py'),
                         str(folder), json.dumps(job), file], log, env)
                elapsed = time.perf_counter() - started
                done.append(key)
                save(complete, done)
                with (folder / 'timing.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps({'job': job['id'], 'file': file, 'seconds': elapsed,
                                             'timing_kind': 'batch offline including metrics and model loading'}) + '\n')
            summarize(folder)
        save(folder / 'status.json', {'status': 'completed', 'completed_jobs': len(done), 'series': len(files)})
    except Exception as error:
        save(folder / 'status.json', {'status': 'failed', 'completed_jobs': len(done),
                                     'error': str(error), 'log_directory': 'logs'})
        print(f'Failed; inspect logs in {logs}. No failed jobs silently skipped.', file=sys.stderr)
        raise
    print(f'Finished. Read {folder / "summary.csv"}', flush=True)


def summarize(folder):
    import pandas as pd
    rows = [json.loads(path.read_text(encoding='utf8')) for path in sorted((folder / 'metrics').glob('*.json'))]
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / 'metrics.csv', index=False)
    if not frame.empty:
        result = frame.groupby('method')[METRICS].mean()
        result['total_series'] = frame.groupby('method').size()
        result['valid_series'] = frame.groupby('method')['VUS-PR'].count()
        result.to_csv(folder / 'summary.csv')


def main():
    configure_storage()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['baselines', 'full'], default='full')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--setup', action='store_true', help='Install isolated dependencies only; no model inference')
    action.add_argument('--prepare', action='store_true', help='Convert existing NAB files only')
    action.add_argument('--check-data', action='store_true', help='Verify/prepare data without inference')
    action.add_argument('--plan', action='store_true', help='Print frozen model configurations; no inference')
    default_nab = (ROOT / 'data/nab') if (ROOT / 'data/nab').exists() else (ROOT.parent / 'data/nab')
    parser.add_argument('--nab-dir', type=Path, default=default_nab)
    parser.add_argument('--only', help='Optional filename substring; otherwise all 58 streams')
    parser.add_argument('--resume', type=Path, help='Resume an existing paper run folder')
    args = parser.parse_args()
    if args.resume and args.only:
        parser.error('--only cannot change a resumed run')
    run(args)


if __name__ == '__main__':
    main()
