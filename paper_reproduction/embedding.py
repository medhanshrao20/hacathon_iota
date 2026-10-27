"""Apply the authors' Figure 8b notebook to a NAB series (offline/transductive)."""
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from momentfm import MOMENTPipeline
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

ROOT = Path(__file__).resolve().parent


def main(folder, file):
    data = pd.read_csv(folder / 'data/TSB-AD-U' / file).dropna().Data.to_numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    moment = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-base', model_kwargs={'task_name': 'embedding'})
    moment.init()
    moment = moment.to(device).float()
    notebook = ROOT / 'upstream/notebooks/embedding_based_TSAD/analysis.ipynb'
    cells = json.loads(notebook.read_text(encoding='utf8'))['cells']
    functions = []
    for cell in cells:
        if cell['cell_type'] == 'code':
            functions.extend(n for n in ast.parse(''.join(cell['source'])).body
                             if isinstance(n, ast.FunctionDef))
    scope = {'np': np, 'torch': torch, 'moment': moment, 'device': device,
             'X_test': data, 'EMBEDDING_DIMENSION_MOMENT': 768}
    # The original centering/padding (including its last-window overwrite) is retained.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(notebook), 'exec'), scope)
    embeddings = scope['compute_embedding'](data, 64)
    destination = folder / 'reproductions/intermediate/moment_base_embedding_64'
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / f'{Path(file).stem}.npy', embeddings)
    detectors = [('ECOD', ECOD()), ('IForest', IForest(random_state=0)),
                 ('LOF', LOF()), ('COPOD', COPOD()), ('KNN_default', KNN())]
    detectors += [(f'KNN_{k}', KNN(n_neighbors=k)) for k in (1, 3, 5, 10, 20, 50)]
    for name, detector in detectors:
        detector.fit(embeddings)
        scores = detector.decision_function(embeddings)
        output = folder / 'reproductions/intermediate' / f'moment_embedding_64_{name}' / 'scores'
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / f'{Path(file).stem}.npy', scores)


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2])
