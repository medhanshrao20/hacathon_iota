import os
import sys
import pandas as pd
import numpy as np
import argparse
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from TSB_AD.evaluation.metrics import get_metrics

# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# Usage:
#   python scripts/evaluate_anomaly_scores.py moment_base_64
#   python scripts/evaluate_anomaly_scores.py bolt_small_512
    
parser = argparse.ArgumentParser(description="Evaluate anomaly scores.")
parser.add_argument("folder_name", help="Name of the folder containing the scores (e.g., moment_base_64)")
args = parser.parse_args()

# Read the file list and initialize the results DataFrame
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_list_path = os.path.join(project_root, "data", "File_List", "TSB-AD-U-Eva.csv")
data_dir = os.path.join(project_root, "data", "TSB-AD-U")

files_df = pd.read_csv(file_list_path)
files = files_df["file_name"].dropna().tolist()
results = pd.DataFrame({'file_name': files})

metrics = ('AUC-PR',
 'AUC-ROC',
 'VUS-PR',
 'VUS-ROC',
 'Standard-F1',
 'PA-F1',
 'Event-based-F1',
 'R-based-F1',
 'Affiliation-F')

for metric in metrics:
    results[metric] = None

# Evaluate anomaly scores for each time series.
path = os.path.join(project_root, "reproductions", "intermediate", args.folder_name, "scores")

# Terminate this script if the specified path doesn't exist.
if not os.path.exists(path):
    print(f"{path} doesn't exist!!")
    sys.exit(1)

for index, row in results.iterrows():

    file_name = row['file_name']
    print(f"Processing {file_name}...")

    df = pd.read_csv(os.path.join(data_dir, file_name)).dropna()
    label = df['Label'].astype(int).to_numpy()
    
    npy_file = os.path.splitext(file_name)[0] + '.npy'
    npy_file = os.path.join(path, npy_file)
        
    if not os.path.exists(npy_file):
        print(f"{npy_file} doesn't exist!!")
        continue
    
    scores = np.load(npy_file)
    evaluation_result = get_metrics(scores, label)
    print(evaluation_result)
        
    for metric in metrics:
        results.at[index, metric] = evaluation_result[metric]

# Save the evaluation results to a CSV file.
results.to_csv(os.path.join(project_root, "reproductions", "tsfms", f'{args.folder_name}.csv'), index=False)