import os
import pandas as pd
import numpy as np
import warnings
import argparse
from sklearn.exceptions import UndefinedMetricWarning
from TSB_AD.evaluation.metrics import get_metrics

# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# Define one-liner baselines

def moving_window_variance(data, win_size):
    return data.rolling(window = win_size, center = True).var().bfill().ffill().to_numpy()

def squared_difference(data, win_size, centered):
    # The operation '* win_size / (win_size - 1.0)' does not change the relative order of the anomaly scores, so it does not affect the VUS-PR metric. It is included to align with the formal definition for this baseline in the paper.
    return ((data.to_numpy() - data.rolling(window = win_size, center = centered).mean().bfill().ffill().to_numpy()) * win_size / (win_size - 1.0)) ** 2

def get_scores(data, baseline_name, win_size):
    if baseline_name == 'var':
        scores = moving_window_variance(data, win_size)
    elif baseline_name == 'last':
        # The '+1' in 'win_size+1' is due to the implementation detail of squared_difference, which requires a window size of 'win_size+1' to achieve the intended behavior for the 'last' baseline.
        scores = squared_difference(data, win_size+1, centered=False)
    elif baseline_name == 'centered':
        scores = squared_difference(data, win_size, centered=True)
    return scores

# Normalization for ensembling
def min_max(array):
    return (array - array.min()) / (array.max() - array.min())

# Parse command-line arguments: baseline_name, win_size
parser = argparse.ArgumentParser(
    description='Evaluate one-liner baselines on TSB-AD-U.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
baseline_name_1: 'var', 'last', or 'centered'
win_size_1     : positive integer (context window size, e.g., 64)
baseline_name_2: 'var', 'last', or 'centered'
win_size_2     : positive integer (context window size, e.g., 64)
-t / --tuning : (optional flag) whether to use the tuning set (instead of the evaluation set)
""")
parser.add_argument('baseline_name_1', type=str, choices=['centered', 'last', 'var'], help="Baseline name: 'var', 'last', or 'centered'")
parser.add_argument('win_size_1', type=int, help='Positive integer for context window size')
parser.add_argument('baseline_name_2', type=str, choices=['centered', 'last', 'var'], help="Baseline name: 'var', 'last', or 'centered'")
parser.add_argument('win_size_2', type=int, help='Positive integer for context window size')
parser.add_argument("-t" ,"--tuning", action='store_true', help="Whether to use the tuning set (instead of the evaluation set)")
args = parser.parse_args()

# Validate win_size is positive
if args.win_size_1 <= 0:
    parser.error("win_size_1 must be a positive integer.")
if args.win_size_2 <= 0:
    parser.error("win_size_2 must be a positive integer.")

baseline_name_1 = args.baseline_name_1
win_size_1 = args.win_size_1
baseline_name_2 = args.baseline_name_2
win_size_2 = args.win_size_2

# Read the file list and initialize the results DataFrame

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if args.tuning:
    file_list_path = os.path.join(project_root, "data", "File_List", "TSB-AD-U-Tuning.csv")
else:
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

for index, row in results.iterrows():

    file_name = row['file_name']
    print(f"Processing {file_name}...")

    df = pd.read_csv(os.path.join(data_dir, file_name)).dropna()
    label = df['Label'].astype(int).to_numpy()
    
    if len(label) < win_size_1 or len(label) < win_size_2:
        print(f"Skipped, shorter than window size: {file_name}")
        continue
    
    scores_1 = get_scores(df['Data'], baseline_name_1, win_size_1)
    scores_2 = get_scores(df['Data'], baseline_name_2, win_size_2)
    scores_1 = min_max(scores_1)
    scores_2 = min_max(scores_2)
    scores = np.maximum(scores_1, scores_2)

    evaluation_result = get_metrics(scores, label)
    print(evaluation_result)
        
    for metric in metrics:
        results.at[index, metric] = evaluation_result[metric]

# Save the evaluation results to a CSV file.
if args.tuning:
    results.to_csv(os.path.join(project_root, "reproductions", "baselines", f'{baseline_name_1}_{win_size_1}_{baseline_name_2}_{win_size_2}_tuning.csv'), index=False)
else:
    results.to_csv(os.path.join(project_root, "reproductions", "baselines", f'{baseline_name_1}_{win_size_1}_{baseline_name_2}_{win_size_2}.csv'), index=False)