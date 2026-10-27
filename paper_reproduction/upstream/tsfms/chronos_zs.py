import numpy as np
import pandas as pd
import os
import sys
import torch
import shutil
import time
import tempfile
import argparse
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from helper import create_time_series_batches

# Determine the available device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments: model_size, window_size, file_name
parser = argparse.ArgumentParser(
    description='Process time series data with Chronos, save predictions and anomaly scores as intermediate results.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
model_size : 's' (for 'small') or 'l' (for 'base')
window_size: positive integer
file_name  : name of the CSV file to process
""")
parser.add_argument('model_size', type=str, choices=['s', 'l'], help="Model size: 's' (for 'small') or 'l' (for 'base')")
parser.add_argument('window_size', type=int, help='Positive integer for window size')
parser.add_argument('file_name', type=str, help='Name of the CSV file to process')
args = parser.parse_args()

# Validate window_size is positive
if args.window_size <= 0:
    parser.error("window_size must be a positive integer.")

model_size = args.model_size.lower()  # model_size can be either 's' (for 'small') or 'l' (for 'base').
model_path = "bolt_small" if model_size == "s" else "bolt_base"
window_size = args.window_size
file_name = args.file_name

n_batch_per_inference = 120
batch_size = 128
forecast_horizon = 1

print(f'processing {file_name} on {device}!')

# Load data and create batches for forecasting.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_direc = os.path.join(project_root, 'data', 'TSB-AD-U', file_name)
df = pd.read_csv(data_direc).dropna()
data = df.iloc[:, 0:-1].values.astype(float).squeeze()

batch_size_tmp = len(data) - window_size - forecast_horizon + 1
X_batches, _ = create_time_series_batches(data, window_size=window_size, forecast_horizon=forecast_horizon, batch_size=batch_size_tmp)

inference_size = batch_size * n_batch_per_inference
n_inference = X_batches[0].shape[0] // inference_size + 1

results = pd.DataFrame()
# print(n_inference)

temp_dir = tempfile.mkdtemp()

try:

    # Run inference in batches.
    for i in range(n_inference):

        X = X_batches[0][inference_size*i:inference_size*(i+1), :]

        if len(X) == 0:
            break

        input_data = pd.DataFrame(X.ravel(), columns=['target'])
        input_data['item_id'] = input_data.index // window_size
        input_data['timestamp'] = input_data.index % window_size
        input_data = input_data.reindex(columns=['item_id', 'timestamp', 'target'])
        
        predictor = TimeSeriesPredictor(prediction_length=forecast_horizon, path=temp_dir).fit(
            input_data,
            hyperparameters={
                "Chronos": {
                    "model_path": model_path,
                    "device": str(device),
                    "batch_size": batch_size}},
            skip_model_selection=True,
            enable_ensemble=False,
            verbosity=0)
        
        predictions = predictor.predict(input_data)
        results = pd.concat([results, predictions], axis=0, ignore_index=True)

        del predictor

finally:
    try:
        # Brief pause to allow file handles to close (especially needed on Windows)
        time.sleep(0.5)
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

# Save predictions and anomaly scores.
file_name = os.path.splitext(file_name)[0]

path = os.path.join(project_root, 'reproductions', 'intermediate', f'{model_path}_{window_size}', 'predictions')
os.makedirs(path, exist_ok=True)
results.to_pickle(os.path.join(path, f'{file_name}.pkl')) #save predictions

path = os.path.join(project_root, 'reproductions', 'intermediate', f'{model_path}_{window_size}', 'scores')
os.makedirs(path, exist_ok=True)
score = (results['mean'].to_numpy() - data[window_size:]) ** 2
score = np.pad(score, (window_size, 0), mode='constant', constant_values=score[0])
np.save(os.path.join(path, f'{file_name}'), score) #save anomaly scores

print(f'finished {file_name}!')
