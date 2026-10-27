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
    description='Process time series data with Chronos (forecast_horizon = 512), save predictions as intermediate results.',
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
forecast_horizon = 512

print(f'processing {file_name} on {device}!')

# Create folder for saving results.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(project_root, 'reproductions', 'intermediate', f'{model_path}_long_horizon_{window_size}', 'predictions', os.path.splitext(file_name)[0])
os.makedirs(path, exist_ok=True)

# Load data and create batches for forecasting.
data_direc = os.path.join(project_root, 'data', 'TSB-AD-U', file_name)
df = pd.read_csv(data_direc).dropna()
data = df.iloc[:, 0:-1].values.astype(float).squeeze()

#  The forecast_horizon for batch creation should be kept to be 1.
batch_size_tmp = len(data) - window_size
X_batches, _ = create_time_series_batches(data, window_size=window_size, forecast_horizon=1, batch_size=batch_size_tmp)

inference_size = batch_size * n_batch_per_inference
n_inference = X_batches[0].shape[0] // inference_size + 1

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

        # Save predictions for the current batch.
        np.save(os.path.join(path, f'{i}.npy'), predictions['mean'].values.reshape(-1, 512))

        del predictor

finally:
    try:
        # Brief pause to allow file handles to close (especially needed on Windows)
        time.sleep(0.5)
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

print(f'finished {file_name}!')
