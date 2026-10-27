import timesfm
import numpy as np
import pandas as pd
import os
import sys
import torch
import argparse
from tqdm import tqdm
from helper import create_time_series_batches

# Determine the available device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments: model_size, window_size, file_name
parser = argparse.ArgumentParser(
    description='Process time series data with TimesFM (with global-Z normalization), save predictions and anomaly scores as intermediate results.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""model_size : 's' (for '1.0-200m') or 'l' (for '2.0-500m')
window_size: positive integer
file_name  : Name of the CSV file to process
""")
parser.add_argument('model_size', type=str, choices=['s', 'l'], help="Model size: 's' (for '1.0-200m') or 'l' (for '2.0-500m')")
parser.add_argument('window_size', type=int, help='Positive integer for window size')
parser.add_argument('file_name', type=str, help='Name of the CSV file to process')
args = parser.parse_args()

# Validate window_size is positive
if args.window_size <= 0:
    parser.error("window_size must be a positive integer.")

model_size = args.model_size.lower()  # model_size can be either 's' (for '1.0-200m') or 'l' (for '2.0-500m').
model_size = "1.0-200m" if model_size == "s" else "2.0-500m"
window_size = args.window_size
file_name = args.file_name

batch_size = 256
prediction_length = 1
min_std = 1e-9

print(f'processing {file_name} on {device}!')

# Load data.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_direc = os.path.join(project_root, 'data', 'TSB-AD-U', file_name)
df = pd.read_csv(data_direc).dropna()
data = df.iloc[:, 0:-1].values.astype(float).squeeze()

# Calculate global mean and std for normalization.
mean, std = data.mean(), data.std()
if std < min_std:
    std = 1

# Create batches for forecasting.
X_batches, _ = create_time_series_batches(data, window_size=window_size, forecast_horizon=prediction_length, batch_size=batch_size)

# Load the pre-trained TimeSFM model for forecasting.
if model_size == "1.0-200m":

    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu" if torch.cuda.is_available() else "cpu",
            per_core_batch_size=32,
            horizon_len=prediction_length,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id=f"google/timesfm-1.0-200m-pytorch"),
    )
    
else:

    tfm = timesfm.TimesFm(
          hparams=timesfm.TimesFmHparams(
              backend="gpu" if torch.cuda.is_available() else "cpu",
              per_core_batch_size=32,
              horizon_len=prediction_length,
              num_layers=50,
              use_positional_embedding=False,
              context_len=window_size,
          ),
          checkpoint=timesfm.TimesFmCheckpoint(
              huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )

predictions = np.array([])

for X in tqdm(X_batches, total=len(X_batches)):
    # Apply global-Z normalization to the input before forecasting, and set normalize=False to prevent the model from applying its own normalization.
    output, _ = tfm.forecast((X - mean) / std, normalize=False)
    predictions = np.concatenate([predictions, output.ravel() * std + mean]) #denormalize the predictions

file_name = os.path.splitext(file_name)[0]
path = os.path.join(project_root, 'reproductions', 'intermediate', f'timesfm-{model_size}_globalZ_{window_size}', 'predictions')
os.makedirs(path, exist_ok=True)
np.save(os.path.join(path, file_name), predictions) #save predictions

path = os.path.join(project_root, 'reproductions', 'intermediate', f'timesfm-{model_size}_globalZ_{window_size}', 'scores')
os.makedirs(path, exist_ok=True)
score = (predictions - data[window_size:]) ** 2
score = np.pad(score, (window_size, 0), mode='constant', constant_values=score[0])
np.save(os.path.join(path, file_name), score) #save anomaly scores

print(f'finished {file_name}!')