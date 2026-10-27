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
    description='Process time series data with TimesFM (forecast_horizon = 512), save predictions as intermediate results.',
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
prediction_length = 512

print(f'processing {file_name} on {device}!')

# Load data and create batches for forecasting.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_direc = os.path.join(project_root, 'data', 'TSB-AD-U', file_name)
df = pd.read_csv(data_direc).dropna()
data = df.iloc[:, 0:-1].values.astype(float).squeeze()
#  The forecast_horizon for batch creation should be kept to be 1.
X_batches, _ = create_time_series_batches(data, window_size=window_size, forecast_horizon=1, batch_size=batch_size)

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

# Create folder for saving results.
file_name = os.path.splitext(file_name)[0]
path = os.path.join(project_root, 'reproductions', 'intermediate', f'timesfm-{model_size}_long_horizon_{window_size}', 'predictions', file_name)
os.makedirs(path, exist_ok=True)

for i, X in tqdm(enumerate(X_batches), total=len(X_batches)):
    # Note: In TimesFM's default setting for inference, normalization is turned off (see their source code in timesfm_base.py).
    # You can turn it on with `tfm.forecast(X, normalize=True)`.
    # In our experiments, the changes brought about by turning on/off normalization were negligible.
    output, _ = tfm.forecast(X)
    np.save(os.path.join(path, f'{i}.npy'), output) #save predictions for the current batch

print(f'finished {file_name}!')