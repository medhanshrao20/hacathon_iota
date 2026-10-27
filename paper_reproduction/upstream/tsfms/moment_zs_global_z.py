# NOTE: Z-normalization is an integral part of the MOMENT pipeline and cannot be disabled externally.
# In our experiments, we manually modified the code base of MOMENT to disable its Z-normalization and the corresponding de-normalization.

import warnings
# Ignore the specific PyTree deprecation warning
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*", category=FutureWarning)

import numpy as np
import pandas as pd
import os
import sys
import math
import torch
import argparse
from tqdm import tqdm
from momentfm import MOMENTPipeline
from helper import create_time_series_batches

# Determine the available device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments: model_size, window_size, file_name
parser = argparse.ArgumentParser(
    description='Process time series data with MOMENT (with global-Z normalization), save anomaly scores as intermediate results.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
model_size : 's' (for 'base') or 'l' (for 'large')
window_size: positive integer
file_name  : name of the CSV file to process
""")
parser.add_argument('model_size', type=str, choices=['s', 'l'], help="Model size: 's' (for 'base') or 'l' (for 'large')")
parser.add_argument('window_size', type=int, help='Positive integer for window size')
parser.add_argument('file_name', type=str, help='Name of the CSV file to process')
args = parser.parse_args()

# Validate window_size is positive
if args.window_size <= 0:
    parser.error("window_size must be a positive integer.")

model_size = args.model_size.lower()  # model_size can be either 's' (for 'base') or 'l' (for 'large').
model_size = "base" if model_size == "s" else "large"

window_size = args.window_size
file_name = args.file_name
batch_size = 128
min_std = 1e-9

print(f'processing {file_name} on {device}!')

# Load data and create batches for reconstruction.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_direc = os.path.join(project_root, 'data', 'TSB-AD-U', file_name)
df = pd.read_csv(data_direc).dropna()
data = df.iloc[:, 0:-1].values.astype(float).squeeze()
mean, std = data.mean(), data.std()
if std < min_std:
    std = 1

X_batches, _ = create_time_series_batches(data, window_size=window_size, forecast_horizon=0, batch_size=batch_size)
file_name_without_ext = os.path.splitext(file_name)[0]

# Load the pre-trained MOMENT model for reconstruction.
model = MOMENTPipeline.from_pretrained(
            f"AutonLab/MOMENT-1-{model_size}", 
            model_kwargs={"task_name": "reconstruction"}
        )

model.init()
# Move the model to the selected device
model = model.float().to(device)
anomaly_criterion = torch.nn.MSELoss(reduction='none')

scores = np.array([])

with torch.no_grad():
    
    for i, X in tqdm(enumerate(X_batches), total=len(X_batches)):
    
        # Send inputs to the correct device
        X = torch.tensor(X, device=device).float()
        masks = torch.full_like(X, 1.0, dtype=torch.float64) # automatically uses X's device

        x_enc = ((X - mean) / std).unsqueeze(1)
        output = model(x_enc=x_enc, input_mask=masks)
        output = output.reconstruction.squeeze()
        output = output * std + mean

        score = torch.mean(anomaly_criterion(X, output), dim=-1).detach().cpu().numpy().ravel()
        scores = np.concatenate([scores, score])

# Create directory to save anomaly scores.
path = os.path.join(project_root, 'reproductions', 'intermediate', f'moment_{model_size}_globalZ_{window_size}', 'scores')
os.makedirs(path, exist_ok=True)

# Calculate and save anomaly scores. We pad the scores to match the original time series length, using the first and last score for padding.
scores = np.pad(scores, (math.ceil((window_size-1)/2), (window_size-1)//2), mode='constant', constant_values=(scores[0], scores[-1]))
np.save(os.path.join(path, file_name_without_ext), scores)

print(f'finished {file_name}!')