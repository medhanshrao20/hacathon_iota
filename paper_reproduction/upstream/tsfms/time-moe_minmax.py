import numpy as np
import pandas as pd
import os
import sys
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from helper import create_time_series_batches

# Determine the available device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments: model_size, window_size, file_name
parser = argparse.ArgumentParser(
    description='Process time series data with Time-MoE (with Min-Max normalization), save predictions and anomaly scores as intermediate results.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
model_size : 's' (for 'base (50M)') or 'l' (for 'large (200M)')
window_size: positive integer
file_name  : name of the CSV file to process
""")
parser.add_argument('model_size', type=str, choices=['s', 'l'], help="Model size: 's' (for 'base (50M)') or 'l' (for 'large (200M)')")
parser.add_argument('window_size', type=int, help='Positive integer for window size')
parser.add_argument('file_name', type=str, help='Name of the CSV file to process')
args = parser.parse_args()

# Validate window_size is positive
if args.window_size <= 0:
    parser.error("window_size must be a positive integer.")

model_size = args.model_size.lower()
model_path = "TimeMoE-50M" if model_size == "s" else "TimeMoE-200M"
window_size = args.window_size
file_name = args.file_name

batch_size = 256
min_range = 1e-9
prediction_length = 1

print(f'processing {file_name} on {device}!')

# Load data and create batches for forecasting.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_direc = os.path.join(project_root, 'data', 'TSB-AD-U', file_name)
df = pd.read_csv(data_direc).dropna()
data = df.iloc[:, 0:-1].values.astype(float).squeeze()
X_batches, _ = create_time_series_batches(data, window_size=window_size, forecast_horizon=prediction_length, batch_size=batch_size)

# Load the pre-trained Time-MoE model for forecasting.
model = AutoModelForCausalLM.from_pretrained(
    f'Maple728/{model_path}',
    device_map=str(device),  # use "cpu" for CPU inference, and "cuda" for GPU inference.
    trust_remote_code=True
)

predictions = np.array([])

# Generate predictions in batches.

with torch.no_grad():
    
    for X in tqdm(X_batches, total=len(X_batches)):
    
        X = torch.tensor(X, dtype=torch.float32).to(device)
        
        max_val, min_val = X.max(dim=-1, keepdim=True).values, X.min(dim=-1, keepdim=True).values
        val_range = max_val - min_val
        val_range_for_division = torch.where(val_range < min_range, 1, val_range)
        
        X = (X - min_val) / val_range_for_division
    
        output = model.generate(X, max_new_tokens=prediction_length)[:, -prediction_length:]
        output = output * val_range_for_division + min_val
    
        predictions = np.concatenate([predictions, output.to("cpu").numpy().ravel()])


# Save predictions and anomaly scores.

file_name = os.path.splitext(file_name)[0]
path = os.path.join(project_root, 'reproductions', 'intermediate', f'{model_path}_minmax_{window_size}', 'predictions')
os.makedirs(path, exist_ok=True)
np.save(os.path.join(path, f'{file_name}'), predictions) #save predictions

path = os.path.join(project_root, 'reproductions', 'intermediate', f'{model_path}_minmax_{window_size}', 'scores')
os.makedirs(path, exist_ok=True)
score = (predictions - data[window_size:]) ** 2
score = np.pad(score, (window_size, 0), mode='constant', constant_values=score[0])
np.save(os.path.join(path, f'{file_name}'), score) #save anomaly scores

print(f'finished {file_name}!')