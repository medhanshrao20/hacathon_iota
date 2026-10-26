# import warnings
# warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Convert long-horizon predictions to anomaly scores.')
parser.add_argument('model_name', type=str, choices=['bolt_small', 'bolt_base', 'timesfm-1.0-200m', 'timesfm-2.0-500m'], help='Name of the model')
parser.add_argument('win_size', type=int, help='Context window size used for long-horizon experiments')
args = parser.parse_args()

model_name = args.model_name
win_size = args.win_size

horizons = 2**np.arange(10) # horizons considered for evaluation: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_list_path = os.path.join(project_root, "data", "File_List", "TSB-AD-U-Eva.csv")
data_dir = os.path.join(project_root, "data", "TSB-AD-U")

files_df = pd.read_csv(file_list_path)
files = files_df["file_name"].dropna().tolist()

for file in files:

    print(f"Processing file: {file}")

    name = os.path.splitext(file)[0]

    path_results = os.path.join(project_root, "reproductions", "intermediate", f"{model_name}_long_horizon_{win_size}", "predictions", name)
    path_scores = os.path.join(project_root, "reproductions", "intermediate", f"{model_name}_long_horizon_{win_size}", "scores", name)
    os.makedirs(path_scores, exist_ok=True)

    # Read time series data and labels

    df = pd.read_csv(os.path.join(data_dir, file)).dropna()
    data = df.iloc[:, 0:-1].values.astype(float).squeeze()
    label = df['Label'].astype(int).to_numpy()

    # Read long-horizon predictions and combine them into a single 2D array

    file_list = list(Path(path_results).glob('*.npy'))
    file_list.sort(key=lambda p: int(p.stem))

    all_arrays = []

    for file_path in file_list:
        arr = np.load(file_path)
        all_arrays.append(arr)

    combined_array = np.concatenate(all_arrays, axis=0)

    # Compute anomaly scores for each horizon and save them as .npy files

    for horizon in horizons:

        # Skip the given horizon if the time series is too short for it
    
        length = len(data) - win_size - horizon + 1
        
        if length <= 0:
            break

        # Compute anomaly scores using mean squared error between predictions and ground truth

        scores = np.zeros(len(data) - win_size - horizon + 1)

        for i in range(len(scores)):
            
            ground_truth = data[win_size + i : win_size + i + horizon]
            prediction = combined_array[i, :horizon]

            score = ((prediction - ground_truth) ** 2).mean()
            scores[i] = score

        # Pad the scores to match the original time series length and save them as .npy files

        scores = np.pad(scores, (win_size, horizon-1), mode='constant', constant_values=(scores[0], scores[-1]))
        np.save(os.path.join(path_scores, f'{horizon}.npy'), scores)
        