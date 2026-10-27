import numpy as np

def create_time_series_batches(data, window_size, forecast_horizon, batch_size):

    X, y = [], []
    total_samples = len(data) - window_size - forecast_horizon + 1
    
    for i in range(total_samples):
        # Input window
        X.append(data[i:i+window_size])
        
        # Target values (future values to predict)
        y.append(data[i+window_size:i+window_size+forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    # Create batches
    X_batches = []
    y_batches = []
    
    # Calculate how many complete batches we can make
    num_complete_batches = len(X) // batch_size
    
    # First add all complete batches
    for i in range(num_complete_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        
        X_batches.append(X[batch_start:batch_end])
        y_batches.append(y[batch_start:batch_end])
    
    # Handle the remainder
    remainder_start = num_complete_batches * batch_size
    if remainder_start < len(X):
        X_batches.append(X[remainder_start:])
        y_batches.append(y[remainder_start:])
    
    return X_batches, y_batches
