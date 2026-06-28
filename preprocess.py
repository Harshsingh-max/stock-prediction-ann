"""
preprocess.py
Handles data preprocessing: scaling, sliding window creation, and chronological splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def scale_data(df, feature_col="Close", scaler=None, save_path="saved_models"):
    """
    Apply MinMaxScaler to the target column.
    If scaler is None, fit a new one. Otherwise, use the provided scaler.

    Returns:
        scaled_values: numpy array of scaled values (shape: n, 1)
        scaler: the fitted MinMaxScaler object
    """
    values = df[feature_col].values.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values)

        # Save scaler for later use in live predictions
        os.makedirs(save_path, exist_ok=True)
        scaler_path = os.path.join(save_path, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    else:
        scaled_values = scaler.transform(values)

    return scaled_values, scaler


def create_sliding_windows(scaled_data, window_size=20):
    """
    Create input-output pairs using a sliding window approach.

    For each window of `window_size` consecutive days, the next day's
    close price is the target.

    Example with window_size=3:
        Input: [day1, day2, day3] -> Output: day4
        Input: [day2, day3, day4] -> Output: day5

    Returns:
        X: input array of shape (num_samples, window_size)
        y: target array of shape (num_samples,)
    """
    X, y = [], []

    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i : i + window_size, 0])
        y.append(scaled_data[i + window_size, 0])

    return np.array(X), np.array(y)


def split_data(X, y, train_ratio=0.8):
    """
    Split data chronologically (NOT randomly) to preserve time order.

    Returns:
        X_train, X_test, y_train, y_test
    """
    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(df, window_size=20, train_ratio=0.8):
    """
    Full preprocessing pipeline: scale -> window -> split.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    scaled_data, scaler = scale_data(df)
    X, y = create_sliding_windows(scaled_data, window_size)
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Quick test
    df = pd.read_csv("data/AAPL_historical.csv", index_col="Date", parse_dates=True)
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(df)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
