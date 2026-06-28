"""
live_run.py
Fetches the latest stock data and predicts the next day's closing price
using the saved model and scaler.
"""

import numpy as np
import joblib
from data_fetch import fetch_recent_data
from model import load_trained_model


def predict_next_close(ticker="AAPL", window_size=20):
    """
    Predict the next trading day's closing price.

    Steps:
        1. Load saved model and scaler.
        2. Fetch recent stock data (last 1 month).
        3. Scale the recent close prices using the saved scaler.
        4. Take the last `window_size` days as input.
        5. Feed into the model and get prediction.
        6. Inverse-transform to get the actual predicted price.
    """
    # Load saved model and scaler
    model = load_trained_model("saved_models/stock_ann_model.keras")
    scaler = joblib.load("saved_models/scaler.pkl")
    print("Scaler loaded successfully.")

    # Fetch recent data
    recent_df = fetch_recent_data(ticker=ticker, period="1mo")

    if len(recent_df) < window_size:
        raise ValueError(
            f"Not enough recent data. Need {window_size} days, got {len(recent_df)}."
        )

    # Get the last `window_size` close prices
    recent_close = recent_df["Close"].values[-window_size:]

    # Scale using the saved scaler
    recent_scaled = scaler.transform(recent_close.reshape(-1, 1)).flatten()

    # Reshape for model input: (1, window_size)
    model_input = recent_scaled.reshape(1, -1)

    # Predict (scaled)
    pred_scaled = model.predict(model_input, verbose=0)

    # Inverse transform to get actual price
    predicted_price = scaler.inverse_transform(pred_scaled).flatten()[0]

    # Display results
    last_close = recent_df["Close"].values[-1]
    last_date = recent_df.index[-1].strftime("%Y-%m-%d")

    print("\n========= LIVE PREDICTION =========")
    print(f"  Ticker:             {ticker}")
    print(f"  Last Close Date:    {last_date}")
    print(f"  Last Close Price:   ${last_close:.2f}")
    print(f"  Predicted Next Close: ${predicted_price:.2f}")
    print("====================================\n")

    return predicted_price


if __name__ == "__main__":
    predict_next_close()
