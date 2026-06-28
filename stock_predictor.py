"""
=============================================================================
  STOCK MARKET PRICE PREDICTION USING ARTIFICIAL NEURAL NETWORK (ANN)
  Single-file version — run this one file to train, evaluate, and predict.
=============================================================================

Usage:
    python stock_predictor.py              # Train model + evaluate + plot
    python stock_predictor.py --live       # Predict next day's close price

Requirements:
    pip install yfinance pandas numpy scikit-learn matplotlib tensorflow joblib
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import yfinance as yf
from tensorflow import keras
from tensorflow.keras import layers

# ── Configuration ───────────────────────────────────────────────────────────
TICKER = "AAPL"
PERIOD = "5y"
WINDOW_SIZE = 20
TRAIN_RATIO = 0.8
EPOCHS = 50
BATCH_SIZE = 32

DATA_DIR = "data"
MODEL_DIR = "saved_models"
PLOT_DIR = "plots"
MODEL_PATH = os.path.join(MODEL_DIR, "stock_ann_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_stock_data(ticker=TICKER, period=PERIOD):
    """Download historical stock data and save to CSV."""
    print(f"Downloading {ticker} data for the last {period}...")
    df = yf.Ticker(ticker).history(period=period)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"

    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{ticker}_historical.csv")
    df.to_csv(filepath)
    print(f"Saved {len(df)} rows to {filepath}")
    return df


def fetch_recent_data(ticker=TICKER):
    """Fetch last 1 month of data for live prediction."""
    print(f"Fetching recent {ticker} data...")
    df = yf.Ticker(ticker).history(period="1mo")
    if df.empty:
        raise ValueError(f"No recent data found for '{ticker}'.")
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ═══════════════════════════════════════════════════════════════════════════
#  2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def scale_data(df, scaler=None):
    """
    Scale the 'Close' column to 0-1 range using MinMaxScaler.
    If scaler is None, fit a new one and save it.
    """
    values = df["Close"].values.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler saved to {SCALER_PATH}")
    else:
        scaled = scaler.transform(values)

    return scaled, scaler


def create_sliding_windows(scaled_data, window_size=WINDOW_SIZE):
    """
    Sliding window: use `window_size` consecutive days as input,
    the next day's price as output.
    Example (window=3): [day1,day2,day3] -> day4
    """
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i : i + window_size, 0])
        y.append(scaled_data[i + window_size, 0])
    return np.array(X), np.array(y)


def split_data(X, y, ratio=TRAIN_RATIO):
    """Chronological split (no random shuffling) to prevent data leakage."""
    idx = int(len(X) * ratio)
    print(f"Training samples: {idx}, Testing samples: {len(X) - idx}")
    return X[:idx], X[idx:], y[:idx], y[idx:]


# ═══════════════════════════════════════════════════════════════════════════
#  3. ANN MODEL
# ═══════════════════════════════════════════════════════════════════════════

def build_ann(input_dim=WINDOW_SIZE):
    """
    Feedforward ANN (MLP) for regression:
        Input(20) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)  # linear output for regression
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    model.summary()
    return model


def train_model(model, X_train, y_train):
    """Train the model, save it, and return the training history."""
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    return history


# ═══════════════════════════════════════════════════════════════════════════
#  4. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, scaler):
    """
    Predict on test data, inverse-transform to real prices,
    and compute RMSE, MAE, MAPE.
    """
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_predicted = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mae = mean_absolute_error(y_actual, y_predicted)
    mask = y_actual != 0
    mape = np.mean(np.abs((y_actual[mask] - y_predicted[mask]) / y_actual[mask])) * 100

    print("\n===== Evaluation Metrics =====")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  MAPE  : {mape:.4f}%")
    print("==============================\n")

    return y_actual, y_predicted


# ═══════════════════════════════════════════════════════════════════════════
#  5. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_loss(history):
    """Plot training vs validation loss."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Model Training Loss", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_loss.png"), dpi=150)
    plt.show()
    print("Training loss plot saved.")


def plot_predictions(y_actual, y_predicted):
    """Plot actual vs predicted stock prices."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(y_actual, label="Actual Price", color="#2196F3", linewidth=1.5)
    plt.plot(y_predicted, label="Predicted Price", color="#FF5722",
             linewidth=1.5, linestyle="--")
    plt.title("Stock Price: Actual vs Predicted", fontsize=14)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Price (USD)")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.show()
    print("Prediction plot saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  6. LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_next_close(ticker=TICKER):
    """
    Load saved model & scaler, fetch latest prices,
    and predict the next trading day's closing price.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("ERROR: No saved model found. Run training first:")
        print("       python stock_predictor.py")
        return

    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded.")

    recent_df = fetch_recent_data(ticker)

    if len(recent_df) < WINDOW_SIZE:
        print(f"ERROR: Need {WINDOW_SIZE} days of data, got {len(recent_df)}.")
        return

    recent_close = recent_df["Close"].values[-WINDOW_SIZE:]
    recent_scaled = scaler.transform(recent_close.reshape(-1, 1)).flatten()
    model_input = recent_scaled.reshape(1, -1)

    pred_scaled = model.predict(model_input, verbose=0)
    predicted_price = scaler.inverse_transform(pred_scaled).flatten()[0]

    last_close = recent_df["Close"].values[-1]
    last_date = recent_df.index[-1].strftime("%Y-%m-%d")

    print("\n========= LIVE PREDICTION =========")
    print(f"  Ticker:               {ticker}")
    print(f"  Last Close Date:      {last_date}")
    print(f"  Last Close Price:     ${last_close:.2f}")
    print(f"  Predicted Next Close: ${predicted_price:.2f}")
    print("====================================\n")


# ═══════════════════════════════════════════════════════════════════════════
#  7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Execute the complete training pipeline."""
    print("=" * 55)
    print("  STOCK PRICE PREDICTION USING ANN")
    print("=" * 55)

    # Step 1: Fetch
    print("\n[Step 1/5] Fetching data...")
    df = fetch_stock_data()

    # Step 2: Preprocess
    print("\n[Step 2/5] Preprocessing...")
    scaled, scaler = scale_data(df)
    X, y = create_sliding_windows(scaled)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Build & Train
    print("\n[Step 3/5] Building and training ANN...")
    model = build_ann()
    history = train_model(model, X_train, y_train)

    # Step 4: Evaluate
    print("\n[Step 4/5] Evaluating model...")
    y_actual, y_predicted = evaluate_model(model, X_test, y_test, scaler)

    # Step 5: Plots
    print("\n[Step 5/5] Generating plots...")
    plot_training_loss(history)
    plot_predictions(y_actual, y_predicted)

    print("\nPipeline complete successfully!")
    print("  Model saved in:  saved_models/")
    print("  Plots saved in:  plots/")
    print("\n  For live prediction run:")
    print("    python stock_predictor.py --live")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--live" in sys.argv:
        predict_next_close()
    else:
        run_full_pipeline()
