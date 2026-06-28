"""
evaluate.py
Evaluates the trained model on test data using RMSE, MAE, and MAPE.
Generates actual vs predicted comparison plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


def calculate_metrics(y_actual, y_predicted):
    """
    Calculate regression evaluation metrics.

    Returns:
        dict with RMSE, MAE, and MAPE values.
    """
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mae = mean_absolute_error(y_actual, y_predicted)

    # MAPE: Mean Absolute Percentage Error
    # Avoid division by zero
    mask = y_actual != 0
    mape = np.mean(np.abs((y_actual[mask] - y_predicted[mask]) / y_actual[mask])) * 100

    metrics = {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape}

    print("\n===== Evaluation Metrics =====")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("==============================\n")

    return metrics


def evaluate_model(model, X_test, y_test, scaler):
    """
    Run predictions on test data and evaluate.

    The predictions and actuals are inverse-transformed back to original
    price scale for meaningful comparison.

    Returns:
        y_actual: actual prices (original scale)
        y_predicted: predicted prices (original scale)
        metrics: dict of evaluation metrics
    """
    # Predict on test data (scaled)
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform to get real prices
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_predicted = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    metrics = calculate_metrics(y_actual, y_predicted)

    return y_actual, y_predicted, metrics


def plot_predictions(y_actual, y_predicted, save_path="plots"):
    """
    Plot actual vs predicted stock prices.
    """
    os.makedirs(save_path, exist_ok=True)

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

    filepath = os.path.join(save_path, "actual_vs_predicted.png")
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"Prediction plot saved to {filepath}")
