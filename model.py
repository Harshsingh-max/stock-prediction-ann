"""
model.py
Builds, trains, and saves the ANN (feedforward MLP) model using TensorFlow/Keras.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import os


def build_ann(input_dim, units_1=64, units_2=32):
    """
    Build a simple feedforward Artificial Neural Network (MLP).

    Architecture:
        Input (window_size features)
        -> Dense(64, relu)
        -> Dense(32, relu)
        -> Dense(1, linear)   # regression output

    This is a fully-connected network that takes a window of past prices
    and predicts the next day's closing price.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(units_1, activation="relu"),
        layers.Dense(units_2, activation="relu"),
        layers.Dense(1)  # linear activation for regression
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )

    model.summary()
    return model


def train_model(model, X_train, y_train, epochs=50, batch_size=32,
                validation_split=0.1, save_path="saved_models"):
    """
    Train the ANN model and save it.

    Returns:
        history: training history object (contains loss values for plotting)
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # Save the trained model
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "stock_ann_model.keras")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    return history


def plot_training_loss(history, save_path="plots"):
    """
    Plot training and validation loss curves.
    """
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Model Training Loss", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(save_path, "training_loss.png")
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"Loss plot saved to {filepath}")


def load_trained_model(model_path="saved_models/stock_ann_model.keras"):
    """Load a previously saved model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Train the model first.")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return model
