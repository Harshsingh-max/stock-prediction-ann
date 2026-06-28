"""
main.py
Main entry point that runs the entire pipeline:
    1. Fetch data
    2. Preprocess
    3. Build and train ANN
    4. Evaluate
    5. Plot results

Run this file to train the model from scratch.
After training, use live_run.py for predictions on new data.
"""

from data_fetch import fetch_stock_data
from preprocess import preprocess_pipeline
from model import build_ann, train_model, plot_training_loss
from evaluate import evaluate_model, plot_predictions


def run_pipeline(ticker="AAPL", period="5y", window_size=20,
                 epochs=50, batch_size=32):
    """
    Execute the full training pipeline.
    """
    print("=" * 50)
    print("  STOCK PRICE PREDICTION USING ANN")
    print("=" * 50)

    # Step 1: Fetch historical data
    print("\n[Step 1/5] Fetching data...")
    df = fetch_stock_data(ticker=ticker, period=period)

    # Step 2: Preprocess data
    print("\n[Step 2/5] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        df, window_size=window_size
    )

    # Step 3: Build and train the ANN model
    print("\n[Step 3/5] Building and training ANN model...")
    model = build_ann(input_dim=window_size)
    history = train_model(
        model, X_train, y_train,
        epochs=epochs, batch_size=batch_size
    )

    # Step 4: Evaluate on test data
    print("\n[Step 4/5] Evaluating model...")
    y_actual, y_predicted, metrics = evaluate_model(model, X_test, y_test, scaler)

    # Step 5: Generate plots
    print("\n[Step 5/5] Generating plots...")
    plot_training_loss(history)
    plot_predictions(y_actual, y_predicted)

    print("\nPipeline complete!")
    print("Model and scaler saved in 'saved_models/' folder.")
    print("Plots saved in 'plots/' folder.")
    print("\nTo make a live prediction, run:")
    print("  python live_run.py")


if __name__ == "__main__":
    run_pipeline()
