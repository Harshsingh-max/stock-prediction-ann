# Stock Market Price Prediction Using Artificial Neural Network (ANN)

A Python-based final-year college project that uses a feedforward Artificial Neural Network (MLP) to predict stock market closing prices based on historical data fetched from **yfinance**.

---

## Project Overview

This project downloads historical stock price data (default: Apple/AAPL), preprocesses it using a sliding window technique, trains a simple ANN model, evaluates its accuracy, and provides a live prediction script that forecasts the next day's closing price.

## How the ANN Works in This Project

1. **Data Collection**: We download 5 years of daily stock prices using the `yfinance` library.
2. **Normalization**: Close prices are scaled to a 0–1 range using `MinMaxScaler` so the neural network can learn effectively.
3. **Sliding Window**: We create input-output pairs — for every 20 consecutive days of prices (input), the 21st day's price is the target (output).
4. **ANN Architecture**: A feedforward neural network with:
   - Input layer: 20 neurons (one per day in the window)
   - Hidden layer 1: 64 neurons, ReLU activation
   - Hidden layer 2: 32 neurons, ReLU activation
   - Output layer: 1 neuron, linear activation (regression)
5. **Training**: The model learns to map patterns in 20-day windows to the next day's price using the Adam optimizer and Mean Squared Error loss.
6. **Prediction**: For live predictions, we take the most recent 20 days of prices, scale them, feed them into the trained model, and inverse-transform the output to get the predicted price in dollars.

## Project Structure

```
stock-prediction-ann/
├── data_fetch.py       # Downloads stock data from yfinance
├── preprocess.py       # Scaling, sliding windows, train/test split
├── model.py            # ANN model building, training, saving
├── evaluate.py         # Model evaluation (RMSE, MAE, MAPE) and plots
├── live_run.py         # Live prediction using saved model
├── main.py             # Runs the full pipeline
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── data/               # (created at runtime) Saved CSV data
├── saved_models/       # (created at runtime) Trained model & scaler
└── plots/              # (created at runtime) Generated plots
```

## File Descriptions

| File | Purpose |
|------|---------|
| `data_fetch.py` | Downloads historical and recent stock data using yfinance |
| `preprocess.py` | Normalizes data, creates sliding window features, splits chronologically |
| `model.py` | Defines, compiles, trains, and saves the Keras ANN model |
| `evaluate.py` | Evaluates model with RMSE/MAE/MAPE; plots actual vs predicted |
| `live_run.py` | Fetches latest data and predicts the next day's closing price |
| `main.py` | Orchestrates the entire training pipeline |

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Internet connection (for downloading stock data)

### Setup

```bash
# 1. Navigate to the project folder
cd stock-prediction-ann

# 2. (Recommended) Create a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## How to Run

### Train the Model (Full Pipeline)
```bash
python main.py
```
This will:
- Download 5 years of AAPL data
- Preprocess and split the data
- Train the ANN for 50 epochs
- Evaluate and print metrics (RMSE, MAE, MAPE)
- Save the model, scaler, and plots

### Live Prediction (After Training)
```bash
python live_run.py
```
This fetches the latest stock data and predicts the next day's closing price.

### Change the Stock Ticker
Edit the ticker in `main.py` or `live_run.py`:
```python
run_pipeline(ticker="GOOGL")       # in main.py
predict_next_close(ticker="GOOGL") # in live_run.py
```

---

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **RMSE** (Root Mean Squared Error) | Average prediction error in dollar terms; penalizes large errors |
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual prices |
| **MAPE** (Mean Absolute Percentage Error) | Average percentage error; useful for comparing across stocks |

---

## Limitations

1. **Not financial advice**: This is an academic project. Real stock trading involves much more complexity.
2. **Single feature**: The model only uses past close prices. Real models use volume, sentiment, indicators, etc.
3. **ANN vs specialized models**: LSTMs and Transformers are better suited for sequential/time-series data.
4. **Market randomness**: Stock prices are influenced by unpredictable events (news, policy changes, etc.).
5. **No real-time trading**: This project predicts the next close price but does not execute trades.
6. **Look-ahead bias prevention**: The chronological split prevents data leakage, but the model still only learns from historical patterns.

---

## Technologies Used

- **Python 3.9+**
- **yfinance** – Stock data API
- **pandas / numpy** – Data manipulation
- **scikit-learn** – MinMaxScaler, metrics
- **TensorFlow / Keras** – ANN model
- **matplotlib** – Visualization

---

## License

This project is for educational purposes only.
