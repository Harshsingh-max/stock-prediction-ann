"""
data_fetch.py
Downloads historical stock data from yfinance and saves it as a CSV file.
"""

import yfinance as yf
import pandas as pd
import os


def fetch_stock_data(ticker="AAPL", period="5y", save_path="data"):
    """
    Download historical stock data for the given ticker.

    Args:
        ticker: Stock ticker symbol (default: AAPL).
        period: How far back to fetch data (default: 5 years).
        save_path: Folder to save the CSV file.

    Returns:
        DataFrame with the stock data.
    """
    print(f"Downloading {ticker} data for the last {period}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    # Keep only useful columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"

    # Save to CSV
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f"{ticker}_historical.csv")
    df.to_csv(filepath)
    print(f"Saved {len(df)} rows to {filepath}")

    return df


def fetch_recent_data(ticker="AAPL", period="1mo"):
    """
    Fetch recent stock data for live prediction.
    We need at least `window_size` days of recent data.
    """
    print(f"Fetching recent {ticker} data...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No recent data found for '{ticker}'.")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


if __name__ == "__main__":
    data = fetch_stock_data()
    print(data.tail())
