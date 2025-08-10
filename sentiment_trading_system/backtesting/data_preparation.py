
import yfinance as yf
import pandas as pd
import numpy as np

def prepare_backtest_data(ticker, period="2y", random_seed=None):
    """
    Prepares data for backtesting. A random_seed can be provided for reproducibility.
    """
    print(f"Fetching historical data for {ticker}...")
    hist_data = yf.download(ticker, period=period, interval="1d")
    if hist_data.empty:
        raise ValueError(f"No data found for ticker {ticker}. Please check the symbol.")
    hist_data.reset_index(inplace=True)

    print(f"Generating synthetic sentiment data with random seed: {random_seed}...")
    if random_seed is not None:
        np.random.seed(random_seed)

    hist_data['returns'] = hist_data['Close'].pct_change()
    sentiment_momentum = hist_data['returns'].rolling(window=5).mean().shift(-2)
    random_noise = np.random.normal(0, 0.5, len(hist_data))
    synthetic_sentiment = (sentiment_momentum * 5) + random_noise
    hist_data['sentiment'] = np.clip(synthetic_sentiment, -1, 1)
    confidence = 0.5 + (abs(hist_data['sentiment']) * 0.4) + np.random.normal(0, 0.05, len(hist_data))
    hist_data['confidence'] = np.clip(confidence, 0.3, 1.0)
    hist_data.fillna(0, inplace=True)
    print("Data preparation complete.")
    return hist_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'sentiment', 'confidence']]
