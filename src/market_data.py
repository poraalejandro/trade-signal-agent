"""Download historical OHLC data for the v1 ticker universe and store it as CSV."""

import yfinance as yf
from pathlib import Path

# Tickers in scope for v1 (see CLAUDE.md)
TICKERS = ["NVDA", "META", "MSFT", "AAPL", "TSM"]

# Where each ticker's CSV will be written (data/<TICKER>.csv)
DATA_DIR = Path("data")

# How far back to pull daily history. Needs to be long enough for backtest.py
# to have a meaningful sample, without mixing too many different market regimes.
PERIOD = "5y"


def fetch_ticker_data(ticker):
    """
    Download historical daily OHLC data for a single ticker.


    Args:
        ticker: ticker symbol, e.g. "NVDA".

    Returns:
        A pandas DataFrame with OHLC columns, indexed/sorted by date.
    """
    ticker_history = yf.Ticker(ticker).history(period=PERIOD)
    if ticker_history.empty:
        print(f"Warning: No data found for {ticker}.")
        return None

    return ticker_history


def save_ticker_data(df, ticker):
    """
    Persist a ticker's DataFrame to data/<ticker>.csv.

    Args:
        df: DataFrame returned by fetch_ticker_data.
        ticker: ticker symbol, used to build the output filename.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(DATA_DIR / f"{ticker}.csv", index=True)


def main():
    for ticker in TICKERS:
        df = fetch_ticker_data(ticker)
        if df is not None:
            save_ticker_data(df, ticker)
            print(f"{len(df)} rows of data saved for {ticker}.")


if __name__ == "__main__":
    main()
