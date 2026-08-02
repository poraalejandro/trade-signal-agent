"""Hand-rolled technical indicators. Each function takes a price (or volume) Series and returns a value/signal."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def load_price_data(ticker):
    """
    Load a ticker's OHLC history from its CSV in DATA_DIR.

    Args:
        ticker: ticker symbol, e.g. "NVDA".

    Returns:
        A pandas DataFrame indexed by date, with OHLC + Volume columns.
    """
    file_path = DATA_DIR / f"{ticker}.csv"

    data_frame = pd.read_csv(file_path, index_col=0)
    data_frame.index = pd.to_datetime(data_frame.index, utc=True)
    return data_frame


def rsi(prices, period=14):
    """
    Compute the Relative Strength Index over a closing-price series.

    Args:
        prices: pandas Series of closing prices, indexed by date.
        period: lookback window for average gain/loss (default 14).

    Returns:
        A pandas Series of RSI values, indexed by date.
    """
    prices_diff = prices.diff()
    gains = prices_diff.where(prices_diff > 0, 0)
    losses = -prices_diff.where(prices_diff < 0, 0)

    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    flat_rsi_mask = (avg_loss == 0) & (avg_gain == 0)

    rsi_values[flat_rsi_mask] = (
        50  # RSI is undefined when both avg_gain and avg_loss are 0; set to 50 as a placeholder.
    )

    return rsi_values


def ema(prices, span):
    """
    Compute the Exponential Moving Average over a closing-price series.

    Args:
        prices: pandas Series of closing prices, indexed by date.
        span: smoothing factor for EMA.

    Returns:
        A pandas Series of EMA values, indexed by date.
    """
    return prices.ewm(span=span, adjust=False).mean()


def macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Compute the Moving Average Convergence Divergence (MACD) indicator.

    Args:
        prices: pandas Series of closing prices, indexed by date.
        fast_period: period for the fast EMA (default 12).
        slow_period: period for the slow EMA (default 26).
        signal_period: period for the signal line EMA (default 9).

    Returns:
        A pandas DataFrame with columns 'MACD', 'Signal' and 'Histogram', indexed by date.
    """
    ema_fast = ema(prices, span=fast_period)
    ema_slow = ema(prices, span=slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, span=signal_period)

    return pd.DataFrame(
        {"MACD": macd_line, "Signal": signal_line, "Histogram": macd_line - signal_line}
    )
