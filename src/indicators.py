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


def moving_average_crossover(prices, short_period=25, mid_period=50, long_period=200):
    """
    Classify trend alignment across a short/mid/long EMA triple.

    Args:
        prices: pandas Series of closing prices, indexed by date.
        short_period: period for the short EMA (default 25).
        mid_period: period for the mid EMA (default 50).
        long_period: period for the long EMA (default 200).

    Returns:
        A pandas DataFrame with columns 'EMA_short', 'EMA_mid', 'EMA_long',
        'Trend' ('Bullish'/'Bearish'/'Mixed'), and 'CrossEvent'
        ('Golden Cross'/'Death Cross'/'None' on the mid/long pair).
    """
    ema_short = ema(prices, span=short_period)
    ema_mid = ema(prices, span=mid_period)
    ema_long = ema(prices, span=long_period)

    bullish_mask = (ema_short > ema_mid) & (ema_mid > ema_long)
    bearish_mask = (ema_short < ema_mid) & (ema_mid < ema_long)

    trend = pd.Series("Mixed", index=prices.index)
    trend[bullish_mask] = "Bullish"
    trend[bearish_mask] = "Bearish"

    mid_above_long_today = ema_mid > ema_long
    mid_above_long_yesterday = mid_above_long_today.shift(1, fill_value=False)

    golden_cross = mid_above_long_today & ~mid_above_long_yesterday
    death_cross = ~mid_above_long_today & mid_above_long_yesterday

    cross_event = pd.Series("None", index=prices.index)
    cross_event[golden_cross] = "Golden Cross"
    cross_event[death_cross] = "Death Cross"

    return pd.DataFrame(
        {
            "EMA_short": ema_short,
            "EMA_mid": ema_mid,
            "EMA_long": ema_long,
            "Trend": trend,
            "CrossEvent": cross_event,
        }
    )


def bollinger_bands(prices, period=20, num_std=2):
    """
    Compute Bollinger Bands over a closing-price series.

    Args:
        prices: pandas Series of closing prices, indexed by date.
        period: lookback window for the SMA and standard deviation (default 20).
        num_std: number of standard deviations for the upper/lower bands (default 2).

    Returns:
        A pandas DataFrame with columns 'Middle' (SMA), 'Upper', 'Lower', and
        'Signal' ('Overbought'/'Oversold'/'Neutral' based on price vs. the bands).
    """
    sma = prices.rolling(window=period).mean()
    standard_deviation = prices.rolling(window=period).std()
    upper_end = sma + (num_std * standard_deviation)
    lower_end = sma - (num_std * standard_deviation)

    overbought_mask = prices > upper_end
    oversold_mask = prices < lower_end
    trend = pd.Series("Neutral", index=prices.index)
    trend[overbought_mask] = "Overbought"
    trend[oversold_mask] = "Oversold"

    return pd.DataFrame(
        {
            "Middle": sma,
            "Upper": upper_end,
            "Lower": lower_end,
            "Signal": trend,
        }
    )


def volume_anomaly(volume, period=20, threshold=2):
    """
    Detect abnormally high trading volume relative to its recent average.

    Args:
        volume: pandas Series of daily trading volume, indexed by date.
        period: lookback window for the average volume (default 20).
        threshold: ratio of today's volume to the average above which a day
            is flagged as anomalous (default 2, i.e. double the average).

    Returns:
        A pandas DataFrame with columns 'AvgVolume', 'Ratio', and 'Signal'
        ('High Volume'/'Normal'). AvgVolume excludes the current day itself
        (computed over the preceding `period` days) to avoid a spike
        inflating its own baseline.
    """
    avg_volume = volume.shift(1).rolling(window=period).mean()
    ratio = volume / avg_volume

    high_volume_mask = ratio > threshold

    volume_signal = pd.Series("Normal", index=volume.index)
    volume_signal[high_volume_mask] = "High Volume"

    return pd.DataFrame(
        {
            "AvgVolume": avg_volume,
            "Ratio": ratio,
            "Signal": volume_signal,
        }
    )
