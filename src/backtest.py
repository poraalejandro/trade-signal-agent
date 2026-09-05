"""Backtest the confluence logic (no LLM) over historical data per ticker."""

import pandas as pd
from confluence import CONFLUENCE_THRESHOLD, RSI_OVERBOUGHT, RSI_OVERSOLD
from indicators import (
    bollinger_bands,
    load_price_data,
    macd,
    moving_average_crossover,
    rsi,
)
from market_data import TICKERS

HOLD_DAYS = 7
TARGET_PCT = 0.05


def compute_confluence_signals(prices):
    """
    Compute a Call/Put/None confluence signal for every day in a price
    history, using the same 3-of-4 directional-agreement rule as agent.py's
    SYSTEM_INSTRUCTIONS: RSI, MACD, EMA Trend, and Bollinger Bands each cast
    a bullish/bearish/no vote, and a signal fires when at least 3 of the 4
    agree on the same direction.

    Args:
        prices: pandas Series of closing prices, indexed by date.

    Returns:
        A pandas Series of "Call"/"Put"/"None" values, indexed by date.
    """
    rsi_values = rsi(prices)
    rsi_vote = pd.Series("None", index=prices.index)

    rsi_bullish_mask = rsi_values < RSI_OVERSOLD
    rsi_bearish_mask = rsi_values > RSI_OVERBOUGHT

    rsi_vote[rsi_bullish_mask] = "Bullish"
    rsi_vote[rsi_bearish_mask] = "Bearish"

    macd_results = macd(prices)
    macd_vote = pd.Series("Bearish", index=prices.index)
    macd_vote[macd_results["MACD"] > macd_results["Signal Line"]] = "Bullish"

    # EMA Trend and Bollinger Signal already come out as
    # Bullish/Bearish/Mixed and Overbought/Oversold/Neutral respectively,
    # so we compare their raw values directly instead of building another
    # intermediate "vote" Series.
    ema_trend = moving_average_crossover(prices)["Trend"]
    bollinger_signal = bollinger_bands(prices)["Signal"]

    bullish_votes = (
        (rsi_vote == "Bullish").astype(int)
        + (macd_vote == "Bullish").astype(int)
        + (ema_trend == "Bullish").astype(int)
        + (bollinger_signal == "Oversold").astype(int)
    )
    bearish_votes = (
        (rsi_vote == "Bearish").astype(int)
        + (macd_vote == "Bearish").astype(int)
        + (ema_trend == "Bearish").astype(int)
        + (bollinger_signal == "Overbought").astype(int)
    )

    confluence_signal = pd.Series("None", index=prices.index)
    confluence_signal[bullish_votes >= CONFLUENCE_THRESHOLD] = "Call"
    confluence_signal[bearish_votes >= CONFLUENCE_THRESHOLD] = "Put"

    return confluence_signal


def evaluate_signal(
    prices, signal_date, direction, hold_days=HOLD_DAYS, target_pct=TARGET_PCT
):
    """
    Check whether a signal fired on signal_date would have hit its target
    move within the following hold_days trading days.

    Args:
        prices: pandas Series of closing prices, indexed by date.
        signal_date: the date the signal fired on (must exist in prices.index).
        direction: "Call" (bullish) or "Put" (bearish).
        hold_days: how many trading days after signal_date to look forward.
        target_pct: minimum favorable move (e.g. 0.05 for 5%) to count as a win.

    Returns:
        A dict with 'success' (bool), 'actual_return' (best move achieved,
        as a fraction), and 'excess_pct' (actual_return - target_pct).
    """

    signal_position = prices.index.get_loc(signal_date)
    future_window = prices.iloc[signal_position + 1 : signal_position + 1 + hold_days]
    signal_actual_price = prices.iloc[signal_position]

    if direction == "Call":
        success = future_window.max() >= signal_actual_price * (1 + target_pct)
        actual_return = (
            future_window.max() - signal_actual_price
        ) / signal_actual_price
    else:
        success = future_window.min() <= signal_actual_price * (1 - target_pct)
        actual_return = (
            signal_actual_price - future_window.min()
        ) / signal_actual_price

    excess_pct = actual_return - target_pct

    return {
        "success": success,
        "actual_return": actual_return,
        "excess_pct": excess_pct,
    }


def backtest_ticker(ticker, hold_days=HOLD_DAYS, target_pct=TARGET_PCT):
    """
    Backtest the confluence rule for a single ticker: fire signals over its
    full price history, evaluate each one that has enough future data, and
    report win rate and average return.

    Args:
        ticker: ticker symbol, e.g. "NVDA".
        hold_days: how many trading days to look forward per signal.
        target_pct: minimum favorable move to count as a win.

    Returns:
        A dict with 'ticker', 'total_signals', 'win_rate', and
        'avg_return'. win_rate/avg_return are None if no signal had
        enough future data to be evaluated.
    """
    ticker_prices = load_price_data(ticker)["Close"]
    signals = compute_confluence_signals(ticker_prices)
    results = []

    for signal_date, direction in signals.items():
        if direction == "None":
            continue
        signal_position = ticker_prices.index.get_loc(signal_date)
        if signal_position + hold_days < len(ticker_prices):
            result = evaluate_signal(
                ticker_prices,
                signal_date,
                direction,
                hold_days,
                target_pct,
            )
            results.append(result)

    total_signals = len(results)
    if total_signals == 0:
        return {
            "ticker": ticker,
            "total_signals": 0,
            "win_rate": None,
            "avg_return": None,
        }

    wins = sum(1 for result in results if result["success"])
    win_rate = wins / total_signals
    avg_return = sum(result["actual_return"] for result in results) / total_signals

    return {
        "ticker": ticker,
        "total_signals": total_signals,
        "win_rate": win_rate,
        "avg_return": avg_return,
    }


def backtest_all(tickers=TICKERS, hold_days=HOLD_DAYS, target_pct=TARGET_PCT):
    """
    Backtest the confluence rule across multiple tickers.

    Args:
        tickers: list of ticker symbols (default: TICKERS).
        hold_days: how many trading days to look forward per signal.
        target_pct: minimum favorable move to count as a win.

    Returns:
        A dict with 'per_ticker' (list of backtest_ticker results),
        'total_signals', 'overall_win_rate', and 'overall_avg_return'
        (the latter two weighted by each ticker's signal count, not a
        simple average across tickers).
    """
    results = []

    for ticker in tickers:
        results.append(backtest_ticker(ticker, hold_days, target_pct))

    total_signals = sum(r["total_signals"] for r in results)
    total_wins = sum(
        r["win_rate"] * r["total_signals"] for r in results if r["total_signals"] > 0
    )
    total_return_sum = sum(
        r["avg_return"] * r["total_signals"] for r in results if r["total_signals"] > 0
    )

    if total_signals == 0:
        overall_win_rate = None
        overall_avg_return = None
    else:
        overall_win_rate = total_wins / total_signals
        overall_avg_return = total_return_sum / total_signals

    return {
        "per_ticker": results,
        "total_signals": total_signals,
        "overall_win_rate": overall_win_rate,
        "overall_avg_return": overall_avg_return,
    }
