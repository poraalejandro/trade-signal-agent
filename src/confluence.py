"""Shared confluence rule: thresholds and vote logic used by both agent.py
(live tool wrappers) and backtest.py (historical evaluation), so the two
can never independently drift out of sync on what "confluence" means."""

RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
CONFLUENCE_THRESHOLD = 3


def classify_rsi(rsi_value):
    """Map a raw RSI value to Oversold/Overbought/Neutral."""
    if rsi_value < RSI_OVERSOLD:
        return "Oversold"
    if rsi_value > RSI_OVERBOUGHT:
        return "Overbought"
    return "Neutral"


def classify_macd(macd_value, signal_line_value):
    """Map MACD vs its Signal Line to Bullish/Bearish."""
    return "Bullish" if macd_value > signal_line_value else "Bearish"


def vote_from_signal(signal_label):
    """Map any indicator's qualitative label to Bullish/Bearish/None, for
    counting toward the confluence majority."""
    if signal_label in ("Oversold", "Bullish"):
        return "Bullish"
    if signal_label in ("Overbought", "Bearish"):
        return "Bearish"
    return "None"
