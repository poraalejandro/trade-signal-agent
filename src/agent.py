"""Core agent: calls indicator/fundamental tools per ticker and reasons over confluence."""

import os
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from fundamental_check import check_recent_filings
from indicators import (
    bollinger_bands,
    load_price_data,
    macd,
    moving_average_crossover,
    rsi,
    volume_anomaly,
)
from market_data import TICKERS

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not set in .env file")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_INSTRUCTIONS = """
You are a trading-signal screening assistant. Your job is to analyze a single
stock ticker by combining technical indicator confluence with a fundamental
filings check, and decide whether it's worth flagging as a candidate for a
human to review — never to recommend or imply a direct trade execution.

For the given ticker:
1. Call the available tools to gather the current technical indicator
   readings (RSI, MACD, EMA trend/crossover, Bollinger Bands, volume) and
   the fundamental filings check.
2. Reason over confluence: do multiple technical indicators agree with each
   other (not just one in isolation)? Does the fundamental filing
   information support, contradict, or add relevant context to that
   technical picture?
3. Decide whether to flag:
   - Confluence means at least 3 of the 4 directional indicators (RSI,
     MACD, EMA Trend, Bollinger Bands) agree on the same direction
     (Bullish/Oversold => bullish case; Bearish/Overbought => bearish
     case). Neutral/Mixed readings do not count toward either side.
   - Volume anomaly and the fundamental filings check are context, not
     votes: use them to strengthen or add caution to your reasoning, but
     the 3-of-4 majority is the flagging threshold, not a strict veto.
   - If fewer than 3 of the 4 agree, do NOT flag — silence is valid.

Hard rules:
- NEVER state or imply a direct trade execution instruction (e.g. "buy",
  "sell", "execute"). Only describe what you observed and why it's worth a
  human's attention.
- Ground your reasoning in the specific tool results you retrieved — do not
  make vague or unsupported claims.

Respond with the ticker analyzed, whether you are flagging it, the
direction (call/put) only if flagged, a concise reasoning explaining the
confluence (or lack of it), and which specific signals support your
conclusion.
"""

class TradeSignal(BaseModel):
    ticker: str
    flagged: bool
    direction: Literal["call", "put"] | None = None
    reasoning: str
    supporting_signals: list[str]


def get_rsi_signal(ticker: str) -> dict:
    """
    Get the current RSI (Relative Strength Index) for a ticker, to check
    for overbought/oversold momentum conditions.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
    """
    rsi_results = rsi(load_price_data(ticker)["Close"]).iloc[-1]
    if rsi_results > 70:
        rsi_status = "Overbought"
    elif rsi_results < 30:
        rsi_status = "Oversold"
    else:
        rsi_status = "Neutral"

    return {"RSI": round(rsi_results, 2), "Signal": rsi_status}


def get_macd_signal(ticker: str) -> dict:
    """
    Get the current MACD (Moving Average Convergence Divergence) for a
    ticker, to check for bullish/bearish momentum shifts.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
    """
    macd_results = macd(load_price_data(ticker)["Close"]).iloc[-1].round(2)

    if macd_results["MACD"] > macd_results["Signal Line"]:
        macd_signal = "Bullish"
    else:
        macd_signal = "Bearish"

    result = macd_results.to_dict()
    result["Signal"] = macd_signal

    return result


def get_crossover_signal(ticker: str) -> dict:
    """
    Get the current EMA trend alignment (25/50/200) for a ticker, whether a
    golden/death cross happened on the 50/200 pair in the last 5 trading
    days, and whether one appears to be approaching (the 50/200 gap has
    been narrowing over that same window without crossing yet).

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
    """
    crossover_data = moving_average_crossover(load_price_data(ticker)["Close"])
    recent = crossover_data.tail(5)
    actual_trend = recent["Trend"].iloc[-1]
    recent_crosses = recent[recent["CrossEvent"] != "None"]

    if not recent_crosses.empty:
        recent_cross_event = recent_crosses["CrossEvent"].iloc[-1]
        days_since_cross = (recent.index[-1] - recent_crosses.index[-1]).days
    else:
        recent_cross_event = "None"
        days_since_cross = None

    gap_now = recent["EMA_mid"].iloc[-1] - recent["EMA_long"].iloc[-1]
    gap_5_days_ago = recent["EMA_mid"].iloc[0] - recent["EMA_long"].iloc[0]

    if recent_cross_event == "None" and abs(gap_now) < abs(gap_5_days_ago):
        approaching_cross = (
            "Approaching Death Cross" if gap_now > 0 else "Approaching Golden Cross"
        )
    else:
        approaching_cross = "None"

    return {
        "Trend": actual_trend,
        "RecentCross": recent_cross_event,
        "DaysSinceCross": days_since_cross,
        "ApproachingCross": approaching_cross,
    }


def get_bollinger_bands(ticker: str) -> dict:
    """
    Get the current Bollinger Bands for a ticker, to check if price is
    trading outside its normal volatility range (overbought/oversold).

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
    """
    return bollinger_bands(load_price_data(ticker)["Close"]).round(2).iloc[-1].to_dict()


def get_volume_anomaly(ticker: str) -> dict:
    """
    Get today's trading volume for a ticker relative to its recent average,
    to check for unusually high activity that may signal a significant move.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
    """
    return volume_anomaly(load_price_data(ticker)["Volume"]).round(2).iloc[-1].to_dict()


TOOLS = [
    get_rsi_signal,
    get_macd_signal,
    get_crossover_signal,
    get_bollinger_bands,
    get_volume_anomaly,
    check_recent_filings,
]


def analyze_ticker(ticker: str) -> TradeSignal:
    """
    Analyze a ticker via the confluence agent: lets the model call the
    available tools freely, then asks it to commit to a structured
    decision (flag or stay silent) based on what it found.
    """
    gather_response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"Analyze {ticker}.",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            tools=TOOLS,
        ),
    )

    # automatic_function_calling_history holds the tool-call turns, but not
    # the model's final synthesized text — append that turn explicitly so
    # the structured-output call has the full context.
    history = gather_response.automatic_function_calling_history + [
        gather_response.candidates[0].content
    ]
    history.append("Now respond with your final decision in the required structured format.")

    decision_response = client.models.generate_content(
        model=MODEL_NAME,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            response_mime_type="application/json",
            response_schema=TradeSignal,
        ),
    )

    return decision_response.parsed
