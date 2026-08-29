"""Core agent: calls indicator/fundamental tools per ticker and reasons over confluence."""

import json
import os
from typing import Literal

from dotenv import load_dotenv
from groq import Groq
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
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError("GROQ_API_KEY not set in .env file")

client = Groq(api_key=api_key, max_retries=6)

MODEL_NAME = "qwen/qwen3.8-27b"

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


class TradeSignalList(BaseModel):
    signals: list[TradeSignal]


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


TOOL_FUNCTIONS = {
    "get_rsi_signal": get_rsi_signal,
    "get_macd_signal": get_macd_signal,
    "get_crossover_signal": get_crossover_signal,
    "get_bollinger_bands": get_bollinger_bands,
    "get_volume_anomaly": get_volume_anomaly,
    "check_recent_filings": check_recent_filings,
}


def build_tool_schema(name: str, description: str) -> dict:
    """
    Build a Groq/OpenAI-style function-calling schema for one of our tools.
    Every tool here takes the same single argument, so this one helper
    covers all 6 instead of writing out 6 near-identical schemas by hand.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": 'Stock ticker symbol, e.g. "NVDA".',
                    }
                },
                "required": ["ticker"],
            },
        },
    }


TOOLS = [
    build_tool_schema(
        "get_rsi_signal",
        "Get the current RSI for a ticker, to check for overbought/oversold "
        "momentum conditions.",
    ),
    build_tool_schema(
        "get_macd_signal",
        "Get the current MACD for a ticker, to check for bullish/bearish "
        "momentum shifts.",
    ),
    build_tool_schema(
        "get_crossover_signal",
        "Get the current EMA trend alignment (25/50/200) for a ticker, "
        "recent golden/death cross events, and whether one appears to be "
        "approaching.",
    ),
    build_tool_schema(
        "get_bollinger_bands",
        "Get the current Bollinger Bands for a ticker, to check if price is "
        "trading outside its normal volatility range.",
    ),
    build_tool_schema(
        "get_volume_anomaly",
        "Get today's trading volume for a ticker relative to its recent "
        "average, to check for unusually high activity.",
    ),
    build_tool_schema(
        "check_recent_filings",
        "Check days since a ticker's most recent earnings-related SEC "
        "filing and most recent annual report.",
    ),
]


def run_conversation_with_tools(messages: list[dict]) -> list[dict]:
    """
    Manual tool-calling loop: ask the model, execute any tool calls it
    requests, feed the results back, and repeat until it stops requesting
    tools. Returns the full message history including the final reply.
    """
    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return messages

        for tool_call in message.tool_calls:
            function = TOOL_FUNCTIONS[tool_call.function.name]
            arguments = json.loads(tool_call.function.arguments)
            result = function(**arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )


def analyze_ticker(ticker: str) -> TradeSignal:
    """
    Analyze a ticker via the confluence agent: lets the model call the
    available tools freely, then asks it to commit to a structured
    decision (flag or stay silent) based on what it found.
    """
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Analyze {ticker}."},
    ]
    messages = run_conversation_with_tools(messages)
    messages.append(
        {
            "role": "user",
            "content": "Now respond with your final decision in the required structured format.",
        }
    )

    decision = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "TradeSignal",
                "schema": TradeSignal.model_json_schema(),
            },
        },
    )

    signal = TradeSignal.model_validate_json(decision.choices[0].message.content)

    # The model doesn't always respect "direction only if flagged" — enforce
    # it in code rather than trusting the prompt alone.
    if not signal.flagged:
        signal.direction = None

    return signal


def analyze_all_tickers(tickers: list[str] = TICKERS) -> list[TradeSignal]:
    """
    Analyze multiple tickers in a single conversation, instead of calling
    analyze_ticker once per ticker, to conserve API requests.
    """
    ticker_list = ", ".join(tickers)
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {
            "role": "user",
            "content": (
                f"Analyze each of these tickers, one at a time: {ticker_list}. "
                "Call the tools as needed for each ticker before moving to the next."
            ),
        },
    ]
    messages = run_conversation_with_tools(messages)
    messages.append(
        {
            "role": "user",
            "content": (
                "Now respond with your final decision for EVERY ticker you were "
                "asked to analyze, one entry per ticker, in the required "
                "structured format."
            ),
        }
    )

    decision = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "TradeSignalList",
                "schema": TradeSignalList.model_json_schema(),
            },
        },
    )

    signals = TradeSignalList.model_validate_json(
        decision.choices[0].message.content
    ).signals

    for signal in signals:
        if not signal.flagged:
            signal.direction = None

    return signals
