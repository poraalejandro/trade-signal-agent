trade-signal-agent
What this project is

An agent that flags potential call/put trade setups by combining technical indicator confluence with a fundamental filings check, and backtests those signals against historical data before trusting them. It never outputs a direct "execute this trade" recommendation — only flagged candidates with reasoning attached, for the user to review.

This is portfolio project #2 in a 3-project transition into AI engineering. Project #1 (github.com/poraalejandro/rag-finance) was a RAG system over SEC 10-K filings. This project's job is to demonstrate a different skill: an agent that reasons over multiple tool calls and makes an autonomous decision (flag vs. stay silent), not just retrieval.

Working style — read this before writing any code

Do not hand me finished code by default. I'm learning Python and AI engineering hands-on, coming from OutSystems/Java background. For any new piece of logic:

Explain the approach and what the function/module needs to do.
Let me write a first attempt myself.
Review it, correct what's wrong, and explain why — don't just replace it silently.

This applies to every file in this project, not just the first one. If I explicitly ask for something to be generated directly (e.g. boilerplate, repetitive config), that's fine — but the default assumption is: guide me, don't solve it for me.

Tickers (v1 scope)

NVDA, META, MSFT, TSM, IREN, NBIS

Stack decisions (already made — don't relitigate these)
Python, no heavy trading/backtesting frameworks. Indicators are hand-rolled (no pandas-ta), specifically so I understand each formula, not just call a library.
yfinance for historical OHLC price data (free, no API key needed).
Gemini API (gemini-2.5-flash, same as project #1) for the confluence reasoning step and for generating the explanation attached to each flagged signal.
SEC EDGAR fundamental-check logic is reused/adapted from the rag-finance project (same source, different purpose: checking for recent/upcoming filings, not Q&A).
Output interface: a Streamlit app, consistent with project #1.
Build order (don't skip ahead)
src/market_data.py — download historical OHLC for the tickers above via yfinance.
src/indicators.py — hand-rolled technical indicators: RSI, MACD, moving average crossover, Bollinger Bands, volume anomaly. Each indicator is an independent function taking a price series and returning a value/signal.
src/fundamental_check.py — reuse SEC EDGAR fetching logic to detect a recent or imminent filing/earnings event for a given ticker.
src/agent.py — the core piece. Defines the above as tools (real function calling, not a fixed pipeline). For each ticker: call indicator tools, call the fundamental check, then reason over the combination (confluence) — does the technical picture agree across indicators, and does anything fundamental contradict or support it? If yes → emit a flagged candidate with reasoning. If no → emit nothing. Silence is a valid, expected outcome, not a failure state.
src/backtest.py — run the confluence logic (simplified — don't call the LLM per historical step, that's slow and expensive) over N months of past data per ticker. For each signal that would have fired: did price move favorably by at least X% within Y days? Report win rate and average return. This is the evaluation harness — same role RAGAS played for project #1. No backtest numbers, no claim that this works.
app.py — Streamlit UI wrapping the above.
Hard rules
Never output or imply a direct trade execution instruction. Flag candidates with reasoning only — the human reviews and decides.
.env, .venv/, __pycache__/, and any credentials go in .gitignore from the very first commit — this project holds a Gemini API key like project #1 did.
Commit in small, descriptive steps per module (matches the build order above), not one giant commit at the end — the git history is part of what a recruiter reviews.