# trade-signal-agent

An agent that flags potential call/put trade setups by combining technical indicator confluence with a fundamental SEC filings check, then backtests that logic against historical data before trusting it.

**It never outputs a direct "execute this trade" instruction.** It only surfaces flagged candidates with reasoning attached, for a human to review. Silence — no candidates flagged — is a valid, expected outcome, not a failure state.

This is portfolio project #2 in a 3-project transition into AI engineering. Project #1, [rag-finance](https://github.com/poraalejandro/rag-finance), was a RAG system over SEC 10-K filings. This project demonstrates a different skill: an agent that reasons over multiple tool calls and makes an autonomous decision (flag vs. stay silent), not just retrieval.

## How it works

1. **`market_data.py`** downloads historical daily OHLC data for the ticker universe via [yfinance](https://github.com/ranaroussi/yfinance), dropping any incomplete same-day rows (a session that hadn't closed yet when downloaded).
2. **`indicators.py`** — five hand-rolled technical indicators (no `pandas-ta`, so every formula is understood, not just called): RSI, MACD, a 25/50/200 EMA crossover with trend alignment and golden/death cross detection, Bollinger Bands, and volume anomaly detection.
3. **`fundamental_check.py`** queries SEC EDGAR directly for how recently a ticker filed an earnings-related report (8-K/10-Q, or 6-K for foreign private issuers) or an annual report (10-K, or 20-F for foreign private issuers) — metadata only, no document content is downloaded or read.
4. **`agent.py`** is the core piece: each indicator and the filings check is exposed as a tool the LLM ([Groq](https://groq.com/), `qwen/qwen3.8-27b`) can call freely for a given ticker. Once it has gathered what it needs, it commits to a structured decision — flag or stay silent — following one hard rule: **confluence requires at least 3 of 4 directional indicators (RSI, MACD, EMA trend, Bollinger Bands) to agree on the same direction.** Volume and filings are context that shapes the reasoning, not votes.
5. **`backtest.py`** evaluates that exact same 3-of-4 rule in plain code — no LLM calls — over the full price history of each ticker. For every day a signal would have fired, it checks whether price moved at least 5% in the signaled direction within the following 7 trading days, and reports win rate and average return, per ticker and aggregated. This is the evaluation harness, the same role [RAGAS](https://github.com/explodinggears/ragas) played for project #1: an honest check on whether the logic works, not a claim that it does.
6. **`app.py`** is a [Streamlit](https://streamlit.io/) UI over all of the above: pick a ticker (or all six) to analyze live, see a price chart with an optional Bollinger Bands / EMA trend overlay, a per-indicator vote breakdown showing how close RSI/MACD/EMA trend/Bollinger are to flagging, and the backtest stats per ticker and aggregated.

## Tickers (v1 scope)

`NVDA, META, MSFT, TSM, IREN, NBIS`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root with a [Groq API key](https://console.groq.com):

```env
GROQ_API_KEY=your_key_here
```

## Usage

```bash
python src/market_data.py        # download/refresh price history for the ticker universe

cd src
python -c "from agent import analyze_ticker; print(analyze_ticker('NVDA'))"
python -c "from backtest import backtest_all; print(backtest_all())"
```

(Modules import each other as `from indicators import ...` rather than `from src.indicators import ...`, so they need to be run with `src/` itself as the working directory, not the project root.)

To run the Streamlit app (from the project root, not `src/`):

```bash
streamlit run app.py
```

## Stack

Python · [yfinance](https://github.com/ranaroussi/yfinance) · [Groq](https://groq.com/) (`qwen/qwen3.8-27b`) · [Pydantic](https://docs.pydantic.dev/) for structured LLM output · [pandas](https://pandas.pydata.org/) · [Altair](https://altair-viz.github.io/) · [Streamlit](https://streamlit.io/)
