"""Streamlit UI wrapping the confluence agent and backtest harness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import altair as alt
import pandas as pd
import streamlit as st

from agent import (
    analyze_ticker,
    get_bollinger_bands,
    get_crossover_signal,
    get_macd_signal,
    get_rsi_signal,
)
from backtest import backtest_all
from indicators import bollinger_bands, load_price_data, moving_average_crossover
from market_data import TICKERS

CHART_LOOKBACK_DAYS = 180
CHART_OVERLAYS = ["None", "Bollinger Bands", "EMA Trend (25/50/200)"]

# Validated categorical palette (blue/orange/aqua slots 1-3 — the only three
# of the eight hues that clear the strict all-pairs CVD/contrast checks
# together), light/dark variants per Streamlit's active theme.
PALETTE = {
    "light": {"ink": "#0b0b0b", "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"},
    "dark": {"ink": "#ffffff", "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70"},
}

st.set_page_config(page_title="trade-signal-agent", page_icon="\U0001F4C8")

# st.context.theme.type is unreliable on the very first run of a session
# (documented Streamlit limitation, github.com/streamlit/streamlit/issues/11920)
# — force one extra rerun up front so it has settled before we pick colors.
if "theme_settled" not in st.session_state:
    st.session_state.theme_settled = True
    st.rerun()

st.title("trade-signal-agent")
st.caption(
    "Flags potential call/put candidates from technical + fundamental "
    "confluence. Never a trade instruction — review and decide for yourself."
)


DATA_DIR = Path(__file__).parent / "data"


def data_file_mtime(ticker: str) -> float:
    """Last-modified time of a ticker's CSV, used to invalidate caches when
    market_data.py refreshes the data — Streamlit's cache has no way to
    know the file on disk changed unless it's part of the cached args."""
    return (DATA_DIR / f"{ticker}.csv").stat().st_mtime


@st.cache_data
def cached_analyze_ticker(ticker: str, data_version: float):
    return analyze_ticker(ticker)


@st.cache_data
def cached_backtest_all(data_version: float):
    return backtest_all()


def build_price_chart(ticker, overlay, colors):
    """
    Build a price chart for a ticker, optionally layering Bollinger Bands or
    the 25/50/200 EMA trend on top, over the last CHART_LOOKBACK_DAYS days.
    """
    prices = load_price_data(ticker)["Close"]

    price_df = prices.tail(CHART_LOOKBACK_DAYS).reset_index()
    price_df.columns = ["Date", "Price"]

    price_line = (
        alt.Chart(price_df)
        .mark_line(strokeWidth=2, color=colors["ink"])
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Price:Q", title="Price (USD)", scale=alt.Scale(zero=False)),
        )
    )

    layers = [price_line]
    caption = None

    if overlay == "Bollinger Bands":
        bands = bollinger_bands(prices).tail(CHART_LOOKBACK_DAYS).reset_index()
        bands = bands.rename(columns={bands.columns[0]: "Date"})

        band_area = (
            alt.Chart(bands)
            .mark_area(opacity=0.15, color=colors["blue"])
            .encode(x="Date:T", y="Lower:Q", y2="Upper:Q")
        )
        middle_line = (
            alt.Chart(bands)
            .mark_line(strokeWidth=1, strokeDash=[4, 3], color=colors["blue"])
            .encode(x="Date:T", y="Middle:Q")
        )
        layers = [band_area, middle_line, price_line]
        caption = "Shaded band: Bollinger Bands (20-day, 2 std dev). Dashed line: 20-day SMA."

    elif overlay == "EMA Trend (25/50/200)":
        ema = moving_average_crossover(prices).tail(CHART_LOOKBACK_DAYS).reset_index()
        ema = ema.rename(columns={ema.columns[0]: "Date"})
        ema_long_df = ema.melt(
            id_vars="Date",
            value_vars=["EMA_short", "EMA_mid", "EMA_long"],
            var_name="EMA",
            value_name="Value",
        )
        ema_lines = (
            alt.Chart(ema_long_df)
            .mark_line(strokeWidth=1.5)
            .encode(
                x="Date:T",
                y="Value:Q",
                color=alt.Color(
                    "EMA:N",
                    scale=alt.Scale(
                        domain=["EMA_short", "EMA_mid", "EMA_long"],
                        range=[colors["blue"], colors["orange"], colors["aqua"]],
                    ),
                    legend=alt.Legend(title=None),
                ),
            )
        )
        layers = [ema_lines, price_line]
        caption = "Colored lines: 25/50/200-day EMAs. Solid line: closing price."

    hover = alt.selection_point(
        fields=["Date"], nearest=True, on="pointerover", empty=False, clear="pointerout"
    )
    selectors = (
        alt.Chart(price_df)
        .mark_point()
        .encode(x="Date:T", opacity=alt.value(0))
        .add_params(hover)
    )
    crosshair = (
        alt.Chart(price_df)
        .mark_rule(color=colors["ink"], strokeWidth=1, opacity=0.3)
        .encode(x="Date:T")
        .transform_filter(hover)
    )
    hover_point = price_line.mark_point(size=45, color=colors["ink"]).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )
    hover_label = (
        alt.Chart(price_df)
        .mark_text(align="left", dx=7, dy=-8, color=colors["ink"], fontSize=12)
        .encode(
            x="Date:T",
            y="Price:Q",
            text=alt.condition(hover, alt.Text("Price:Q", format="$.2f"), alt.value("")),
        )
    )
    layers += [selectors, crosshair, hover_point, hover_label]

    chart = alt.layer(*layers).properties(height=350)
    return chart, caption


def vote_from_signal(signal_label):
    """Map an indicator's Signal label to Bullish/Bearish/None, mirroring
    the 3-of-4 confluence rule in agent.py's SYSTEM_INSTRUCTIONS."""
    if signal_label in ("Oversold", "Bullish"):
        return "Bullish"
    if signal_label in ("Overbought", "Bearish"):
        return "Bearish"
    return "None"


def render_vote_metric(column, label, value, vote, help_text=None):
    delta_color = {"Bullish": "green", "Bearish": "red", "None": "gray"}[vote]
    delta_arrow = {"Bullish": "up", "Bearish": "down", "None": "off"}[vote]
    column.metric(
        label,
        value,
        vote if vote != "None" else "Neutral",
        delta_color=delta_color,
        delta_arrow=delta_arrow,
        help=help_text,
    )


def render_indicator_breakdown(ticker):
    """
    Show each of the 4 directional indicators' current vote, plus how close
    RSI and Bollinger %B are to their bullish/bearish thresholds. Pure local
    computation (no LLM), so this can run just from picking a ticker.
    """
    rsi_data = get_rsi_signal(ticker)
    macd_data = get_macd_signal(ticker)
    crossover_data = get_crossover_signal(ticker)
    bollinger_data = get_bollinger_bands(ticker)
    current_price = load_price_data(ticker)["Close"].iloc[-1]

    rsi_vote = vote_from_signal(rsi_data["Signal"])
    macd_vote = vote_from_signal(macd_data["Signal"])
    ema_vote = vote_from_signal(crossover_data["Trend"])
    bollinger_vote = vote_from_signal(bollinger_data["Signal"])

    bullish_count = [rsi_vote, macd_vote, ema_vote, bollinger_vote].count("Bullish")
    bearish_count = [rsi_vote, macd_vote, ema_vote, bollinger_vote].count("Bearish")

    if bullish_count >= 3:
        st.success(f":material/check_circle: {bullish_count} of 4 bullish — meets the confluence threshold.")
    elif bearish_count >= 3:
        st.success(f":material/check_circle: {bearish_count} of 4 bearish — meets the confluence threshold.")
    else:
        st.caption(f"{bullish_count} bullish, {bearish_count} bearish — needs 3 of 4 to flag.")

    col1, col2, col3, col4 = st.columns(4)

    render_vote_metric(col1, "RSI", f"{rsi_data['RSI']:.1f}", rsi_vote)
    col1.progress(
        min(1.0, max(0.0, rsi_data["RSI"] / 100)), text="Bullish <35 · Bearish >65"
    )

    render_vote_metric(
        col2,
        "MACD histogram",
        f"{macd_data['Histogram']:.2f}",
        macd_vote,
        help_text=f"MACD {macd_data['MACD']:.2f} vs Signal Line {macd_data['Signal Line']:.2f}",
    )

    render_vote_metric(
        col3,
        "EMA trend",
        crossover_data["Trend"],
        ema_vote,
        help_text=f"Cross: {crossover_data['ApproachingCross']}",
    )

    percent_b = (current_price - bollinger_data["Lower"]) / (
        bollinger_data["Upper"] - bollinger_data["Lower"]
    )
    render_vote_metric(col4, "Bollinger %B", f"{percent_b:.0%}", bollinger_vote)
    col4.progress(
        min(1.0, max(0.0, percent_b)), text="Bullish <0% · Bearish >100%"
    )


def render_signal(signal):
    if signal.flagged:
        icon = ":material/trending_up:" if signal.direction == "call" else ":material/trending_down:"
        st.success(f"{icon} **{signal.ticker}** flagged — {signal.direction}")
    else:
        st.info(f"**{signal.ticker}** — no confluence, staying silent.")
    st.write(signal.reasoning)
    if signal.supporting_signals:
        st.caption("Supporting signals: " + ", ".join(signal.supporting_signals))


st.header("Live signal")

selected_ticker = st.segmented_control("Ticker", TICKERS, default=TICKERS[0])

if selected_ticker:
    overlay = st.segmented_control("Chart overlay", CHART_OVERLAYS, default="None")
    theme_colors = PALETTE["dark"] if st.context.theme.type == "dark" else PALETTE["light"]
    chart, chart_caption = build_price_chart(selected_ticker, overlay, theme_colors)
    st.altair_chart(chart, width="stretch", theme=None)
    if chart_caption:
        st.caption(chart_caption)

    st.subheader("Indicator votes")
    render_indicator_breakdown(selected_ticker)

analyze_col, analyze_all_col = st.columns(2)
with analyze_col:
    analyze_one = st.button(
        "Analyze selected ticker", disabled=selected_ticker is None, width="stretch"
    )
with analyze_all_col:
    analyze_all = st.button("Analyze all tickers", width="stretch")

if analyze_one and selected_ticker:
    with st.spinner(f"Analyzing {selected_ticker}..."):
        signal = cached_analyze_ticker(selected_ticker, data_file_mtime(selected_ticker))
    render_signal(signal)

if analyze_all:
    with st.status("Analyzing all tickers...", expanded=True) as status:
        for ticker in TICKERS:
            st.write(f"Analyzing {ticker}...")
            signal = cached_analyze_ticker(ticker, data_file_mtime(ticker))
            render_signal(signal)
        status.update(label="Done", state="complete")

st.divider()

st.header("Backtest results")
st.caption(
    "Evaluates the same confluence rule in code (no LLM) over full price "
    "history. No claim this strategy works — see the numbers below."
)

backtest_results = cached_backtest_all(max(data_file_mtime(t) for t in TICKERS))

signals_col, win_rate_col, return_col = st.columns(3)
signals_col.metric("Total signals", backtest_results["total_signals"])
win_rate_col.metric(
    "Overall win rate",
    f"{backtest_results['overall_win_rate']:.1%}"
    if backtest_results["overall_win_rate"] is not None
    else "N/A",
)
return_col.metric(
    "Overall avg return",
    f"{backtest_results['overall_avg_return']:.1%}"
    if backtest_results["overall_avg_return"] is not None
    else "N/A",
)

per_ticker_df = pd.DataFrame(backtest_results["per_ticker"])
st.dataframe(per_ticker_df, width="stretch")
