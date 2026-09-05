"""Unit tests for indicators.py — verify each formula against hand-computed
values or known edge cases, using small synthetic price series."""

import operator

import numpy as np
import pandas as pd
import pytest

from indicators import bollinger_bands, ema, macd, moving_average_crossover, rsi, volume_anomaly


def test_ema_matches_hand_calculation():
    """
    EMA (adjust=False) formula: ema[0] = price[0], then
    ema[i] = price[i] * alpha + ema[i-1] * (1 - alpha), where alpha = 2/(span+1).

    For prices = [10, 12, 14] with span=2 (alpha = 2/3):
      ema[0] = 10
      ema[1] = 12*(2/3) + 10*(1/3) = 34/3  = 11.333...
      ema[2] = 14*(2/3) + (34/3)*(1/3) = 118/9 = 13.111...
    """
    prices = pd.Series([10, 12, 14])
    result = ema(prices, span=2)

    assert result.iloc[0] == pytest.approx(10)
    assert result.iloc[1] == pytest.approx(34 / 3)
    assert result.iloc[2] == pytest.approx(118 / 9)


@pytest.mark.parametrize(
    "prices, expected_rsi",
    [
        pytest.param(pd.Series([1, 2, 3, 4, 5, 6, 7]), 100, id="strictly_increasing"),
        pytest.param(pd.Series([7, 6, 5, 4, 3, 2, 1]), 0, id="strictly_decreasing"),
        pytest.param(pd.Series([5, 5, 5, 5, 5]), 50, id="flat"),
    ],
)
def test_rsi_extreme_and_flat_cases(prices, expected_rsi):
    """
    - No losses at all -> avg_loss=0, avg_gain>0 -> RSI=100.
    - No gains at all -> avg_gain=0, avg_loss>0 -> RSI=0.
    - No movement at all -> avg_gain=0 and avg_loss=0 -> the flat-price
      special case, defined as neutral (50).
    """
    result = rsi(prices, period=3)

    assert result.iloc[-1] == pytest.approx(expected_rsi)


def test_macd_histogram_equals_macd_minus_signal_line():
    """The Histogram column should always equal MACD - Signal Line exactly,
    regardless of the input data — this is true by definition."""
    prices = pd.Series(np.linspace(100, 150, 60))
    result = macd(prices)

    pd.testing.assert_series_equal(
        result["Histogram"], result["MACD"] - result["Signal Line"], check_names=False
    )


def test_macd_line_equals_fast_ema_minus_slow_ema():
    """MACD's own fast/slow EMA lines should match calling ema() directly
    with the same periods — catches a wrong span or a swapped fast/slow."""
    prices = pd.Series(np.linspace(100, 150, 60))
    result = macd(prices, fast_period=12, slow_period=26)

    expected_macd_line = ema(prices, span=12) - ema(prices, span=26)
    pd.testing.assert_series_equal(result["MACD"], expected_macd_line, check_names=False)


def test_bollinger_bands_match_hand_rolling_calculation():
    """Upper/Lower should equal the SMA +/- num_std * rolling std, computed
    independently here with plain pandas as the reference."""
    prices = pd.Series([10, 12, 11, 13, 12, 14, 13, 15, 14, 16], dtype=float)
    period, num_std = 5, 2
    result = bollinger_bands(prices, period=period, num_std=num_std)

    expected_middle = prices.rolling(window=period).mean()
    expected_std = prices.rolling(window=period).std()
    expected_upper = expected_middle + num_std * expected_std
    expected_lower = expected_middle - num_std * expected_std

    pd.testing.assert_series_equal(result["Middle"], expected_middle, check_names=False)
    pd.testing.assert_series_equal(result["Upper"], expected_upper, check_names=False)
    pd.testing.assert_series_equal(result["Lower"], expected_lower, check_names=False)


def test_bollinger_bands_flags_overbought_and_oversold():
    """A price forced above the upper band, or below the lower band, should
    be flagged accordingly.

    Note: the rolling window includes the day being judged, so a spike needs
    to be large relative to the window size to outrun its own effect on the
    mean/std it's being measured against — a small period or a mild spike
    can get diluted into "Neutral" even though it's a real outlier.
    """
    prices = pd.Series([10.0] * 19 + [200.0] + [10.0] * 19 + [-180.0])
    result = bollinger_bands(prices, period=20, num_std=2)

    assert result["Signal"].iloc[19] == "Overbought"  # the spike to 200
    assert result["Signal"].iloc[-1] == "Oversold"  # the drop to -180


@pytest.mark.parametrize(
    "start, end, expected_trend, comparator",
    [
        pytest.param(100, 400, "Bullish", operator.gt, id="rising"),
        pytest.param(400, 100, "Bearish", operator.lt, id="falling"),
    ],
)
def test_moving_average_crossover_trend(start, end, expected_trend, comparator):
    """A long, steadily trending price series should eventually settle into
    the matching Trend once all 3 EMAs have had time to reflect it."""
    prices = pd.Series(np.linspace(start, end, 300))
    result = moving_average_crossover(prices)

    assert result["Trend"].iloc[-1] == expected_trend
    assert comparator(result["EMA_short"].iloc[-1], result["EMA_mid"].iloc[-1])
    assert comparator(result["EMA_mid"].iloc[-1], result["EMA_long"].iloc[-1])


def test_volume_anomaly_flags_high_volume():
    """avg_volume excludes the current day (shift(1)), so a spike doesn't
    inflate its own baseline the way Bollinger's does — a simple 5x spike
    over a flat history is enough to trigger the threshold."""
    volume = pd.Series([1000.0] * 20 + [5000.0])
    result = volume_anomaly(volume, period=20, threshold=2)

    assert result["Signal"].iloc[-1] == "High Volume"
    assert result["Ratio"].iloc[-1] == pytest.approx(5.0)


def test_volume_anomaly_is_normal_for_steady_volume():
    volume = pd.Series([1000.0] * 25)
    result = volume_anomaly(volume, period=20, threshold=2)

    assert result["Signal"].iloc[-1] == "Normal"
