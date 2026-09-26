"""Feature engineering for the market movement prediction model."""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_5d",
    "volatility_20d",
    "volume_zscore_20d",
    "high_low_range",
    "rsi_14",
    "relative_return_spy_5d",
    "relative_return_spy_20d",
]


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index for a price series."""
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_features(
    stock_data: pd.DataFrame,
    spy_data: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """Build market features used by the prediction model."""
    df = stock_data.copy()

    # Returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # Volatility
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Volume relative to its recent history
    volume_mean = df["Volume"].rolling(20).mean()
    volume_std = df["Volume"].rolling(20).std()

    df["volume_zscore_20d"] = (
        (df["Volume"] - volume_mean) / volume_std
    )

    # Daily trading range
    df["high_low_range"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    # Momentum
    df["rsi_14"] = calculate_rsi(df["Close"])

    # Align SPY to the stock's trading dates
    spy_close = spy_data["Close"].reindex(df.index)

    spy_return_5d = spy_close.pct_change(5)
    spy_return_20d = spy_close.pct_change(20)

    df["relative_return_spy_5d"] = (
        df["return_5d"] - spy_return_5d
    )

    df["relative_return_spy_20d"] = (
        df["return_20d"] - spy_return_20d
    )

    if include_target:
        future_return_5d = (
            df["Close"].shift(-5) / df["Close"] - 1
        ).abs()

        expected_move_5d = (
            df["volatility_20d"] * np.sqrt(5)
        )

        df["target"] = (
            future_return_5d > 1.5 * expected_move_5d
        ).astype(int)

        # Last five rows have no known 5-day future return
        df = df.iloc[:-5]

    return df.dropna()
