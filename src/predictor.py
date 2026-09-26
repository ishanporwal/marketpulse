"""Inference utilities for the abnormal movement model."""

import pandas as pd
import xgboost as xgb

from src.features import FEATURE_COLUMNS, build_features
from src.market_data import get_stock_history


MODEL_PATH = "models/xgb_volatility.json"


def load_model() -> xgb.XGBClassifier:
    """Load the trained XGBoost model."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    return model


def predict_abnormal_move(ticker: str) -> tuple[float, pd.Series]:
    """Return the abnormal-move risk score and latest market features."""
    stock_data = get_stock_history(ticker, period="6mo")
    spy_data = get_stock_history("SPY", period="6mo")

    features = build_features(
        stock_data,
        spy_data,
        include_target=False,
    )

    if features.empty:
        raise ValueError(
            f"Not enough historical data to generate features for {ticker}"
        )

    latest_features = features[FEATURE_COLUMNS].iloc[-1]

    model = load_model()

    score = model.predict_proba(
        latest_features.to_frame().T
    )[0, 1]

    return float(score), latest_features
