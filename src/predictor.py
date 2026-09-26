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


def get_feature_contributions(
    model: xgb.XGBClassifier,
    features: pd.Series,
) -> list[dict]:
    """Return each feature's contribution to the current prediction."""
    feature_frame = features.to_frame().T

    dmatrix = xgb.DMatrix(
        feature_frame,
        feature_names=FEATURE_COLUMNS,
    )

    contributions = model.get_booster().predict(
        dmatrix,
        pred_contribs=True,
    )[0]

    # The final value is the model bias/intercept, not a feature.
    feature_contributions = contributions[:-1]

    results = [
        {
            "feature": feature,
            "contribution": float(contribution),
            "value": float(features[feature]),
        }
        for feature, contribution in zip(
            FEATURE_COLUMNS,
            feature_contributions,
        )
    ]

    return sorted(
        results,
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )


def predict_abnormal_move(
    ticker: str,
) -> tuple[float, pd.Series, list[dict]]:
    """Return the risk score, latest features, and model contributions."""
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

    contributions = get_feature_contributions(
        model,
        latest_features,
    )

    return float(score), latest_features, contributions
