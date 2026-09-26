"""Train and evaluate the abnormal market movement classifier."""

import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import FEATURE_COLUMNS


DATA_PATH = "training/training_data.csv"
MODEL_PATH = "models/xgb_volatility.json"


def load_dataset() -> pd.DataFrame:
    """Load the generated training dataset."""
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], utc=True)

    return data.sort_values("date")


def split_dataset(data: pd.DataFrame):
    """Split the dataset chronologically into training and test sets."""
    dates = sorted(data["date"].unique())

    split_index = int(len(dates) * 0.8)

    # Leave a five-trading-day gap so training labels cannot
    # use price movement from the test period.
    train_end_index = split_index - 5

    train_end = dates[train_end_index]
    test_start = dates[split_index]

    train = data[data["date"] <= train_end]
    test = data[data["date"] >= test_start]

    return train, test


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBClassifier:
    """Train the XGBoost movement classifier."""
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print(f"Class weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    return model

def evaluate_model(
    model: xgb.XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Print classification metrics for the test period."""
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.3f}")
    print(
        f"PR-AUC:    "
        f"{average_precision_score(y_test, probabilities):.3f}"
    )
    print(f"Precision: {precision_score(y_test, predictions):.3f}")
    print(f"Recall:    {recall_score(y_test, predictions):.3f}")
    print(f"F1:        {f1_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    data = load_dataset()
    train, test = split_dataset(data)

    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"]

    x_test = test[FEATURE_COLUMNS]
    y_test = test["target"]

    print(f"Training rows: {len(train)}")
    print(f"Test rows:     {len(test)}")
    print(f"Training through: {train['date'].max().date()}")
    print(f"Testing from:     {test['date'].min().date()}")
    print()
    print(f"Train target rate: {y_train.mean():.3f}")
    print(f"Test target rate:  {y_test.mean():.3f}")
    print()

    model = train_model(x_train, y_train)

    evaluate_model(model, x_test, y_test)

    model.save_model(MODEL_PATH)
    print()
    print(f"Saved model to {MODEL_PATH}")
