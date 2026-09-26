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
    """Split data chronologically into train, validation, and test sets."""
    dates = sorted(data["date"].unique())

    train_split = int(len(dates) * 0.70)
    validation_split = int(len(dates) * 0.85)

    gap = 5

    train_dates = dates[: train_split - gap]
    validation_dates = dates[train_split : validation_split - gap]
    test_dates = dates[validation_split:]

    train = data[data["date"].isin(train_dates)]
    validation = data[data["date"].isin(validation_dates)]
    test = data[data["date"].isin(test_dates)]

    return train, validation, test

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

    train, validation, test = split_dataset(data)

    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"]

    x_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["target"]

    x_test = test[FEATURE_COLUMNS]
    y_test = test["target"]

    print(f"Training rows:   {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Test rows:       {len(test)}")

    print()
    print(
        f"Training:   "
        f"{train['date'].min().date()} -> "
        f"{train['date'].max().date()}"
    )
    print(
        f"Validation: "
        f"{validation['date'].min().date()} -> "
        f"{validation['date'].max().date()}"
    )
    print(
        f"Test:       "
        f"{test['date'].min().date()} -> "
        f"{test['date'].max().date()}"
    )

    print()
    print(f"Train target rate:      {y_train.mean():.3f}")
    print(f"Validation target rate: {y_validation.mean():.3f}")
    print(f"Test target rate:       {y_test.mean():.3f}")

    # First train only on the training period.
    model = train_model(x_train, y_train)

    print("\nValidation metrics:")
    evaluate_model(
        model,
        x_validation,
        y_validation,
    )

    # Hyperparameters are now frozen. Retrain using all non-test data.
    final_training_data = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    x_final_train = final_training_data[FEATURE_COLUMNS]
    y_final_train = final_training_data["target"]

    print("\nTraining final model on train + validation...")
    final_model = train_model(
        x_final_train,
        y_final_train,
    )

    print("\nFinal test metrics:")
    evaluate_model(
        final_model,
        x_test,
        y_test,
    )

    final_model.save_model(MODEL_PATH)

    print()
    print(f"Saved final model to {MODEL_PATH}")
