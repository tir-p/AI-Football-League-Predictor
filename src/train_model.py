import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier

from config import FEATURE_COLUMNS_FILE, FEATURES_FILE, MODEL_FILE, MODEL_METRICS_FILE


def get_feature_columns():
    """Return the model input columns."""
    return [
        "home_avg_points_last_5",
        "away_avg_points_last_5",
        "home_goals_scored_avg_last_5",
        "away_goals_scored_avg_last_5",
        "home_xg_avg_last_5",
        "away_xg_avg_last_5",
        "home_elo",
        "away_elo",
        "elo_difference",
        "home_rest_days",
        "away_rest_days",
    ]


def split_data_by_season(df):
    """Split the data into train, validation, and test sets by season."""
    train_df = df[(df["season"] >= 2014) & (df["season"] <= 2021)].copy()
    validation_df = df[df["season"] == 2022].copy()
    test_df = df[df["season"] == 2023].copy()
    return train_df, validation_df, test_df


def separate_features_and_target(df, feature_columns):
    """Separate model inputs and target values."""
    X = df[feature_columns]
    y = df["match_result"]
    return X, y


def train_xgboost_model(X_train, y_train):
    """Train a simple XGBoost classifier for 3 match outcomes."""
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_split(model, X, y):
    """Calculate predictions, probabilities, and simple metrics."""
    predicted_classes = model.predict(X)
    predicted_probabilities = model.predict_proba(X)

    metrics = {
        "accuracy": accuracy_score(y, predicted_classes),
        "macro_f1": f1_score(y, predicted_classes, average="macro"),
        "log_loss": log_loss(y, predicted_probabilities, labels=[0, 1, 2]),
    }
    return predicted_classes, predicted_probabilities, metrics


def save_metrics(metrics):
    """Save the model metrics to JSON."""
    MODEL_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def add_probability_columns(df, probabilities):
    """Add class probability columns to a copy of the data frame."""
    result_df = df.copy()
    result_df["prob_home_win"] = probabilities[:, 0]
    result_df["prob_draw"] = probabilities[:, 1]
    result_df["prob_away_win"] = probabilities[:, 2]
    return result_df


def main():
    df = pd.read_csv(FEATURES_FILE)
    df = df.sort_values("date").reset_index(drop=True)
    feature_columns = [
        *get_feature_columns(),
    ]

    train_df, validation_df, test_df = split_data_by_season(df)
    X_train, y_train = separate_features_and_target(train_df, feature_columns)
    X_validation, y_validation = separate_features_and_target(validation_df, feature_columns)
    X_test, y_test = separate_features_and_target(test_df, feature_columns)

    model = train_xgboost_model(X_train, y_train)

    _, validation_probabilities, validation_metrics = evaluate_split(model, X_validation, y_validation)
    _, test_probabilities, test_metrics = evaluate_split(model, X_test, y_test)

    validation_predictions_df = add_probability_columns(validation_df, validation_probabilities)
    test_predictions_df = add_probability_columns(test_df, test_probabilities)

    metrics = {
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "feature_columns": feature_columns,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(feature_columns, FEATURE_COLUMNS_FILE)
    save_metrics(metrics)

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Metrics saved to: {MODEL_METRICS_FILE}")
    print(f"Validation accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"Validation macro F1: {validation_metrics['macro_f1']:.4f}")
    print(f"Validation log loss: {validation_metrics['log_loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test log loss: {test_metrics['log_loss']:.4f}")
    print()
    print("Sample predicted probabilities for validation data:")
    print(
        validation_predictions_df[
            ["date", "home_team", "away_team", "prob_home_win", "prob_draw", "prob_away_win"]
        ].head(5)
    )
    print()
    print("Sample predicted probabilities for test data:")
    print(
        test_predictions_df[
            ["date", "home_team", "away_team", "prob_home_win", "prob_draw", "prob_away_win"]
        ].head(5)
    )


if __name__ == "__main__":
    main()
