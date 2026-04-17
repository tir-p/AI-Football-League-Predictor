import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from config import (
    FEATURE_COLUMNS_FILE,
    FEATURES_FILE,
    MODEL_FILE,
    MODEL_METRICS_FILE,
    PREPROCESSOR_FILE,
    TEST_FILE,
    TRAIN_FILE,
)
from preprocessing import fit_preprocessor, transform_features


def split_data_by_season(df):
    """Create chronological backtest splits from played matches only."""
    played_df = df[df["match_result"].notna()].copy()
    train_df = played_df[(played_df["season"] >= 2014) & (played_df["season"] <= 2022)].copy()
    validation_df = played_df[played_df["season"] == 2023].copy()
    test_df = played_df[played_df["season"] == 2024].copy()
    return train_df, validation_df, test_df


def get_final_training_data(df):
    """Return all played matches available for production training."""
    return df[df["match_result"].notna() & (df["season"] <= 2024)].copy()


def separate_features_and_target(df, preprocessor):
    """Convert raw rows into the encoded and scaled model matrix."""
    X = transform_features(df, preprocessor)
    y = df["match_result"].astype(int)
    return X, y


def train_xgboost_model(X_train, y_train):
    """Train a simple XGBoost classifier for 3 match outcomes."""
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=250,
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

def evaluate_home_win_baseline(y_true):
    """Baseline: always predict home win (class 0)."""
    baseline_predictions = [0] * len(y_true)

    metrics = {
        "accuracy": accuracy_score(y_true, baseline_predictions),
        "macro_f1": f1_score(y_true, baseline_predictions, average="macro"),
    }
    return metrics

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


def save_transformed_matrix(output_path, X, y):
    """Persist a transformed feature matrix for inspection."""
    matrix_df = X.copy()
    matrix_df["match_result"] = y.to_numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(output_path, index=False)


def main():
    df = pd.read_csv(FEATURES_FILE)
    df = df.sort_values("date").reset_index(drop=True)

    train_df, validation_df, test_df = split_data_by_season(df)

    backtest_preprocessor = fit_preprocessor(train_df)
    X_train, y_train = separate_features_and_target(train_df, backtest_preprocessor)
    X_validation, y_validation = separate_features_and_target(validation_df, backtest_preprocessor)
    X_test, y_test = separate_features_and_target(test_df, backtest_preprocessor)

    validation_baseline_metrics = evaluate_home_win_baseline(y_validation)
    test_baseline_metrics = evaluate_home_win_baseline(y_test)

    evaluation_model = train_xgboost_model(X_train, y_train)

    _, validation_probabilities, validation_metrics = evaluate_split(
        evaluation_model, X_validation, y_validation
    )
    test_predictions, test_probabilities, test_metrics = evaluate_split(
    evaluation_model, X_test, y_test)

    validation_predictions_df = add_probability_columns(validation_df, validation_probabilities)
    test_predictions_df = add_probability_columns(test_df, test_probabilities)

    final_train_df = get_final_training_data(df)
    final_preprocessor = fit_preprocessor(final_train_df)
    X_final_train, y_final_train = separate_features_and_target(final_train_df, final_preprocessor)
    final_model = train_xgboost_model(X_final_train, y_final_train)

    metrics = {
        "backtest_train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "final_train_rows": len(final_train_df),
        "final_train_end_season": 2024,
        "categorical_feature_columns": final_preprocessor["categorical_columns"],
        "scaled_numeric_feature_columns": final_preprocessor["numeric_columns"],
        "feature_columns": final_preprocessor["feature_columns"],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_home_win_baseline_metrics": validation_baseline_metrics,
        "test_home_win_baseline_metrics": test_baseline_metrics,
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_FILE)
    joblib.dump(final_preprocessor, PREPROCESSOR_FILE)
    joblib.dump(final_preprocessor["feature_columns"], FEATURE_COLUMNS_FILE)
    save_metrics(metrics)
    save_transformed_matrix(TRAIN_FILE, X_final_train, y_final_train)
    save_transformed_matrix(TEST_FILE, X_test, y_test)

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Preprocessor saved to: {PREPROCESSOR_FILE}")
    print(f"Metrics saved to: {MODEL_METRICS_FILE}")
    print("Backtest feature matrix includes encoded team IDs and scaled numeric features.")
    print("Backtest validation season: 2023")
    print(f"Validation accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"Validation macro F1: {validation_metrics['macro_f1']:.4f}")
    print(f"Validation log loss: {validation_metrics['log_loss']:.4f}")
    print("Backtest test season: 2024")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test log loss: {test_metrics['log_loss']:.4f}")
    print(f"Final production model trained on {len(final_train_df)} played rows through season 2024.")
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
    print()
    print("Home-win baseline comparison")
    print(f"Validation baseline accuracy: {validation_baseline_metrics['accuracy']:.4f}")
    print(f"Validation baseline macro F1: {validation_baseline_metrics['macro_f1']:.4f}")
    print(f"Test baseline accuracy: {test_baseline_metrics['accuracy']:.4f}")
    print(f"Test baseline macro F1: {test_baseline_metrics['macro_f1']:.4f}")
    print()
    print("Generating confusion matrix for test set (2024)...")

    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1, 2])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Home Win", "Draw", "Away Win"]
    )

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Test Set (2024)")
    plt.savefig(MODEL_FILE.parent / "confusion_matrix_test.png")
    plt.show()

    print()
    print("Generating feature importance plot...")

    feature_names = backtest_preprocessor["feature_columns"]
    importances = evaluation_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print()
    print("Top 10 most important features:")
    print(importance_df.head(10))

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"].head(10)[::-1], importance_df["importance"].head(10)[::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances - XGBoost")
    plt.tight_layout()
    plt.savefig(MODEL_FILE.parent / "feature_importance_top10.png")
    plt.show()

if __name__ == "__main__":
    main()
