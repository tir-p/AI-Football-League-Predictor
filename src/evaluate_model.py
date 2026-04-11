import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from config import EVALUATION_FILE, FEATURE_COLUMNS_FILE, FEATURES_FILE, MODEL_FILE, TEST_PREDICTIONS_FILE


RESULT_LABELS = {
    0: "Home Win",
    1: "Draw",
    2: "Away Win",
}


def recreate_test_split(df):
    """Return only the 2024 test season."""
    return df[df["season"] == 2024].copy()


def evaluate_predictions(y_true, y_pred, y_prob):
    """Compute the required evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1, 2]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    return metrics


def create_predictions_dataframe(test_df, predictions, probabilities):
    """Create a readable file with actual results, predictions, and probabilities."""
    test_df = test_df.reset_index(drop=True)

    prediction_df = pd.DataFrame(
        {
            "date": test_df["date"],
            "home_team": test_df["home_team"],
            "away_team": test_df["away_team"],
            "actual_result": test_df["match_result"].map(RESULT_LABELS),
            "predicted_result": pd.Series(predictions).map(RESULT_LABELS),
            "predicted_home_win_probability": probabilities[:, 0],
            "predicted_draw_probability": probabilities[:, 1],
            "predicted_away_win_probability": probabilities[:, 2],
        }
    )
    return prediction_df


def main():
    df = pd.read_csv(FEATURES_FILE)
    df = df.sort_values("date").reset_index(drop=True)
    test_df = recreate_test_split(df)

    feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
    model = joblib.load(MODEL_FILE)

    X_test = test_df[feature_columns]
    y_test = test_df["match_result"].astype(int)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    metrics = evaluate_predictions(y_test, predictions, probabilities)

    predictions_df = create_predictions_dataframe(test_df, predictions, probabilities)

    lines = [
        "Football Match Prediction Evaluation",
        "=" * 40,
        "Test season: 2024",
        f"Number of matches: {len(test_df)}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Log loss: {metrics['log_loss']:.4f}",
        f"Confusion matrix: {metrics['confusion_matrix']}",
    ]

    EVALUATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    predictions_df.to_csv(TEST_PREDICTIONS_FILE, index=False)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Evaluation report saved to: {EVALUATION_FILE}")
    print(f"Detailed predictions saved to: {TEST_PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
