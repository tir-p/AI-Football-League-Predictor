import pandas as pd

from common import (
    RESULTS_DIR,
    ensure_results_dir,
    evaluate_model,
    get_balanced_sample_weights,
    prepare_design_matrices,
    save_json,
    train_xgboost_classifier,
)
from feature_sets import build_expanded_feature_frame, get_expanded_numeric_columns


MIN_TRAIN_SEASONS = 5


def run_experiment():
    """Run rolling out-of-sample validation on later seasons."""
    ensure_results_dir()

    df = build_expanded_feature_frame()
    played_df = df[df["match_result"].notna()].copy()
    numeric_columns = get_expanded_numeric_columns()

    season_rows = []
    for target_season in sorted(played_df["season"].unique()):
        train_df = played_df[played_df["season"] < target_season].copy()
        test_df = played_df[played_df["season"] == target_season].copy()

        if test_df.empty or train_df["season"].nunique() < MIN_TRAIN_SEASONS:
            continue

        _, X_train, _, X_test, y_train, _, y_test = prepare_design_matrices(
            train_df=train_df,
            validation_df=test_df,
            test_df=test_df,
            numeric_columns=numeric_columns,
            impute_numeric=True,
            add_missing_indicators=True,
        )

        sample_weights = get_balanced_sample_weights(y_train)
        model = train_xgboost_classifier(X_train, y_train, sample_weight=sample_weights)
        _, test_metrics = evaluate_model(model, X_test, y_test)

        season_rows.append(
            {
                "season": int(target_season),
                "train_start_season": int(train_df["season"].min()),
                "train_end_season": int(train_df["season"].max()),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                **test_metrics,
            }
        )

    results_df = pd.DataFrame(season_rows).sort_values("season")
    results_df.to_csv(RESULTS_DIR / "out_of_sample_validation_results.csv", index=False)

    if results_df.empty:
        summary = {
            "evaluated_seasons": [],
            "mean_accuracy": None,
            "mean_macro_f1": None,
            "latest_season": None,
        }
        save_json(RESULTS_DIR / "out_of_sample_validation_summary.json", summary)
        return summary

    summary = {
        "evaluated_seasons": results_df["season"].tolist(),
        "mean_accuracy": float(results_df["accuracy"].mean()),
        "mean_macro_f1": float(results_df["macro_f1"].mean()),
        "latest_season": int(results_df["season"].max()),
    }
    save_json(RESULTS_DIR / "out_of_sample_validation_summary.json", summary)
    return summary


def main():
    summary = run_experiment()
    print("Out-of-sample validation complete.")
    print(f"Evaluated seasons: {summary['evaluated_seasons']}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.4f}")
    print(f"Mean macro F1: {summary['mean_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
