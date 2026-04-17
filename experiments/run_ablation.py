import pandas as pd

from common import (
    RESULTS_DIR,
    ensure_results_dir,
    evaluate_model,
    prepare_design_matrices,
    save_json,
    split_chronologically_by_season,
    train_xgboost_classifier,
)
from feature_sets import build_expanded_feature_frame, get_expanded_numeric_columns


FEATURE_FAMILIES = {
    "without_elo_family": [
        "home_elo",
        "away_elo",
        "elo_difference",
    ],
    "without_form_family": [
        "home_avg_points_last_5",
        "away_avg_points_last_5",
        "home_goals_scored_avg_last_5",
        "away_goals_scored_avg_last_5",
        "home_goals_conceded_avg_last_5",
        "away_goals_conceded_avg_last_5",
    ],
    "without_xg_family": [
        "home_xg_avg_last_5",
        "away_xg_avg_last_5",
        "home_graph_xg_diff_avg_last_5",
        "away_graph_xg_diff_avg_last_5",
        "home_xg_conceded_avg_last_5",
        "away_xg_conceded_avg_last_5",
        "home_home_xg_avg_last_5",
        "away_away_xg_avg_last_5",
        "head_to_head_home_xg_diff_avg_last_3",
    ],
    "without_rest_days": [
        "home_rest_days",
        "away_rest_days",
    ],
    "without_venue_strength": [
        "home_home_points_avg_last_5",
        "away_away_points_avg_last_5",
        "home_home_xg_avg_last_5",
        "away_away_xg_avg_last_5",
    ],
    "without_head_to_head": [
        "head_to_head_home_points_avg_last_3",
        "head_to_head_home_xg_diff_avg_last_3",
        "head_to_head_matches_seen",
    ],
    "without_missing_xg_signals": [
        "home_missing_xg_rate_last_5",
        "away_missing_xg_rate_last_5",
    ],
}


def evaluate_columns(name, df, numeric_columns):
    """Evaluate a single ablation setting."""
    train_df, validation_df, test_df = split_chronologically_by_season(df)
    _, X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_design_matrices(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        numeric_columns=numeric_columns,
        impute_numeric=True,
        add_missing_indicators=True,
    )

    model = train_xgboost_classifier(X_train, y_train)
    _, validation_metrics = evaluate_model(model, X_validation, y_validation)
    _, test_metrics = evaluate_model(model, X_test, y_test)

    return {
        "ablation": name,
        "remaining_numeric_features": len(numeric_columns),
        "final_feature_count": X_train.shape[1],
        **{f"validation_{metric_name}": metric_value for metric_name, metric_value in validation_metrics.items()},
        **{f"test_{metric_name}": metric_value for metric_name, metric_value in test_metrics.items()},
    }


def run_experiment():
    """Run the planned ablation study."""
    ensure_results_dir()

    df = build_expanded_feature_frame()
    full_numeric_columns = get_expanded_numeric_columns()

    results = [evaluate_columns("full_feature_set", df, full_numeric_columns)]
    for ablation_name, removed_columns in FEATURE_FAMILIES.items():
        filtered_columns = [column for column in full_numeric_columns if column not in removed_columns]
        results.append(evaluate_columns(ablation_name, df, filtered_columns))

    results_df = pd.DataFrame(results)
    baseline_test_macro_f1 = float(
        results_df.loc[results_df["ablation"] == "full_feature_set", "test_macro_f1"].iloc[0]
    )
    results_df["delta_test_macro_f1"] = results_df["test_macro_f1"] - baseline_test_macro_f1
    results_df = results_df.sort_values(["test_macro_f1", "test_log_loss"], ascending=[False, True])
    results_df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    largest_drop_row = results_df[results_df["ablation"] != "full_feature_set"].sort_values(
        "delta_test_macro_f1"
    ).iloc[0]
    summary = {
        "baseline_test_macro_f1": baseline_test_macro_f1,
        "largest_drop_ablation": largest_drop_row["ablation"],
        "largest_drop_delta_test_macro_f1": largest_drop_row["delta_test_macro_f1"],
    }
    save_json(RESULTS_DIR / "ablation_summary.json", summary)
    return summary


def main():
    summary = run_experiment()
    print("Ablation study complete.")
    print(f"Baseline test macro F1: {summary['baseline_test_macro_f1']:.4f}")
    print(f"Largest drop: {summary['largest_drop_ablation']}")
    print(
        "Delta test macro F1: "
        f"{summary['largest_drop_delta_test_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
