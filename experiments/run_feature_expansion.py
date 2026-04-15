import pandas as pd

from common import (
    RESULTS_DIR,
    ensure_results_dir,
    evaluate_model,
    get_base_numeric_columns,
    load_base_features,
    prepare_design_matrices,
    save_json,
    split_chronologically_by_season,
    train_xgboost_classifier,
)
from feature_sets import ADDITIONAL_NUMERIC_COLUMNS, build_expanded_feature_frame


def evaluate_configuration(name, df, numeric_columns, impute_numeric, add_missing_indicators):
    """Train and evaluate one feature-set configuration."""
    train_df, validation_df, test_df = split_chronologically_by_season(df)
    _, X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_design_matrices(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        numeric_columns=numeric_columns,
        impute_numeric=impute_numeric,
        add_missing_indicators=add_missing_indicators,
    )

    model = train_xgboost_classifier(X_train, y_train)
    _, validation_metrics = evaluate_model(model, X_validation, y_validation)
    _, test_metrics = evaluate_model(model, X_test, y_test)

    return {
        "configuration": name,
        "base_numeric_features": len(get_base_numeric_columns()),
        "extra_numeric_features": max(len(numeric_columns) - len(get_base_numeric_columns()), 0),
        "final_feature_count": X_train.shape[1],
        "impute_numeric": impute_numeric,
        "add_missing_indicators": add_missing_indicators,
        **{f"validation_{metric_name}": metric_value for metric_name, metric_value in validation_metrics.items()},
        **{f"test_{metric_name}": metric_value for metric_name, metric_value in test_metrics.items()},
    }


def run_experiment():
    """Run the planned feature-expansion experiment."""
    ensure_results_dir()

    base_numeric_columns = get_base_numeric_columns()
    expanded_numeric_columns = base_numeric_columns + list(ADDITIONAL_NUMERIC_COLUMNS)
    expanded_df = build_expanded_feature_frame()

    results = [
        evaluate_configuration(
            name="baseline_current_features",
            df=load_base_features(),
            numeric_columns=base_numeric_columns,
            impute_numeric=False,
            add_missing_indicators=False,
        ),
        evaluate_configuration(
            name="expanded_context_features",
            df=expanded_df,
            numeric_columns=expanded_numeric_columns,
            impute_numeric=False,
            add_missing_indicators=False,
        ),
        evaluate_configuration(
            name="expanded_context_with_missing_value_handling",
            df=expanded_df,
            numeric_columns=expanded_numeric_columns,
            impute_numeric=True,
            add_missing_indicators=True,
        ),
    ]

    results_df = pd.DataFrame(results).sort_values(
        ["test_macro_f1", "test_log_loss"],
        ascending=[False, True],
    )
    results_df.to_csv(RESULTS_DIR / "feature_expansion_results.csv", index=False)

    best_row = results_df.iloc[0].to_dict()
    summary = {
        "best_configuration": best_row["configuration"],
        "best_test_macro_f1": best_row["test_macro_f1"],
        "best_test_accuracy": best_row["test_accuracy"],
        "unavailable_raw_feature_sources": [
            "shots",
            "shots_on_target",
            "goalkeeper_statistics",
        ],
    }
    save_json(RESULTS_DIR / "feature_expansion_summary.json", summary)
    return summary


def main():
    summary = run_experiment()
    print("Feature expansion experiment complete.")
    print(f"Best configuration: {summary['best_configuration']}")
    print(f"Best test accuracy: {summary['best_test_accuracy']:.4f}")
    print(f"Best test macro F1: {summary['best_test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
