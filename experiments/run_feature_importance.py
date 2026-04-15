import pandas as pd
from sklearn.inspection import permutation_importance

from common import (
    RESULTS_DIR,
    ensure_results_dir,
    evaluate_model,
    get_balanced_sample_weights,
    prepare_design_matrices,
    save_json,
    select_best_xgb_params,
    split_chronologically_by_season,
    train_xgboost_classifier,
)
from feature_sets import build_expanded_feature_frame, get_expanded_numeric_columns


def run_experiment():
    """Run feature-importance analysis for the tuned XGBoost model."""
    ensure_results_dir()

    df = build_expanded_feature_frame()
    numeric_columns = get_expanded_numeric_columns()
    train_df, validation_df, test_df = split_chronologically_by_season(df)

    _, X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_design_matrices(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        numeric_columns=numeric_columns,
        impute_numeric=True,
        add_missing_indicators=True,
    )

    sample_weights = get_balanced_sample_weights(y_train)
    best_params, tuning_results_df, _ = select_best_xgb_params(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        sample_weight=sample_weights,
    )
    tuning_results_df.to_csv(RESULTS_DIR / "feature_importance_xgboost_tuning_results.csv", index=False)

    train_and_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    _, X_train_final, _, X_test_final, y_train_final, _, y_test_final = prepare_design_matrices(
        train_df=train_and_validation_df,
        validation_df=test_df,
        test_df=test_df,
        numeric_columns=numeric_columns,
        impute_numeric=True,
        add_missing_indicators=True,
    )

    final_sample_weights = get_balanced_sample_weights(y_train_final)
    model = train_xgboost_classifier(
        X_train=X_train_final,
        y_train=y_train_final,
        params=best_params,
        sample_weight=final_sample_weights,
    )
    _, test_metrics = evaluate_model(model, X_test_final, y_test_final)

    builtin_importance_df = pd.DataFrame(
        {
            "feature": X_train_final.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    builtin_importance_df.to_csv(RESULTS_DIR / "feature_importance_builtin.csv", index=False)

    permutation = permutation_importance(
        estimator=model,
        X=X_test_final,
        y=y_test_final,
        scoring="f1_macro",
        n_repeats=5,
        random_state=42,
        n_jobs=4,
    )
    permutation_df = pd.DataFrame(
        {
            "feature": X_test_final.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    permutation_df.to_csv(RESULTS_DIR / "feature_importance_permutation.csv", index=False)

    summary = {
        "selected_xgboost_params": best_params,
        "test_metrics": test_metrics,
        "top_builtin_features": builtin_importance_df.head(10)["feature"].tolist(),
        "top_permutation_features": permutation_df.head(10)["feature"].tolist(),
    }
    save_json(RESULTS_DIR / "feature_importance_summary.json", summary)
    return summary


def main():
    summary = run_experiment()
    print("Feature importance analysis complete.")
    print(f"Top builtin feature: {summary['top_builtin_features'][0]}")
    print(f"Top permutation feature: {summary['top_permutation_features'][0]}")


if __name__ == "__main__":
    main()
