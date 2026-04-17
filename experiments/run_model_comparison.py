import pandas as pd

from common import (
    RESULTS_DIR,
    average_probabilities,
    ensure_results_dir,
    evaluate_model,
    evaluate_probabilities,
    get_balanced_sample_weights,
    get_base_numeric_columns,
    load_base_features,
    prepare_design_matrices,
    save_json,
    select_best_xgb_params,
    split_chronologically_by_season,
    train_hist_gradient_boosting_classifier,
    train_logistic_regression_classifier,
    train_random_forest_classifier,
    train_xgboost_classifier,
)
from feature_sets import build_expanded_feature_frame, get_expanded_numeric_columns


def build_result_row(model_name, feature_set_name, validation_metrics, test_metrics, final_feature_count):
    """Create a standard result row."""
    return {
        "model": model_name,
        "feature_set": feature_set_name,
        "final_feature_count": final_feature_count,
        **{f"validation_{metric_name}": metric_value for metric_name, metric_value in validation_metrics.items()},
        **{f"test_{metric_name}": metric_value for metric_name, metric_value in test_metrics.items()},
    }


def run_experiment():
    """Run the planned model comparison and ensemble study."""
    ensure_results_dir()

    base_df = load_base_features()
    expanded_df = build_expanded_feature_frame()
    base_numeric_columns = get_base_numeric_columns()
    expanded_numeric_columns = get_expanded_numeric_columns()

    base_train_df, base_validation_df, base_test_df = split_chronologically_by_season(base_df)
    _, base_X_train, base_X_validation, base_X_test, base_y_train, base_y_validation, base_y_test = (
        prepare_design_matrices(
            train_df=base_train_df,
            validation_df=base_validation_df,
            test_df=base_test_df,
            numeric_columns=base_numeric_columns,
            impute_numeric=False,
            add_missing_indicators=False,
        )
    )

    expanded_train_df, expanded_validation_df, expanded_test_df = split_chronologically_by_season(expanded_df)
    _, X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_design_matrices(
        train_df=expanded_train_df,
        validation_df=expanded_validation_df,
        test_df=expanded_test_df,
        numeric_columns=expanded_numeric_columns,
        impute_numeric=True,
        add_missing_indicators=True,
    )

    sample_weights = get_balanced_sample_weights(y_train)
    tuning_params, tuning_results_df, _ = select_best_xgb_params(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )
    tuning_results_df.to_csv(RESULTS_DIR / "xgboost_tuning_results.csv", index=False)

    result_rows = []
    model_probabilities = {}

    baseline_model = train_xgboost_classifier(base_X_train, base_y_train)
    baseline_validation_probabilities, baseline_validation_metrics = evaluate_model(
        baseline_model, base_X_validation, base_y_validation
    )
    baseline_test_probabilities, baseline_test_metrics = evaluate_model(baseline_model, base_X_test, base_y_test)
    result_rows.append(
        build_result_row(
            model_name="baseline_xgboost",
            feature_set_name="current_features",
            validation_metrics=baseline_validation_metrics,
            test_metrics=baseline_test_metrics,
            final_feature_count=base_X_train.shape[1],
        )
    )
    model_probabilities["baseline_xgboost"] = (baseline_validation_probabilities, baseline_test_probabilities)

    expanded_xgb_model = train_xgboost_classifier(X_train, y_train)
    expanded_validation_probabilities, expanded_validation_metrics = evaluate_model(
        expanded_xgb_model, X_validation, y_validation
    )
    expanded_test_probabilities, expanded_test_metrics = evaluate_model(expanded_xgb_model, X_test, y_test)
    result_rows.append(
        build_result_row(
            model_name="xgboost_expanded",
            feature_set_name="expanded_features",
            validation_metrics=expanded_validation_metrics,
            test_metrics=expanded_test_metrics,
            final_feature_count=X_train.shape[1],
        )
    )
    model_probabilities["xgboost_expanded"] = (expanded_validation_probabilities, expanded_test_probabilities)

    weighted_xgb_model = train_xgboost_classifier(X_train, y_train, sample_weight=sample_weights)
    weighted_validation_probabilities, weighted_validation_metrics = evaluate_model(
        weighted_xgb_model, X_validation, y_validation
    )
    weighted_test_probabilities, weighted_test_metrics = evaluate_model(weighted_xgb_model, X_test, y_test)
    result_rows.append(
        build_result_row(
            model_name="weighted_xgboost_expanded",
            feature_set_name="expanded_features",
            validation_metrics=weighted_validation_metrics,
            test_metrics=weighted_test_metrics,
            final_feature_count=X_train.shape[1],
        )
    )
    model_probabilities["weighted_xgboost_expanded"] = (
        weighted_validation_probabilities,
        weighted_test_probabilities,
    )

    tuned_xgb_model = train_xgboost_classifier(X_train, y_train, params=tuning_params)
    tuned_validation_probabilities, tuned_validation_metrics = evaluate_model(
        tuned_xgb_model, X_validation, y_validation
    )
    tuned_test_probabilities, tuned_test_metrics = evaluate_model(tuned_xgb_model, X_test, y_test)
    result_rows.append(
        build_result_row(
            model_name="tuned_xgboost_expanded",
            feature_set_name="expanded_features",
            validation_metrics=tuned_validation_metrics,
            test_metrics=tuned_test_metrics,
            final_feature_count=X_train.shape[1],
        )
    )
    model_probabilities["tuned_xgboost_expanded"] = (tuned_validation_probabilities, tuned_test_probabilities)

    logistic_model = train_logistic_regression_classifier(X_train, y_train)
    logistic_validation_probabilities, logistic_validation_metrics = evaluate_model(
        logistic_model, X_validation, y_validation
    )
    logistic_test_probabilities, logistic_test_metrics = evaluate_model(logistic_model, X_test, y_test)
    result_rows.append(
        build_result_row(
            model_name="logistic_regression_expanded",
            feature_set_name="expanded_features",
            validation_metrics=logistic_validation_metrics,
            test_metrics=logistic_test_metrics,
            final_feature_count=X_train.shape[1],
        )
    )
    model_probabilities["logistic_regression_expanded"] = (
        logistic_validation_probabilities,
        logistic_test_probabilities,
    )

    random_forest_model = train_random_forest_classifier(X_train, y_train)
    random_forest_validation_probabilities, random_forest_validation_metrics = evaluate_model(
        random_forest_model, X_validation, y_validation
    )
    random_forest_test_probabilities, random_forest_test_metrics = evaluate_model(
        random_forest_model, X_test, y_test
    )
    result_rows.append(
        build_result_row(
            model_name="random_forest_expanded",
            feature_set_name="expanded_features",
            validation_metrics=random_forest_validation_metrics,
            test_metrics=random_forest_test_metrics,
            final_feature_count=X_train.shape[1],
        )
    )
    model_probabilities["random_forest_expanded"] = (
        random_forest_validation_probabilities,
        random_forest_test_probabilities,
    )

    hist_gradient_model = train_hist_gradient_boosting_classifier(X_train, y_train)
    hist_gradient_validation_probabilities, hist_gradient_validation_metrics = evaluate_model(
        hist_gradient_model, X_validation, y_validation
    )
    hist_gradient_test_probabilities, hist_gradient_test_metrics = evaluate_model(
        hist_gradient_model, X_test, y_test
    )
    result_rows.append(
        build_result_row(
            model_name="hist_gradient_boosting_expanded",
            feature_set_name="expanded_features",
            validation_metrics=hist_gradient_validation_metrics,
            test_metrics=hist_gradient_test_metrics,
            final_feature_count=X_train.shape[1],
        )
    )
    model_probabilities["hist_gradient_boosting_expanded"] = (
        hist_gradient_validation_probabilities,
        hist_gradient_test_probabilities,
    )

    results_df = pd.DataFrame(result_rows)
    eligible_for_ensemble = results_df[results_df["model"] != "baseline_xgboost"].sort_values(
        ["validation_macro_f1", "validation_log_loss"],
        ascending=[False, True],
    )
    top_models = eligible_for_ensemble.head(3)["model"].tolist()
    ensemble_weights = eligible_for_ensemble.head(3)["validation_macro_f1"].to_numpy()

    ensemble_validation_probabilities = average_probabilities(
        [model_probabilities[model_name][0] for model_name in top_models],
        weights=ensemble_weights,
    )
    ensemble_test_probabilities = average_probabilities(
        [model_probabilities[model_name][1] for model_name in top_models],
        weights=ensemble_weights,
    )
    ensemble_validation_metrics = evaluate_probabilities(y_validation, ensemble_validation_probabilities)
    ensemble_test_metrics = evaluate_probabilities(y_test, ensemble_test_probabilities)
    result_rows.append(
        build_result_row(
            model_name="soft_voting_ensemble_expanded",
            feature_set_name="expanded_features",
            validation_metrics=ensemble_validation_metrics,
            test_metrics=ensemble_test_metrics,
            final_feature_count=X_train.shape[1],
        )
    )

    results_df = pd.DataFrame(result_rows).sort_values(
        ["test_macro_f1", "test_log_loss"],
        ascending=[False, True],
    )
    results_df.to_csv(RESULTS_DIR / "model_comparison_results.csv", index=False)

    best_row = results_df.iloc[0].to_dict()
    summary = {
        "best_model": best_row["model"],
        "best_test_macro_f1": best_row["test_macro_f1"],
        "best_test_accuracy": best_row["test_accuracy"],
        "ensemble_members": top_models,
        "selected_tuned_xgboost_params": tuning_params,
    }
    save_json(RESULTS_DIR / "model_comparison_summary.json", summary)
    return summary


def main():
    summary = run_experiment()
    print("Model comparison complete.")
    print(f"Best model: {summary['best_model']}")
    print(f"Best test accuracy: {summary['best_test_accuracy']:.4f}")
    print(f"Best test macro F1: {summary['best_test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
