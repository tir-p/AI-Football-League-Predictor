import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from common import (
    RESULTS_DIR,
    build_calibration_curve_frame,
    ensure_results_dir,
    evaluate_model,
    evaluate_probabilities,
    normalise_probabilities,
    prepare_design_matrices,
    save_json,
    select_best_xgb_params,
    split_chronologically_by_season,
    train_xgboost_classifier,
)
from feature_sets import build_expanded_feature_frame, get_expanded_numeric_columns


def logit_transform(values):
    """Apply a numerically stable logit transform."""
    values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values)).reshape(-1, 1)


def fit_sigmoid_calibrators(y_true, probabilities):
    """Fit one-vs-rest sigmoid calibrators."""
    calibrators = []
    y_true = np.asarray(y_true, dtype=int)

    for class_index in range(probabilities.shape[1]):
        calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
        calibrator.fit(logit_transform(probabilities[:, class_index]), (y_true == class_index).astype(int))
        calibrators.append(calibrator)

    return calibrators


def apply_sigmoid_calibrators(probabilities, calibrators):
    """Apply one-vs-rest sigmoid calibrators and renormalise."""
    calibrated_columns = []
    for class_index, calibrator in enumerate(calibrators):
        calibrated_columns.append(
            calibrator.predict_proba(logit_transform(probabilities[:, class_index]))[:, 1]
        )
    return normalise_probabilities(np.column_stack(calibrated_columns))


def fit_isotonic_calibrators(y_true, probabilities):
    """Fit one-vs-rest isotonic regressors."""
    calibrators = []
    y_true = np.asarray(y_true, dtype=int)

    for class_index in range(probabilities.shape[1]):
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(probabilities[:, class_index], (y_true == class_index).astype(int))
        calibrators.append(calibrator)

    return calibrators


def apply_isotonic_calibrators(probabilities, calibrators):
    """Apply one-vs-rest isotonic calibration and renormalise."""
    calibrated_columns = []
    for class_index, calibrator in enumerate(calibrators):
        calibrated_columns.append(calibrator.transform(probabilities[:, class_index]))
    return normalise_probabilities(np.column_stack(calibrated_columns))


def run_experiment():
    """Run the planned probability calibration analysis."""
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

    best_params, tuning_results_df, _ = select_best_xgb_params(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )
    tuning_results_df.to_csv(RESULTS_DIR / "calibration_xgboost_tuning_results.csv", index=False)

    base_model = train_xgboost_classifier(X_train, y_train, params=best_params)
    validation_probabilities, validation_metrics = evaluate_model(base_model, X_validation, y_validation)
    test_probabilities, test_metrics = evaluate_model(base_model, X_test, y_test)

    sigmoid_calibrators = fit_sigmoid_calibrators(y_validation, validation_probabilities)
    isotonic_calibrators = fit_isotonic_calibrators(y_validation, validation_probabilities)

    sigmoid_test_probabilities = apply_sigmoid_calibrators(test_probabilities, sigmoid_calibrators)
    isotonic_test_probabilities = apply_isotonic_calibrators(test_probabilities, isotonic_calibrators)

    results = [
        {
            "method": "uncalibrated",
            **{f"test_{metric_name}": metric_value for metric_name, metric_value in test_metrics.items()},
        },
        {
            "method": "sigmoid",
            **{
                f"test_{metric_name}": metric_value
                for metric_name, metric_value in evaluate_probabilities(y_test, sigmoid_test_probabilities).items()
            },
        },
        {
            "method": "isotonic",
            **{
                f"test_{metric_name}": metric_value
                for metric_name, metric_value in evaluate_probabilities(y_test, isotonic_test_probabilities).items()
            },
        },
    ]

    results_df = pd.DataFrame(results).sort_values(["test_log_loss", "test_multiclass_brier"])
    results_df.to_csv(RESULTS_DIR / "calibration_results.csv", index=False)

    calibration_curve_df = pd.concat(
        [
            build_calibration_curve_frame(y_test, test_probabilities, "uncalibrated", "test"),
            build_calibration_curve_frame(y_test, sigmoid_test_probabilities, "sigmoid", "test"),
            build_calibration_curve_frame(y_test, isotonic_test_probabilities, "isotonic", "test"),
        ],
        ignore_index=True,
    )
    calibration_curve_df.to_csv(RESULTS_DIR / "calibration_curves_test.csv", index=False)

    best_row = results_df.iloc[0].to_dict()
    summary = {
        "best_calibration_method_on_test_log_loss": best_row["method"],
        "validation_reference_metrics_before_calibration": validation_metrics,
        "selected_xgboost_params": best_params,
    }
    save_json(RESULTS_DIR / "calibration_summary.json", summary)
    return summary


def main():
    summary = run_experiment()
    print("Calibration analysis complete.")
    print(
        "Best calibration method on test log loss: "
        f"{summary['best_calibration_method_on_test_log_loss']}"
    )


if __name__ == "__main__":
    main()
