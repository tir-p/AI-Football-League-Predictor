import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLEAN_MATCHES_FILE, FEATURES_FILE  # noqa: E402
from preprocessing import get_categorical_feature_columns, get_numeric_feature_columns  # noqa: E402


CLASS_LABELS = [0, 1, 2]
RESULT_LABELS = {
    0: "home_win",
    1: "draw",
    2: "away_win",
}


def ensure_results_dir():
    """Create the experiment output directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def load_base_features():
    """Load the existing engineered feature matrix."""
    df = pd.read_csv(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_clean_matches():
    """Load the cleaned match-level dataset."""
    df = pd.read_csv(CLEAN_MATCHES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def split_chronologically_by_season(df, train_end_season=2022, validation_season=2023, test_season=2024):
    """Create train, validation, and test splits using played matches only."""
    played_df = df[df["match_result"].notna()].copy()
    train_df = played_df[played_df["season"] <= train_end_season].copy()
    validation_df = played_df[played_df["season"] == validation_season].copy()
    test_df = played_df[played_df["season"] == test_season].copy()
    return train_df, validation_df, test_df


def fit_experiment_preprocessor(
    train_df,
    numeric_columns,
    categorical_columns=None,
    impute_numeric=False,
    add_missing_indicators=False,
):
    """Fit a simple encoder and numeric standardisation statistics."""
    if categorical_columns is None:
        categorical_columns = get_categorical_feature_columns()

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train_df[categorical_columns].fillna("Unknown"))

    numeric_train = train_df[numeric_columns].copy()
    imputation_values = numeric_train.median() if impute_numeric else None

    if imputation_values is not None:
        numeric_train = numeric_train.fillna(imputation_values)

    numeric_means = numeric_train.mean()
    numeric_stds = numeric_train.std().replace(0, 1.0).fillna(1.0)

    feature_columns = [f"{column}_encoded" for column in categorical_columns] + list(numeric_columns)
    if add_missing_indicators:
        feature_columns.extend([f"{column}_missing" for column in numeric_columns])

    return {
        "encoder": encoder,
        "categorical_columns": list(categorical_columns),
        "numeric_columns": list(numeric_columns),
        "numeric_means": numeric_means,
        "numeric_stds": numeric_stds,
        "imputation_values": imputation_values,
        "add_missing_indicators": add_missing_indicators,
        "feature_columns": feature_columns,
    }


def transform_experiment_features(df, preprocessor):
    """Transform raw rows into an encoded and scaled model matrix."""
    categorical_columns = preprocessor["categorical_columns"]
    numeric_columns = preprocessor["numeric_columns"]

    encoded_values = preprocessor["encoder"].transform(df[categorical_columns].fillna("Unknown"))
    numeric_df = df[numeric_columns].copy()
    missing_df = numeric_df.isna().astype(float)

    if preprocessor["imputation_values"] is not None:
        numeric_df = numeric_df.fillna(preprocessor["imputation_values"])

    scaled_numeric = (numeric_df - preprocessor["numeric_means"]) / preprocessor["numeric_stds"]
    frames = [
        pd.DataFrame(
            encoded_values,
            columns=[f"{column}_encoded" for column in categorical_columns],
            index=df.index,
        ),
        scaled_numeric,
    ]

    if preprocessor["add_missing_indicators"]:
        missing_df = missing_df.rename(columns=lambda column: f"{column}_missing")
        frames.append(missing_df)

    return pd.concat(frames, axis=1)


def prepare_design_matrices(
    train_df,
    validation_df,
    test_df,
    numeric_columns,
    categorical_columns=None,
    impute_numeric=False,
    add_missing_indicators=False,
):
    """Fit preprocessing on train only, then transform all splits."""
    preprocessor = fit_experiment_preprocessor(
        train_df=train_df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        impute_numeric=impute_numeric,
        add_missing_indicators=add_missing_indicators,
    )

    X_train = transform_experiment_features(train_df, preprocessor)
    X_validation = transform_experiment_features(validation_df, preprocessor)
    X_test = transform_experiment_features(test_df, preprocessor)

    y_train = train_df["match_result"].astype(int)
    y_validation = validation_df["match_result"].astype(int)
    y_test = test_df["match_result"].astype(int)
    return preprocessor, X_train, X_validation, X_test, y_train, y_validation, y_test


def get_balanced_sample_weights(y):
    """Return inverse-frequency sample weights for a three-class problem."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=len(CLASS_LABELS))
    counts = np.where(counts == 0, 1, counts)
    weights = len(y) / (len(CLASS_LABELS) * counts)
    return weights[y]


def get_baseline_xgb_params(overrides=None):
    """Return the baseline XGBoost configuration used in experiments."""
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": 4,
    }
    if overrides:
        params.update(overrides)
    return params


def train_xgboost_classifier(X_train, y_train, params=None, sample_weight=None):
    """Train an XGBoost classifier."""
    model = XGBClassifier(**get_baseline_xgb_params(params))
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_logistic_regression_classifier(X_train, y_train):
    """Train a multinomial logistic regression model."""
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_classifier(X_train, y_train):
    """Train a random-forest baseline."""
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    return model


def train_hist_gradient_boosting_classifier(X_train, y_train):
    """Train a histogram-based gradient boosting baseline."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=5,
        max_iter=250,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def multiclass_brier_score(y_true, probabilities):
    """Return the multiclass Brier score."""
    y_true = np.asarray(y_true, dtype=int)
    one_hot = np.eye(len(CLASS_LABELS))[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true, probabilities, n_bins=10):
    """Compute a simple one-vs-rest expected calibration error."""
    y_true = np.asarray(y_true, dtype=int)
    class_eces = []

    for class_index in range(probabilities.shape[1]):
        class_targets = (y_true == class_index).astype(int)
        class_probabilities = probabilities[:, class_index]
        bin_ids = np.minimum((class_probabilities * n_bins).astype(int), n_bins - 1)
        class_ece = 0.0

        for bin_id in range(n_bins):
            mask = bin_ids == bin_id
            if not np.any(mask):
                continue

            observed_frequency = class_targets[mask].mean()
            mean_confidence = class_probabilities[mask].mean()
            class_ece += mask.mean() * abs(observed_frequency - mean_confidence)

        class_eces.append(class_ece)

    return float(np.mean(class_eces))


def evaluate_probabilities(y_true, probabilities):
    """Evaluate prediction probabilities using classification and calibration metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = normalise_probabilities(np.asarray(probabilities, dtype=float))
    predicted_classes = probabilities.argmax(axis=1)

    return {
        "accuracy": float(accuracy_score(y_true, predicted_classes)),
        "macro_f1": float(f1_score(y_true, predicted_classes, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=CLASS_LABELS)),
        "multiclass_brier": multiclass_brier_score(y_true, probabilities),
        "ece": expected_calibration_error(y_true, probabilities),
    }


def evaluate_model(model, X, y):
    """Predict probabilities with a fitted model and compute metrics."""
    probabilities = model.predict_proba(X)
    return probabilities, evaluate_probabilities(y, probabilities)


def average_probabilities(probability_matrices, weights=None):
    """Average class probabilities across models."""
    stacked = np.stack(probability_matrices, axis=0)
    if weights is None:
        return stacked.mean(axis=0)

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return np.tensordot(weights, stacked, axes=(0, 0))


def normalise_probabilities(probabilities):
    """Normalise possibly unscaled class probabilities row-wise."""
    probabilities = np.clip(probabilities, 1e-9, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probabilities / row_sums


def build_calibration_curve_frame(y_true, probabilities, method_name, split_name, n_bins=10):
    """Build a calibration-curve table for each class."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for class_index, label in RESULT_LABELS.items():
        class_targets = (y_true == class_index).astype(int)
        class_probabilities = probabilities[:, class_index]
        bin_ids = np.digitize(class_probabilities, bin_edges[1:-1], right=True)

        for bin_id in range(n_bins):
            mask = bin_ids == bin_id
            if not np.any(mask):
                continue

            rows.append(
                {
                    "split": split_name,
                    "method": method_name,
                    "class_label": label,
                    "bin_lower": float(bin_edges[bin_id]),
                    "bin_upper": float(bin_edges[bin_id + 1]),
                    "sample_count": int(mask.sum()),
                    "mean_predicted_probability": float(class_probabilities[mask].mean()),
                    "observed_frequency": float(class_targets[mask].mean()),
                }
            )

    return pd.DataFrame(rows)


def get_xgb_tuning_candidates():
    """Return a compact hyperparameter search space for XGBoost."""
    return [
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.07, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 450, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.85, "colsample_bytree": 0.85},
    ]


def select_best_xgb_params(X_train, y_train, X_validation, y_validation, sample_weight=None):
    """Tune XGBoost on validation macro F1 and keep per-candidate metrics."""
    candidate_rows = []
    best_params = None
    best_score = None
    best_log_loss = None
    best_validation_probabilities = None

    for candidate_index, params in enumerate(get_xgb_tuning_candidates(), start=1):
        model = train_xgboost_classifier(
            X_train=X_train,
            y_train=y_train,
            params=params,
            sample_weight=sample_weight,
        )
        validation_probabilities, validation_metrics = evaluate_model(model, X_validation, y_validation)

        candidate_row = {
            "candidate_id": candidate_index,
            **params,
            **{f"validation_{metric_name}": metric_value for metric_name, metric_value in validation_metrics.items()},
        }
        candidate_rows.append(candidate_row)

        candidate_score = validation_metrics["macro_f1"]
        candidate_log_loss = validation_metrics["log_loss"]

        if best_score is None or candidate_score > best_score or (
            np.isclose(candidate_score, best_score) and candidate_log_loss < best_log_loss
        ):
            best_score = candidate_score
            best_log_loss = candidate_log_loss
            best_params = params
            best_validation_probabilities = validation_probabilities

    return best_params, pd.DataFrame(candidate_rows), best_validation_probabilities


def save_json(output_path, payload):
    """Save a JSON file with indentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def get_base_numeric_columns():
    """Return the existing numeric feature set from the production pipeline."""
    return list(get_numeric_feature_columns())
