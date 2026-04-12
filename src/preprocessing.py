import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def get_categorical_feature_columns():
    """Return the string columns that must be encoded for the model."""
    return ["home_team", "away_team"]


def get_numeric_feature_columns():
    """Return the numeric pre-match columns used by the model."""
    return [
        "home_avg_points_last_5",
        "away_avg_points_last_5",
        "home_goals_scored_avg_last_5",
        "away_goals_scored_avg_last_5",
        "home_xg_avg_last_5",
        "away_xg_avg_last_5",
        "home_graph_xg_diff_avg_last_5",
        "away_graph_xg_diff_avg_last_5",
        "home_elo",
        "away_elo",
        "elo_difference",
        "home_rest_days",
        "away_rest_days",
    ]


def get_transformed_feature_columns():
    """Return the final 2D feature-matrix columns seen by the model."""
    return [
        "home_team_encoded",
        "away_team_encoded",
        *get_numeric_feature_columns(),
    ]


def fit_preprocessor(train_df):
    """Fit categorical encoders and numeric scalers on training data only."""
    categorical_columns = get_categorical_feature_columns()
    numeric_columns = get_numeric_feature_columns()

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train_df[categorical_columns].fillna("Unknown"))

    scaler = StandardScaler()
    scaler.fit(train_df[numeric_columns])

    return {
        "encoder": encoder,
        "scaler": scaler,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "feature_columns": get_transformed_feature_columns(),
    }


def transform_features(df, preprocessor):
    """Convert raw match rows into the encoded and scaled model matrix."""
    categorical_columns = preprocessor["categorical_columns"]
    numeric_columns = preprocessor["numeric_columns"]

    encoded_categorical = preprocessor["encoder"].transform(df[categorical_columns].fillna("Unknown"))
    scaled_numeric = preprocessor["scaler"].transform(df[numeric_columns])

    transformed_values = np.hstack([encoded_categorical, scaled_numeric])
    return pd.DataFrame(
        transformed_values,
        columns=preprocessor["feature_columns"],
        index=df.index,
    )
