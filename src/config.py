from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

RAW_MATCHES_FILE = RAW_DATA_DIR / "matches.csv"
RAW_MATCHES_RAW_FILE = RAW_DATA_DIR / "matches_raw.csv"
CLEAN_MATCHES_FILE = PROCESSED_DATA_DIR / "matches_clean.csv"
FEATURES_FILE = PROCESSED_DATA_DIR / "matches_features.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train_data.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test_data.csv"
MODEL_FILE = MODELS_DIR / "xgboost_model.joblib"
FEATURE_COLUMNS_FILE = MODELS_DIR / "feature_columns.joblib"
EVALUATION_FILE = OUTPUTS_DIR / "evaluation_report.txt"
MODEL_METRICS_FILE = OUTPUTS_DIR / "model_metrics.json"
TEST_PREDICTIONS_FILE = OUTPUTS_DIR / "test_predictions.csv"
SIMULATED_TABLE_FILE = OUTPUTS_DIR / "simulated_league_table.csv"
SIMULATION_SUMMARY_FILE = OUTPUTS_DIR / "simulation_summary.csv"
SIMULATION_POSITION_PROBABILITIES_FILE = OUTPUTS_DIR / "simulation_position_probabilities.csv"
