# Football Match Prediction Project

This project predicts football match outcomes for the Top 5 European leagues:

- `0`: Home Win
- `1`: Draw
- `2`: Away Win

It also simulates league-table outcomes for the current season by combining:

- played matches already on the board
- modelled probabilities and sampled scorelines for the remaining fixtures

## Project Structure

- `data/raw` - API downloads from Understat
- `data/processed` - cleaned matches and engineered pre-match features
- `src` - Python source files
- `models` - trained model artifacts
- `outputs` - metrics, reports, and simulation outputs

## Data Model

Historical match data is downloaded from Understat for seasons `2014` to `2025`.

Season values use the start year:

- `2024` means `2024/25`
- `2025` means `2025/26`

Raw file:

- `data/raw/matches_raw.csv`

Raw columns:

- `league`
- `season`
- `date`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`
- `home_xg`
- `away_xg`

Cleaned data keeps both played matches and scheduled fixtures. The cleaned file adds:

- `is_played`
- `match_result`

Engineered data is then converted into the final 2D model matrix by:

- encoding `home_team` and `away_team`
- scaling numeric pre-match features from training-set statistics
- saving the fitted preprocessing artifacts alongside the model

## Pipeline

Run the full pipeline step by step:

```bash
pip install -r requirements.txt
python src/collect_data.py
python src/clean_data.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
python src/simulate_table.py
```

What each step does:

- `collect_data.py` downloads Understat data through season `2025`
- `clean_data.py` preserves unplayed fixtures instead of dropping them
- `feature_engineering.py` builds pre-match features without leaking future results and updates Elo from a directed xG-difference match graph
- `train_model.py` backtests on seasons `2023` and `2024`, encodes team names, scales numeric features, saves transformed train/test matrices, then trains the production model on all played matches through `2024/25`
- `evaluate_model.py` writes a detailed season `2024` evaluation report for the saved model
- `simulate_table.py` simulates season `2025` (`2025/26`) from the current table plus remaining fixtures, including goal difference and European qualification probabilities

## Current Artifacts

Latest generated files:

- `data/raw/matches_raw.csv`
- `data/processed/matches_clean.csv`
- `data/processed/matches_features.csv`
- `data/processed/train_data.csv`
- `data/processed/test_data.csv`
- `models/xgboost_model.joblib`
- `models/feature_columns.joblib`
- `models/preprocessor.joblib`
- `outputs/model_metrics.json`
- `outputs/evaluation_report.txt`
- `outputs/test_predictions.csv`
- `outputs/simulation_summary.csv`
- `outputs/simulation_position_probabilities.csv`

Latest run snapshot:

- Raw rows: `21690`
- Cleaned rows: `21690`
- Played matches: `21276`
- Scheduled fixtures: `414`
- Final model matrix columns: `15`
- Encoded categorical columns: `2`
- Scaled numeric columns: `13`
- Backtest validation season: `2023/24`
- Backtest test season: `2024/25`
- Simulation target season: `2025/26`
- Simulation runs: `10000`

Backtest metrics from `outputs/model_metrics.json`:

- Validation accuracy: `0.5365`
- Validation macro F1: `0.4247`
- Test accuracy: `0.5285`
- Test macro F1: `0.4150`

## Notes

- The implementation is intentionally simple and oriented toward learning.
- Feature engineering uses rolling team form, xG, graph-backed Elo, and rest-day features.
- The saved model consumes an encoded and scaled 2D feature matrix.
- The simulator ranks teams by points, goal difference, goals scored, then team name.
