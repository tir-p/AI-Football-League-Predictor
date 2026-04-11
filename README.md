# Football Match Prediction Project

This is a simple Python project for predicting football match outcomes:

- `0`: Home Win
- `1`: Draw
- `2`: Away Win

The project also uses match predictions to simulate a final league table.

## Project Structure

- `data/raw` - original input files
- `data/processed` - cleaned data and engineered features
- `src` - Python source files
- `models` - trained model files
- `outputs` - evaluation reports and simulated table

## Raw Data

The project now downloads historical match data for the Top 5 European leagues
from Understat for seasons `2014` to `2024`.

Raw output file:

- `data/raw/matches_raw.csv`

Columns:

- `league`
- `season`
- `date`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`
- `home_xg`
- `away_xg`

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the pipeline step by step:

```bash
python src/collect_data.py
python src/clean_data.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
python src/simulate_table.py
```

## Notes

- The code is intentionally simple for learning purposes.
- Feature engineering uses basic historical team performance before each match.
- The model uses `XGBoost` for 3-class classification.

## Stored Summaries

Saved pipeline summaries are kept in `outputs/`.

- `outputs/collect_data_summary.txt`
- `outputs/clean_data_summary.txt`
- `outputs/simulate_table_summary.txt`

Current results:

- Raw matches collected: `19938`
- Cleaned matches: `19837`
- Missing values after cleaning: `0`
- Simulation season: `2024`
- Simulation runs: `10000`

Simulation output files:

- `outputs/simulation_summary.csv`
- `outputs/simulation_position_probabilities.csv`
