import joblib
import pandas as pd

from common import REVERSE_RESULT_MAP
from config import FEATURE_COLUMNS_FILE, MODEL_FILE, SIMULATED_TABLE_FILE, TEST_FILE


def create_empty_table(teams):
    """Create a blank league table for a list of teams."""
    table = pd.DataFrame(index=sorted(teams))
    table["Played"] = 0
    table["Won"] = 0
    table["Drawn"] = 0
    table["Lost"] = 0
    table["GoalsFor"] = 0
    table["GoalsAgainst"] = 0
    table["GoalDifference"] = 0
    table["Points"] = 0
    return table


def update_table(table, home_team, away_team, predicted_result):
    """
    Update the table using the predicted match result.
    Goals are not predicted here, so goals columns stay at 0.
    """
    table.loc[home_team, "Played"] += 1
    table.loc[away_team, "Played"] += 1

    if predicted_result == "H":
        table.loc[home_team, "Won"] += 1
        table.loc[away_team, "Lost"] += 1
        table.loc[home_team, "Points"] += 3
    elif predicted_result == "A":
        table.loc[away_team, "Won"] += 1
        table.loc[home_team, "Lost"] += 1
        table.loc[away_team, "Points"] += 3
    else:
        table.loc[home_team, "Drawn"] += 1
        table.loc[away_team, "Drawn"] += 1
        table.loc[home_team, "Points"] += 1
        table.loc[away_team, "Points"] += 1


def simulate_table(fixtures_df, model, feature_columns):
    """Predict each match and build a final table."""
    X = fixtures_df[feature_columns]
    predictions = model.predict(X)
    predicted_labels = [REVERSE_RESULT_MAP[int(pred)] for pred in predictions]

    teams = set(fixtures_df["HomeTeam"]).union(set(fixtures_df["AwayTeam"]))
    table = create_empty_table(teams)

    for (_, row), predicted_result in zip(fixtures_df.iterrows(), predicted_labels):
        update_table(table, row["HomeTeam"], row["AwayTeam"], predicted_result)

    table["GoalDifference"] = table["GoalsFor"] - table["GoalsAgainst"]
    table = table.sort_values(
        by=["Points", "GoalDifference", "GoalsFor"],
        ascending=False,
    ).reset_index()
    table = table.rename(columns={"index": "Team"})

    return table


def main():
    fixtures_df = pd.read_csv(TEST_FILE)
    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURE_COLUMNS_FILE)

    simulated_table = simulate_table(fixtures_df, model, feature_columns)

    SIMULATED_TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    simulated_table.to_csv(SIMULATED_TABLE_FILE, index=False)

    print("Simulated league table:")
    print(simulated_table.head(10))
    print(f"Saved to: {SIMULATED_TABLE_FILE}")


if __name__ == "__main__":
    main()
