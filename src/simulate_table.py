"""
Simulate league table outcomes using the trained XGBoost model.

This script uses the pre-match feature dataset, predicts probabilities for each
fixture, samples match results from those probabilities, and repeats the full
season simulation many times. For simplicity, league tables are ranked by
points only, with team name used as a fixed tie-breaker.
"""

import joblib
import numpy as np
import pandas as pd

from config import (
    FEATURE_COLUMNS_FILE,
    FEATURES_FILE,
    MODEL_FILE,
    SIMULATION_POSITION_PROBABILITIES_FILE,
    SIMULATION_SUMMARY_FILE,
)


TARGET_SEASON = 2024
NUM_SIMULATIONS = 10000
TOP_4_CUTOFF = 4
RELEGATION_SPOTS = 3


def load_simulation_inputs():
    """Load the model, feature columns, and feature data."""
    features_df = pd.read_csv(FEATURES_FILE)
    features_df["date"] = pd.to_datetime(features_df["date"])
    features_df = features_df.sort_values(["league", "date"]).reset_index(drop=True)

    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
    return features_df, model, feature_columns


def get_target_fixtures(features_df, season):
    """
    Select the fixtures to simulate.
    For this project, we simulate the fixtures in the chosen season.
    """
    season_df = features_df[features_df["season"] == season].copy()
    return season_df


def get_fixture_probabilities(model, fixtures_df, feature_columns):
    """Predict match outcome probabilities for each fixture."""
    X = fixtures_df[feature_columns]
    probabilities = model.predict_proba(X)
    return probabilities


def simulate_one_league(league_fixtures, probabilities, rng):
    """Run Monte Carlo simulations for one league."""
    teams = sorted(set(league_fixtures["home_team"]).union(set(league_fixtures["away_team"])))
    team_to_index = {team: index for index, team in enumerate(teams)}

    position_counts = np.zeros((len(teams), len(teams)), dtype=int)
    expected_position_sum = np.zeros(len(teams), dtype=float)
    league_winner_counts = np.zeros(len(teams), dtype=int)
    top_4_counts = np.zeros(len(teams), dtype=int)
    relegation_counts = np.zeros(len(teams), dtype=int)

    home_indices = league_fixtures["home_team"].map(team_to_index).to_numpy()
    away_indices = league_fixtures["away_team"].map(team_to_index).to_numpy()

    for _ in range(NUM_SIMULATIONS):
        points = np.zeros(len(teams), dtype=int)

        # Each fixture is sampled independently using the model probabilities.
        for match_index in range(len(league_fixtures)):
            result = rng.choice([0, 1, 2], p=probabilities[match_index])
            home_index = home_indices[match_index]
            away_index = away_indices[match_index]

            if result == 0:
                points[home_index] += 3
            elif result == 1:
                points[home_index] += 1
                points[away_index] += 1
            else:
                points[away_index] += 3

        ranking = sorted(range(len(teams)), key=lambda index: (-points[index], teams[index]))

        for position, team_index in enumerate(ranking, start=1):
            position_counts[team_index, position - 1] += 1
            expected_position_sum[team_index] += position

        league_winner_counts[ranking[0]] += 1

        top_n = min(TOP_4_CUTOFF, len(teams))
        relegation_n = min(RELEGATION_SPOTS, len(teams))

        for team_index in ranking[:top_n]:
            top_4_counts[team_index] += 1
        for team_index in ranking[-relegation_n:]:
            relegation_counts[team_index] += 1

    summary_rows = []
    position_rows = []

    for team_index, team in enumerate(teams):
        summary_rows.append(
            {
                "league": league_fixtures["league"].iloc[0],
                "season": league_fixtures["season"].iloc[0],
                "team": team,
                "expected_finishing_position": expected_position_sum[team_index] / NUM_SIMULATIONS,
                "probability_of_winning_league": league_winner_counts[team_index] / NUM_SIMULATIONS,
                "probability_of_finishing_top_4": top_4_counts[team_index] / NUM_SIMULATIONS,
                "probability_of_relegation": relegation_counts[team_index] / NUM_SIMULATIONS,
            }
        )

        for position in range(1, len(teams) + 1):
            position_rows.append(
                {
                    "league": league_fixtures["league"].iloc[0],
                    "season": league_fixtures["season"].iloc[0],
                    "team": team,
                    "position": position,
                    "probability": position_counts[team_index, position - 1] / NUM_SIMULATIONS,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["league", "expected_finishing_position", "team"]
    )
    position_df = pd.DataFrame(position_rows).sort_values(["league", "team", "position"])
    return summary_df, position_df


def run_simulations(features_df, model, feature_columns):
    """Run table simulations for each league in the target season."""
    fixtures_df = get_target_fixtures(features_df, TARGET_SEASON)

    if fixtures_df.empty:
        raise ValueError(f"No fixtures found for season {TARGET_SEASON}.")

    rng = np.random.default_rng(seed=42)
    summary_frames = []
    position_frames = []

    for league_name, league_fixtures in fixtures_df.groupby("league"):
        league_fixtures = league_fixtures.sort_values("date").reset_index(drop=True)
        probabilities = get_fixture_probabilities(model, league_fixtures, feature_columns)
        league_summary, league_positions = simulate_one_league(league_fixtures, probabilities, rng)
        summary_frames.append(league_summary)
        position_frames.append(league_positions)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    position_df = pd.concat(position_frames, ignore_index=True)
    return summary_df, position_df


def main():
    features_df, model, feature_columns = load_simulation_inputs()
    summary_df, position_df = run_simulations(features_df, model, feature_columns)

    SIMULATION_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SIMULATION_SUMMARY_FILE, index=False)
    position_df.to_csv(SIMULATION_POSITION_PROBABILITIES_FILE, index=False)

    print(f"Simulation summary saved to: {SIMULATION_SUMMARY_FILE}")
    print(f"Position probabilities saved to: {SIMULATION_POSITION_PROBABILITIES_FILE}")
    print()
    print("Top teams by expected finishing position:")
    print(summary_df.groupby("league").head(5))


if __name__ == "__main__":
    main()
