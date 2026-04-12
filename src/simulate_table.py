"""
Simulate league table outcomes using the trained XGBoost model.

This script uses the encoded and scaled pre-match feature matrix to predict
outcome probabilities for each fixture, samples scorelines for the remaining
matches, and repeats the full season simulation many times. League tables are
ranked by points, then goal difference, then goals scored, with team name as a
fixed final tie-breaker.
"""

import joblib
import numpy as np
import pandas as pd

from config import (
    FEATURES_FILE,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    SIMULATION_POSITION_PROBABILITIES_FILE,
    SIMULATION_SUMMARY_FILE,
)
from preprocessing import transform_features


TARGET_SEASON = 2025
NUM_SIMULATIONS = 10000
TOP_4_CUTOFF = 4
RELEGATION_SPOTS = 3
EUROPEAN_PLACES_BY_LEAGUE = {
    "premier_league": 7,
    "la_liga": 7,
    "serie_a": 7,
    "bundesliga": 6,
    "ligue_1": 4,
}
DEFAULT_SCORELINES = {
    0: (1, 0),
    1: (1, 1),
    2: (0, 1),
}


def load_simulation_inputs():
    """Load the model, preprocessor, and feature data."""
    features_df = pd.read_csv(FEATURES_FILE)
    features_df["date"] = pd.to_datetime(features_df["date"])
    features_df = features_df.sort_values(["league", "date"]).reset_index(drop=True)

    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    return features_df, model, preprocessor


def get_target_fixtures(features_df, season):
    """Return played and unplayed fixtures for the chosen season."""
    season_df = features_df[features_df["season"] == season].copy()
    played_df = season_df[season_df["match_result"].notna()].copy()
    unplayed_df = season_df[season_df["match_result"].isna()].copy()
    return played_df, unplayed_df


def get_fixture_probabilities(model, fixtures_df, preprocessor):
    """Predict match outcome probabilities for each fixture."""
    if fixtures_df.empty:
        return np.empty((0, 3))

    X = transform_features(fixtures_df, preprocessor)
    probabilities = model.predict_proba(X)
    return probabilities


def build_scoreline_pools(features_df):
    """Create historical scoreline pools keyed by league and outcome."""
    played_df = features_df[
        features_df["match_result"].notna()
        & features_df["home_goals"].notna()
        & features_df["away_goals"].notna()
    ].copy()

    scoreline_pools = {}
    for league, league_df in played_df.groupby("league"):
        scoreline_pools[league] = {}
        for result, result_df in league_df.groupby("match_result"):
            scoreline_pools[league][int(result)] = [
                (int(row["home_goals"]), int(row["away_goals"]))
                for _, row in result_df.iterrows()
            ]

    overall_pool = {}
    for result, result_df in played_df.groupby("match_result"):
        overall_pool[int(result)] = [
            (int(row["home_goals"]), int(row["away_goals"]))
            for _, row in result_df.iterrows()
        ]
    return scoreline_pools, overall_pool


def sample_scoreline(league_name, sampled_result, scoreline_pools, overall_pool, rng):
    """Sample a plausible scoreline conditional on the chosen outcome."""
    league_pool = scoreline_pools.get(league_name, {})
    result_pool = league_pool.get(int(sampled_result), [])

    if not result_pool:
        result_pool = overall_pool.get(int(sampled_result), [])

    if not result_pool:
        return DEFAULT_SCORELINES[int(sampled_result)]

    sampled_index = int(rng.integers(len(result_pool)))
    return result_pool[sampled_index]


def get_league_metadata_value(played_fixtures, unplayed_fixtures, column_name):
    """Return one metadata value from either played or unplayed league rows."""
    source_df = played_fixtures if not played_fixtures.empty else unplayed_fixtures
    return source_df[column_name].iloc[0]


def get_european_places(league_name, team_count):
    """Return a simple league-specific count of European qualification places."""
    return min(EUROPEAN_PLACES_BY_LEAGUE.get(league_name, TOP_4_CUTOFF), team_count)


def build_current_table_state(played_fixtures, teams):
    """Create the current points and goal-difference state from completed matches."""
    points = pd.Series(0, index=teams, dtype=int)
    goal_difference = pd.Series(0, index=teams, dtype=int)
    goals_for = pd.Series(0, index=teams, dtype=int)

    for _, match in played_fixtures.iterrows():
        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])

        goals_for.loc[match["home_team"]] += home_goals
        goals_for.loc[match["away_team"]] += away_goals
        goal_difference.loc[match["home_team"]] += home_goals - away_goals
        goal_difference.loc[match["away_team"]] += away_goals - home_goals

        if match["match_result"] == 0:
            points.loc[match["home_team"]] += 3
        elif match["match_result"] == 1:
            points.loc[match["home_team"]] += 1
            points.loc[match["away_team"]] += 1
        else:
            points.loc[match["away_team"]] += 3

    return points.to_numpy(), goal_difference.to_numpy(), goals_for.to_numpy()


def simulate_one_league(
    played_fixtures,
    unplayed_fixtures,
    probabilities,
    scoreline_pools,
    overall_scoreline_pool,
    rng,
):
    """Run Monte Carlo simulations for one league."""
    teams = sorted(
        set(played_fixtures["home_team"])
        .union(set(played_fixtures["away_team"]))
        .union(set(unplayed_fixtures["home_team"]))
        .union(set(unplayed_fixtures["away_team"]))
    )
    team_to_index = {team: index for index, team in enumerate(teams)}
    base_points, base_goal_difference, base_goals_for = build_current_table_state(played_fixtures, teams)

    position_counts = np.zeros((len(teams), len(teams)), dtype=int)
    expected_position_sum = np.zeros(len(teams), dtype=float)
    league_winner_counts = np.zeros(len(teams), dtype=int)
    top_4_counts = np.zeros(len(teams), dtype=int)
    european_counts = np.zeros(len(teams), dtype=int)
    relegation_counts = np.zeros(len(teams), dtype=int)

    home_indices = unplayed_fixtures["home_team"].map(team_to_index).to_numpy()
    away_indices = unplayed_fixtures["away_team"].map(team_to_index).to_numpy()
    league_name = get_league_metadata_value(played_fixtures, unplayed_fixtures, "league")
    european_places = get_european_places(league_name, len(teams))

    for _ in range(NUM_SIMULATIONS):
        points = base_points.copy()
        goal_difference = base_goal_difference.copy()
        goals_for = base_goals_for.copy()

        for match_index in range(len(unplayed_fixtures)):
            result = int(rng.choice([0, 1, 2], p=probabilities[match_index]))
            home_index = home_indices[match_index]
            away_index = away_indices[match_index]
            home_goals, away_goals = sample_scoreline(
                league_name=league_name,
                sampled_result=result,
                scoreline_pools=scoreline_pools,
                overall_pool=overall_scoreline_pool,
                rng=rng,
            )

            goals_for[home_index] += home_goals
            goals_for[away_index] += away_goals
            goal_difference[home_index] += home_goals - away_goals
            goal_difference[away_index] += away_goals - home_goals

            if result == 0:
                points[home_index] += 3
            elif result == 1:
                points[home_index] += 1
                points[away_index] += 1
            else:
                points[away_index] += 3

        ranking = sorted(
            range(len(teams)),
            key=lambda index: (
                -points[index],
                -goal_difference[index],
                -goals_for[index],
                teams[index],
            ),
        )

        for position, team_index in enumerate(ranking, start=1):
            position_counts[team_index, position - 1] += 1
            expected_position_sum[team_index] += position

        league_winner_counts[ranking[0]] += 1

        top_n = min(TOP_4_CUTOFF, len(teams))
        relegation_n = min(RELEGATION_SPOTS, len(teams))

        for team_index in ranking[:top_n]:
            top_4_counts[team_index] += 1
        for team_index in ranking[:european_places]:
            european_counts[team_index] += 1
        for team_index in ranking[-relegation_n:]:
            relegation_counts[team_index] += 1

    summary_rows = []
    position_rows = []

    for team_index, team in enumerate(teams):
        summary_rows.append(
            {
                "league": league_name,
                "season": get_league_metadata_value(played_fixtures, unplayed_fixtures, "season"),
                "team": team,
                "expected_finishing_position": expected_position_sum[team_index] / NUM_SIMULATIONS,
                "probability_of_winning_league": league_winner_counts[team_index] / NUM_SIMULATIONS,
                "probability_of_finishing_top_4": top_4_counts[team_index] / NUM_SIMULATIONS,
                "probability_of_qualifying_for_europe": european_counts[team_index] / NUM_SIMULATIONS,
                "probability_of_relegation": relegation_counts[team_index] / NUM_SIMULATIONS,
                "current_points": int(base_points[team_index]),
                "current_goal_difference": int(base_goal_difference[team_index]),
            }
        )

        for position in range(1, len(teams) + 1):
            position_rows.append(
                {
                    "league": league_name,
                    "season": get_league_metadata_value(played_fixtures, unplayed_fixtures, "season"),
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


def run_simulations(features_df, model, preprocessor):
    """Run table simulations for each league in the target season."""
    played_df, unplayed_df = get_target_fixtures(features_df, TARGET_SEASON)

    if played_df.empty and unplayed_df.empty:
        raise ValueError(f"No fixtures found for season {TARGET_SEASON}.")

    rng = np.random.default_rng(seed=42)
    summary_frames = []
    position_frames = []
    scoreline_pools, overall_scoreline_pool = build_scoreline_pools(features_df)

    season_df = pd.concat([played_df, unplayed_df], ignore_index=True)

    for _, league_fixtures in season_df.groupby("league"):
        league_fixtures = league_fixtures.sort_values("date").reset_index(drop=True)
        league_played = league_fixtures[league_fixtures["match_result"].notna()].copy()
        league_unplayed = league_fixtures[league_fixtures["match_result"].isna()].copy()
        probabilities = get_fixture_probabilities(model, league_unplayed, preprocessor)
        league_summary, league_positions = simulate_one_league(
            league_played,
            league_unplayed,
            probabilities,
            scoreline_pools,
            overall_scoreline_pool,
            rng,
        )
        summary_frames.append(league_summary)
        position_frames.append(league_positions)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    position_df = pd.concat(position_frames, ignore_index=True)
    return summary_df, position_df


def main():
    features_df, model, preprocessor = load_simulation_inputs()
    summary_df, position_df = run_simulations(features_df, model, preprocessor)

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
