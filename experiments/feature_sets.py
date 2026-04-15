from collections import deque

import pandas as pd

from common import get_base_numeric_columns, load_base_features, load_clean_matches


KEY_COLUMNS = ["league", "season", "date", "home_team", "away_team"]
ADDITIONAL_NUMERIC_COLUMNS = [
    "home_goals_conceded_avg_last_5",
    "away_goals_conceded_avg_last_5",
    "home_xg_conceded_avg_last_5",
    "away_xg_conceded_avg_last_5",
    "home_home_points_avg_last_5",
    "away_away_points_avg_last_5",
    "home_home_xg_avg_last_5",
    "away_away_xg_avg_last_5",
    "home_missing_xg_rate_last_5",
    "away_missing_xg_rate_last_5",
    "head_to_head_home_points_avg_last_3",
    "head_to_head_home_xg_diff_avg_last_3",
    "head_to_head_matches_seen",
]


def safe_average(values):
    """Average non-missing values, or return 0 when no usable history exists."""
    clean_values = [value for value in values if pd.notna(value)]
    if not clean_values:
        return 0.0
    return float(sum(clean_values) / len(clean_values))


def create_team_history():
    """Create rolling storage for extra experiment-only team features."""
    return {
        "goals_conceded": deque(maxlen=5),
        "xg_conceded": deque(maxlen=5),
        "home_points": deque(maxlen=5),
        "away_points": deque(maxlen=5),
        "home_xg": deque(maxlen=5),
        "away_xg": deque(maxlen=5),
        "missing_xg": deque(maxlen=5),
    }


def get_pair_key(team_a, team_b):
    """Create a stable key for an unordered team pair."""
    return tuple(sorted([team_a, team_b]))


def get_head_to_head_features(pair_history, current_home_team):
    """Build head-to-head features from the current home team's perspective."""
    home_points = []
    home_xg_diffs = []

    for match in pair_history:
        if match["home_team"] == current_home_team:
            home_points.append(match["home_points"])
            home_xg_diffs.append(match["home_xg_diff"])
        else:
            home_points.append(match["away_points"])
            home_xg_diffs.append(-match["home_xg_diff"] if pd.notna(match["home_xg_diff"]) else pd.NA)

    return {
        "head_to_head_home_points_avg_last_3": safe_average(home_points),
        "head_to_head_home_xg_diff_avg_last_3": safe_average(home_xg_diffs),
        "head_to_head_matches_seen": float(len(pair_history)),
    }


def build_additional_context_features(clean_df):
    """Create extra features that are feasible with the current raw data."""
    clean_df = clean_df.copy()
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    clean_df = clean_df.sort_values("date").reset_index(drop=True)

    team_histories = {}
    pair_histories = {}
    rows = []

    for _, match in clean_df.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        if home_team not in team_histories:
            team_histories[home_team] = create_team_history()
        if away_team not in team_histories:
            team_histories[away_team] = create_team_history()

        pair_key = get_pair_key(home_team, away_team)
        if pair_key not in pair_histories:
            pair_histories[pair_key] = deque(maxlen=3)

        home_history = team_histories[home_team]
        away_history = team_histories[away_team]
        pair_history = pair_histories[pair_key]

        feature_row = {
            "league": match["league"],
            "season": match["season"],
            "date": match["date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_goals_conceded_avg_last_5": safe_average(home_history["goals_conceded"]),
            "away_goals_conceded_avg_last_5": safe_average(away_history["goals_conceded"]),
            "home_xg_conceded_avg_last_5": safe_average(home_history["xg_conceded"]),
            "away_xg_conceded_avg_last_5": safe_average(away_history["xg_conceded"]),
            "home_home_points_avg_last_5": safe_average(home_history["home_points"]),
            "away_away_points_avg_last_5": safe_average(away_history["away_points"]),
            "home_home_xg_avg_last_5": safe_average(home_history["home_xg"]),
            "away_away_xg_avg_last_5": safe_average(away_history["away_xg"]),
            "home_missing_xg_rate_last_5": safe_average(home_history["missing_xg"]),
            "away_missing_xg_rate_last_5": safe_average(away_history["missing_xg"]),
            **get_head_to_head_features(pair_history, home_team),
        }
        rows.append(feature_row)

        if pd.notna(match["match_result"]):
            home_points = 3 if match["match_result"] == 0 else 1 if match["match_result"] == 1 else 0
            away_points = 3 if match["match_result"] == 2 else 1 if match["match_result"] == 1 else 0
            home_xg_missing = float(pd.isna(match["home_xg"]))
            away_xg_missing = float(pd.isna(match["away_xg"]))
            home_xg_diff = (
                float(match["home_xg"] - match["away_xg"])
                if pd.notna(match["home_xg"]) and pd.notna(match["away_xg"])
                else pd.NA
            )

            home_history["goals_conceded"].append(match["away_goals"])
            home_history["xg_conceded"].append(match["away_xg"])
            home_history["home_points"].append(home_points)
            home_history["home_xg"].append(match["home_xg"])
            home_history["missing_xg"].append(home_xg_missing)

            away_history["goals_conceded"].append(match["home_goals"])
            away_history["xg_conceded"].append(match["home_xg"])
            away_history["away_points"].append(away_points)
            away_history["away_xg"].append(match["away_xg"])
            away_history["missing_xg"].append(away_xg_missing)

            pair_history.append(
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_points": home_points,
                    "away_points": away_points,
                    "home_xg_diff": home_xg_diff,
                }
            )

    return pd.DataFrame(rows)


def build_expanded_feature_frame():
    """Merge extra context features onto the existing engineered feature frame."""
    base_df = load_base_features()
    extra_df = build_additional_context_features(load_clean_matches())
    expanded_df = base_df.merge(extra_df, on=KEY_COLUMNS, how="left", validate="one_to_one")
    return expanded_df.sort_values("date").reset_index(drop=True)


def get_expanded_numeric_columns():
    """Return the combined numeric feature set used in richer experiments."""
    return get_base_numeric_columns() + list(ADDITIONAL_NUMERIC_COLUMNS)
