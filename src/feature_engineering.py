from collections import deque

import pandas as pd

from config import CLEAN_MATCHES_FILE, FEATURES_FILE


INITIAL_ELO = 1500
ELO_K_FACTOR = 20
DEFAULT_REST_DAYS = 7


def create_team_history():
    """Create storage for a team's previous matches."""
    return {
        "points": deque(maxlen=5),
        "goals_scored": deque(maxlen=5),
        "xg": deque(maxlen=5),
        "graph_edge_weights": deque(maxlen=5),
        "elo": INITIAL_ELO,
        "last_match_date": None,
    }


def get_average(values):
    """Return the average of a list-like object, or 0 when empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def get_rest_days(last_match_date, current_match_date):
    """
    Calculate rest days before a match.
    If a team has no previous match yet, use a simple default value.
    """
    if last_match_date is None:
        return DEFAULT_REST_DAYS
    return max((current_match_date - last_match_date).days, 0)


def expected_elo_score(team_elo, opponent_elo):
    """Expected result from Elo ratings."""
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def create_match_graph():
    """Create a simple directed weighted graph of historical team matchups."""
    return {}


def get_graph_edge_weight(home_xg, away_xg):
    """
    Return the directed edge weight used by the match graph.

    The edge weight is the xG differential from the source team's perspective.
    """
    if pd.isna(home_xg) or pd.isna(away_xg):
        return 0.0
    return float(home_xg) - float(away_xg)


def add_directed_edge(match_graph, source_team, target_team, match_date, edge_weight):
    """Store one directed edge in the graph."""
    if source_team not in match_graph:
        match_graph[source_team] = deque(maxlen=25)

    match_graph[source_team].append(
        {
            "opponent": target_team,
            "date": match_date,
            "weight": edge_weight,
        }
    )


def register_match_graph_edges(match_graph, home_team, away_team, match_date, home_xg, away_xg):
    """Add the finished match to the directed weighted graph."""
    home_edge_weight = get_graph_edge_weight(home_xg, away_xg)
    away_edge_weight = -home_edge_weight

    add_directed_edge(match_graph, home_team, away_team, match_date, home_edge_weight)
    add_directed_edge(match_graph, away_team, home_team, match_date, away_edge_weight)
    return home_edge_weight, away_edge_weight


def get_elo_weight(edge_weight):
    """Scale Elo updates using the graph edge weight from xG differential."""
    return 1.0 + min(abs(edge_weight), 3.0)


def update_elo_ratings(home_elo, away_elo, match_result, edge_weight):
    """Update Elo ratings after the match result is known."""
    if match_result == 0:
        home_score = 1.0
        away_score = 0.0
    elif match_result == 1:
        home_score = 0.5
        away_score = 0.5
    else:
        home_score = 0.0
        away_score = 1.0

    home_expected = expected_elo_score(home_elo, away_elo)
    away_expected = expected_elo_score(away_elo, home_elo)
    elo_weight = get_elo_weight(edge_weight)

    new_home_elo = home_elo + ELO_K_FACTOR * elo_weight * (home_score - home_expected)
    new_away_elo = away_elo + ELO_K_FACTOR * elo_weight * (away_score - away_expected)
    return new_home_elo, new_away_elo


def get_team_features(team_data, current_match_date, team_prefix):
    """Build the pre-match features for one team."""
    features = {
        f"{team_prefix}_avg_points_last_5": get_average(team_data["points"]),
        f"{team_prefix}_goals_scored_avg_last_5": get_average(team_data["goals_scored"]),
        f"{team_prefix}_xg_avg_last_5": get_average(team_data["xg"]),
        f"{team_prefix}_graph_xg_diff_avg_last_5": get_average(team_data["graph_edge_weights"]),
        f"{team_prefix}_elo": team_data["elo"],
        f"{team_prefix}_rest_days": get_rest_days(team_data["last_match_date"], current_match_date),
    }
    return features


def update_team_history(team_data, points, goals_scored, xg_value, edge_weight, match_date, new_elo):
    """Store the finished match so future rows can use it."""
    team_data["points"].append(points)
    team_data["goals_scored"].append(goals_scored)
    team_data["xg"].append(xg_value)
    team_data["graph_edge_weights"].append(edge_weight)
    team_data["last_match_date"] = match_date
    team_data["elo"] = new_elo


def build_match_features(df):
    """
    Process matches in date order.
    Features are created first, then team history is updated after the match,
    which prevents data leakage.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    team_histories = {}
    match_graph = create_match_graph()
    feature_rows = []

    for _, match in df.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        match_date = match["date"]

        if home_team not in team_histories:
            team_histories[home_team] = create_team_history()
        if away_team not in team_histories:
            team_histories[away_team] = create_team_history()

        home_history = team_histories[home_team]
        away_history = team_histories[away_team]

        home_features = get_team_features(home_history, match_date, "home")
        away_features = get_team_features(away_history, match_date, "away")

        feature_row = {
            "league": match["league"],
            "season": match["season"],
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "is_played": match["is_played"],
            "home_goals": match["home_goals"],
            "away_goals": match["away_goals"],
            "match_result": match["match_result"],
            **home_features,
            **away_features,
        }
        feature_row["elo_difference"] = feature_row["home_elo"] - feature_row["away_elo"]
        feature_rows.append(feature_row)

        # Scheduled fixtures should contribute features but must not update history.
        if pd.notna(match["match_result"]):
            home_edge_weight, away_edge_weight = register_match_graph_edges(
                match_graph=match_graph,
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                home_xg=match["home_xg"],
                away_xg=match["away_xg"],
            )
            home_points = 3 if match["match_result"] == 0 else 1 if match["match_result"] == 1 else 0
            away_points = 3 if match["match_result"] == 2 else 1 if match["match_result"] == 1 else 0

            new_home_elo, new_away_elo = update_elo_ratings(
                home_elo=home_history["elo"],
                away_elo=away_history["elo"],
                match_result=match["match_result"],
                edge_weight=home_edge_weight,
            )

            update_team_history(
                team_data=home_history,
                points=home_points,
                goals_scored=match["home_goals"],
                xg_value=match["home_xg"],
                edge_weight=home_edge_weight,
                match_date=match_date,
                new_elo=new_home_elo,
            )
            update_team_history(
                team_data=away_history,
                points=away_points,
                goals_scored=match["away_goals"],
                xg_value=match["away_xg"],
                edge_weight=away_edge_weight,
                match_date=match_date,
                new_elo=new_away_elo,
            )

    return pd.DataFrame(feature_rows)


def main():
    df = pd.read_csv(CLEAN_MATCHES_FILE)
    features_df = build_match_features(df)

    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(FEATURES_FILE, index=False)

    print(f"Feature data saved to: {FEATURES_FILE}")
    print(f"Number of rows: {len(features_df)}")


if __name__ == "__main__":
    main()
