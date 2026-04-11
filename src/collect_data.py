"""
Collect historical match results and xG data from Understat for the Top 5
European leagues. The script combines all seasons into one file and saves it
to data/raw/matches_raw.csv using simple snake_case column names.
"""

import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from config import RAW_MATCHES_RAW_FILE


LEAGUES = {
    "EPL": "premier_league",
    "La_Liga": "la_liga",
    "Bundesliga": "bundesliga",
    "Serie_A": "serie_a",
    "Ligue_1": "ligue_1",
}
SEASONS = range(2014, 2025)


def safe_float(value):
    """Convert a value to float, or return NaN if conversion fails."""
    try:
        if value in (None, ""):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_int(value):
    """Convert a value to integer, or return NaN if conversion fails."""
    try:
        if value in (None, ""):
            return np.nan
        return int(value)
    except (TypeError, ValueError):
        return np.nan


def get_nested_value(record, *keys):
    """
    Read values safely from nested Understat records.

    This handles both nested dictionaries such as:
    record["h"]["title"]

    and flat keys such as:
    record["goals"]["h"]
    """
    current_value = record
    for key in keys:
        if not isinstance(current_value, dict):
            return None
        current_value = current_value.get(key)
        if current_value is None:
            return None
    return current_value


def normalise_match(match, league_name, season):
    """Convert one Understat match record into the project format."""
    date_value = match.get("datetime") or match.get("date")
    date_value = pd.to_datetime(date_value, errors="coerce")

    return {
        "league": league_name,
        "season": season,
        "date": date_value,
        "home_team": get_nested_value(match, "h", "title"),
        "away_team": get_nested_value(match, "a", "title"),
        "home_goals": safe_int(get_nested_value(match, "goals", "h")),
        "away_goals": safe_int(get_nested_value(match, "goals", "a")),
        "home_xg": safe_float(get_nested_value(match, "xG", "h")),
        "away_xg": safe_float(get_nested_value(match, "xG", "a")),
    }


def fetch_league_season_matches(client, league_code, league_name, season):
    """Download all matches for one league and season."""
    season_matches = client.league(league=league_code).get_match_data(season=str(season))
    return [normalise_match(match, league_name, season) for match in season_matches]


def collect_all_matches():
    """Collect match data across all required leagues and seasons."""
    all_matches = []

    with UnderstatClient() as understat:
        for league_code, league_name in LEAGUES.items():
            for season in SEASONS:
                print(f"Downloading {league_name} season {season}...")
                try:
                    season_matches = fetch_league_season_matches(
                        client=understat,
                        league_code=league_code,
                        league_name=league_name,
                        season=season,
                    )
                    all_matches.extend(season_matches)
                except Exception as error:
                    print(f"Skipped {league_name} {season} because of an error: {error}")

    return pd.DataFrame(all_matches)


def clean_final_dataframe(df):
    """Apply final cleaning before saving."""
    df = df.copy()

    ordered_columns = [
        "league",
        "season",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
    ]
    df = df[ordered_columns]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep rows even if xG is missing, but remove rows missing core match details.
    df = df.dropna(subset=["league", "season", "date", "home_team", "away_team"])

    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df["home_xg"] = pd.to_numeric(df["home_xg"], errors="coerce")
    df["away_xg"] = pd.to_numeric(df["away_xg"], errors="coerce")

    df = df.sort_values(["league", "season", "date", "home_team"]).reset_index(drop=True)
    return df


def main():
    RAW_MATCHES_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)

    matches_df = collect_all_matches()
    matches_df = clean_final_dataframe(matches_df)
    matches_df.to_csv(RAW_MATCHES_RAW_FILE, index=False)

    print(f"Saved {len(matches_df)} matches to: {RAW_MATCHES_RAW_FILE}")
    print("Missing xG values are kept as empty cells in the CSV.")


if __name__ == "__main__":
    main()
