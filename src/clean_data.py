import pandas as pd

from config import CLEAN_MATCHES_FILE, RAW_MATCHES_RAW_FILE


TEAM_NAME_FIXES = {
    "Bayern M\u00fcnchen": "Bayern Munich",
    "Internazionale": "Inter",
    "Paris Saint Germain": "Paris Saint-Germain",
    "Spal": "SPAL",
}


def standardise_team_name(team_name):
    """Clean a team name and apply simple replacements when needed."""
    if pd.isna(team_name):
        return team_name

    cleaned_name = str(team_name).strip()
    cleaned_name = " ".join(cleaned_name.split())
    return TEAM_NAME_FIXES.get(cleaned_name, cleaned_name)


def create_match_result(home_goals, away_goals):
    """Create the target label: 0 home win, 1 draw, 2 away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def clean_matches(df):
    """Clean the raw match data and create the final target column."""
    df = df.copy()

    required_columns = [
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
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["home_team"] = df["home_team"].apply(standardise_team_name)
    df["away_team"] = df["away_team"].apply(standardise_team_name)

    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df["home_xg"] = pd.to_numeric(df["home_xg"], errors="coerce")
    df["away_xg"] = pd.to_numeric(df["away_xg"], errors="coerce")

    df = df.dropna(
        subset=[
            "league",
            "season",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    )

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["match_result"] = df.apply(
        lambda row: create_match_result(row["home_goals"], row["away_goals"]),
        axis=1,
    )

    df = df.drop_duplicates()
    df = df.sort_values("date").reset_index(drop=True)

    return df


def main():
    df = pd.read_csv(RAW_MATCHES_RAW_FILE)
    clean_df = clean_matches(df)

    CLEAN_MATCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_MATCHES_FILE, index=False)

    print(f"Cleaned data saved to: {CLEAN_MATCHES_FILE}")
    print(f"Number of rows: {len(clean_df)}")
    print(f"Number of missing values: {int(clean_df.isna().sum().sum())}")
    print("Class distribution of match_result:")
    print(clean_df["match_result"].value_counts().sort_index())


if __name__ == "__main__":
    main()
