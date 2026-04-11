import pandas as pd


RESULT_MAP = {"H": 0, "D": 1, "A": 2}
REVERSE_RESULT_MAP = {0: "H", 1: "D", 2: "A"}


def get_match_result(home_goals, away_goals):
    """Return H, D, or A based on the final score."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def load_csv(path):
    """Simple helper to load CSV files."""
    return pd.read_csv(path)
