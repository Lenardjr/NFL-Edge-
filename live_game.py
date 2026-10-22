"""Live-game adapter joining nflverse schedules to protected Game Engine 1.0."""
from __future__ import annotations

import pandas as pd

from game_engine import NFLEdgeModel, load_games, pregame_features

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"


def build_live_game_forecast(
    season: int,
    week: int,
    away_team: str,
    home_team: str,
    simulations: int = 50000,
    seed: int = 42,
) -> dict:
    """Fit Game Engine 1.0 and forecast one scheduled game without postgame leakage.

    The feature row is built chronologically by the protected engine. Training is
    restricted to completed games before the target game's date. The target row
    may carry market lines for comparison, but they are not Game Engine features.
    """
    games = load_games(GAMES_URL)
    features = pregame_features(games)
    target = features[
        (pd.to_numeric(features["season"], errors="coerce") == int(season))
        & (pd.to_numeric(features["week"], errors="coerce") == int(week))
        & (features["away_team"].astype(str) == str(away_team))
        & (features["home_team"].astype(str) == str(home_team))
    ].copy()
    if target.empty:
        return {"ok": False, "error": f"No schedule row found for {away_team} @ {home_team}, {season} Week {week}."}

    row = target.iloc[0]
    target_date = pd.to_datetime(row.get("gameday"), errors="coerce")
    if pd.isna(target_date):
        return {"ok": False, "error": "Target game has no usable gameday."}

    # Prevent tonight's result (or any later result) from ever entering training.
    training_games = games[
        (pd.to_datetime(games["gameday"], errors="coerce") < target_date)
        & pd.to_numeric(games["home_score"], errors="coerce").notna()
        & pd.to_numeric(games["away_score"], errors="coerce").notna()
    ].copy()
    if training_games.empty:
        return {"ok": False, "error": "No completed pregame training sample was available."}

    model = NFLEdgeModel().fit(training_games)
    # Rebuild the target feature row from the complete chronological schedule so
    # current-season team state includes every completed game before kickoff.
    target_features = pregame_features(games)
    target_row = target_features[
        (pd.to_numeric(target_features["season"], errors="coerce") == int(season))
        & (pd.to_numeric(target_features["week"], errors="coerce") == int(week))
        & (target_features["away_team"].astype(str) == str(away_team))
        & (target_features["home_team"].astype(str) == str(home_team))
    ].iloc[0]
    pred = model.predict(target_row, n=int(simulations), seed=int(seed))

    result = {
        "ok": True,
        "season": int(season),
        "week": int(week),
        "away_team": str(away_team),
        "home_team": str(home_team),
        "gameday": target_date.strftime("%Y-%m-%d"),
        "training_games": int(len(training_games)),
        "away_score": float(pred["away_score"]),
        "home_score": float(pred["home_score"]),
        "home_margin": float(pred["margin"]),
        "total": float(pred["total"]),
        "home_win_prob": float(pred["home_win_prob"]),
        "sim_home_win_prob": float(pred["sim_home_win_prob"]),
        "calibrated_home_win_prob": float(pred["calibrated_home_win_prob"]),
        "market_spread": float(target_row["spread_line"]) if pd.notna(target_row.get("spread_line")) else None,
        "market_total": float(target_row["total_line"]) if pd.notna(target_row.get("total_line")) else None,
        "home_cover_prob": float(pred["home_cover_prob"]) if "home_cover_prob" in pred else None,
        "over_prob": float(pred["over_prob"]) if "over_prob" in pred else None,
        "upset_prob": float(pred["upset_prob"]) if "upset_prob" in pred else None,
        "upset_score": float(pred["upset_score"]) if "upset_score" in pred else None,
    }
    return result
