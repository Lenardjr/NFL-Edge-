"""Strict pre-week retrospective harness for NFL Edge Game Engine 1.0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from game_engine import NFLEdgeModel, load_games, pregame_features
from live_game import GAMES_URL


def _preweek_mask(df: pd.DataFrame, season: int, week: int) -> pd.Series:
    seasons = pd.to_numeric(df["season"], errors="coerce")
    weeks = pd.to_numeric(df["week"], errors="coerce")
    return (seasons < int(season)) | ((seasons == int(season)) & (weeks < int(week)))


def _hide_target_week_and_future_scores(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Return a replay frame whose state cannot learn target-week/future outcomes."""
    replay = games.copy()
    preweek = _preweek_mask(replay, season, week)
    replay.loc[~preweek, ["home_score", "away_score"]] = np.nan
    return replay


def run_week_retrospective(season: int = 2026, week: int = 6, simulations: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Predict an entire completed week from one strict pre-week information freeze.

    Game Engine 1.0 is trained once on completed games strictly before the target
    week. Target-week and future scores are hidden before stateful pregame features
    are built, so Thursday/Sunday/Monday results cannot influence one another.
    Actual scores are read only from the untouched source frame after predictions.
    """
    games = load_games(GAMES_URL)
    actuals = games.copy()

    replay_games = _hide_target_week_and_future_scores(games, season, week)
    features = pregame_features(replay_games)
    seasons = pd.to_numeric(features["season"], errors="coerce")
    weeks = pd.to_numeric(features["week"], errors="coerce")
    slate = features[(seasons == int(season)) & (weeks == int(week))].copy()
    if slate.empty:
        return pd.DataFrame()

    completed = (
        pd.to_numeric(actuals["home_score"], errors="coerce").notna()
        & pd.to_numeric(actuals["away_score"], errors="coerce").notna()
    )
    training = actuals[_preweek_mask(actuals, season, week) & completed].copy()
    if training.empty:
        return pd.DataFrame()

    model = NFLEdgeModel().fit(training)

    actual_lookup = actuals.copy()
    actual_lookup["_gameday"] = pd.to_datetime(actual_lookup["gameday"], errors="coerce")
    actual_lookup["_season"] = pd.to_numeric(actual_lookup["season"], errors="coerce")
    actual_lookup["_week"] = pd.to_numeric(actual_lookup["week"], errors="coerce")

    rows = []
    for _, target in slate.sort_values(["gameday", "away_team", "home_team"]).iterrows():
        target_date = pd.to_datetime(target.get("gameday"), errors="coerce")
        if pd.isna(target_date):
            continue

        pred = model.predict(target, n=int(simulations), seed=int(seed) + int(len(rows)))

        match = actual_lookup[
            (actual_lookup["_season"] == int(season))
            & (actual_lookup["_week"] == int(week))
            & (actual_lookup["_gameday"] == target_date)
            & (actual_lookup["home_team"].astype(str) == str(target["home_team"]))
            & (actual_lookup["away_team"].astype(str) == str(target["away_team"]))
        ]
        if match.empty:
            continue
        actual = match.iloc[0]
        actual_home = pd.to_numeric(actual.get("home_score"), errors="coerce")
        actual_away = pd.to_numeric(actual.get("away_score"), errors="coerce")
        if pd.isna(actual_home) or pd.isna(actual_away):
            continue

        actual_margin = float(actual_home - actual_away)
        actual_total = float(actual_home + actual_away)
        model_margin = float(pred["margin"])
        model_total = float(pred["total"])
        model_home_pick = model_margin > 0
        actual_home_win = actual_margin > 0
        spread = pd.to_numeric(target.get("spread_line"), errors="coerce")
        total_line = pd.to_numeric(target.get("total_line"), errors="coerce")
        rows.append({
            "gameday": target_date.strftime("%Y-%m-%d"),
            "away_team": str(target["away_team"]),
            "home_team": str(target["home_team"]),
            "actual_away": float(actual_away),
            "actual_home": float(actual_home),
            "projected_away": float(pred["away_score"]),
            "projected_home": float(pred["home_score"]),
            "projected_margin": model_margin,
            "actual_margin": actual_margin,
            "margin_error": abs(model_margin - actual_margin),
            "projected_total": model_total,
            "actual_total": actual_total,
            "total_error": abs(model_total - actual_total),
            "home_win_probability": float(pred["home_win_prob"]),
            "winner_correct": bool(model_home_pick == actual_home_win),
            "market_spread": float(spread) if pd.notna(spread) else np.nan,
            "market_total": float(total_line) if pd.notna(total_line) else np.nan,
            "home_cover_probability": float(pred["home_cover_prob"]) if "home_cover_prob" in pred else np.nan,
            "over_probability": float(pred["over_prob"]) if "over_prob" in pred else np.nan,
            "upset_probability": float(pred["upset_prob"]) if "upset_prob" in pred else np.nan,
            "upset_score": float(pred["upset_score"]) if "upset_score" in pred else np.nan,
            "training_games": int(len(training)),
            "validation_mode": "STRICT_PREWEEK_FREEZE",
        })
    return pd.DataFrame(rows)


def summarize_week(results: pd.DataFrame) -> dict:
    if results is None or results.empty:
        return {"games": 0}
    d = results.copy()
    probs = d["home_win_probability"].clip(0.001, 0.999)
    outcomes = (d["actual_margin"] > 0).astype(float)
    return {
        "games": int(len(d)),
        "winner_accuracy": float(d["winner_correct"].mean()),
        "margin_mae": float(d["margin_error"].mean()),
        "total_mae": float(d["total_error"].mean()),
        "brier_score": float(np.mean((probs - outcomes) ** 2)),
        "avg_projected_total": float(d["projected_total"].mean()),
        "avg_actual_total": float(d["actual_total"].mean()),
        "validation_mode": str(d["validation_mode"].iloc[0]) if "validation_mode" in d.columns else "UNKNOWN",
        "training_games": int(d["training_games"].iloc[0]) if "training_games" in d.columns else 0,
    }


if __name__ == "__main__":
    result = run_week_retrospective()
    print(result.to_string(index=False))
    print(summarize_week(result))
