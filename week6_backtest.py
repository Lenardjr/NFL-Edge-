"""Leakage-safe full-slate retrospective harness for NFL Edge Game Engine 1.0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from game_engine import NFLEdgeModel, load_games, pregame_features
from live_game import GAMES_URL


def run_week_retrospective(season: int = 2026, week: int = 6, simulations: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Replay a completed NFL week using only information available before each game.

    Each matchup gets its own training cutoff at gameday. Actual scores are hidden
    from model fitting/prediction and are joined only after the forecast is made.
    This supports Thursday/international/Sunday/Monday games without letting an
    earlier result from the target week leak into a later target's pregame state.
    """
    games = load_games(GAMES_URL)
    features = pregame_features(games)
    slate = features[(pd.to_numeric(features["season"], errors="coerce") == int(season)) & (pd.to_numeric(features["week"], errors="coerce") == int(week))].copy()
    if slate.empty:
        return pd.DataFrame()

    rows = []
    for i, target in slate.sort_values("gameday").iterrows():
        target_date = pd.to_datetime(target.get("gameday"), errors="coerce")
        if pd.isna(target_date):
            continue
        training = games[
            (pd.to_datetime(games["gameday"], errors="coerce") < target_date)
            & pd.to_numeric(games["home_score"], errors="coerce").notna()
            & pd.to_numeric(games["away_score"], errors="coerce").notna()
        ].copy()
        if training.empty:
            continue
        model = NFLEdgeModel().fit(training)
        pred = model.predict(target, n=int(simulations), seed=int(seed) + int(len(rows)))
        actual_home = pd.to_numeric(target.get("home_score"), errors="coerce")
        actual_away = pd.to_numeric(target.get("away_score"), errors="coerce")
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
    }


if __name__ == "__main__":
    result = run_week_retrospective()
    print(result.to_string(index=False))
    print(summarize_week(result))
