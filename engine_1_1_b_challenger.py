"""NFL Edge Game Engine 1.1-B challenger.

Development-only challenger. It separates winner selection from margin estimation:
- winner pick remains the Engine 1.0 projected-margin winner signal;
- projected margin uses a conservative 70/30 model/market anchor when a line exists.

Week 6 is diagnostic/development data. Multi-week results are reported for robustness,
but this file does not promote a model. A future frozen holdout is required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from week6_backtest import run_week_retrospective

MODEL_WEIGHT = 0.70
MARKET_WEIGHT = 0.30
DEFAULT_WEEKS = (2, 3, 4, 5, 6)


def market_home_margin(spread: pd.Series) -> pd.Series:
    """Return nflverse spread_line as expected home-team margin.

    Positive = home favored; negative = away favored.
    """
    return pd.to_numeric(spread, errors="coerce")


def adjusted_margin(model_margin: pd.Series, spread: pd.Series) -> pd.Series:
    model = pd.to_numeric(model_margin, errors="coerce")
    market = market_home_margin(spread)
    blended = MODEL_WEIGHT * model + MARKET_WEIGHT * market
    return blended.where(market.notna(), model)


def evaluate_week(season: int, week: int, simulations: int = 10000, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    d = run_week_retrospective(season, week, simulations=simulations, seed=seed).copy()
    if d.empty:
        return {"season": season, "week": week, "games": 0}, d

    d["v10_margin"] = pd.to_numeric(d["projected_margin"], errors="coerce")
    d["v11b_margin"] = adjusted_margin(d["v10_margin"], d["market_spread"])
    d["actual_home_win"] = pd.to_numeric(d["actual_margin"], errors="coerce") > 0

    # Winner selection is deliberately frozen to Engine 1.0 semantics.
    # The challenger may improve margin estimation without silently changing picks.
    d["v11b_home_pick"] = d["v10_margin"] > 0
    d["v11b_winner_correct"] = d["v11b_home_pick"] == d["actual_home_win"]
    d["winner_definition_match"] = d["v11b_winner_correct"] == d["winner_correct"]
    d["v11b_margin_error"] = (d["v11b_margin"] - d["actual_margin"]).abs()
    d["v10_market_gap"] = (d["v10_margin"] - market_home_margin(d["market_spread"])).abs()
    d["v11b_market_gap"] = (d["v11b_margin"] - market_home_margin(d["market_spread"])).abs()

    if not bool(d["winner_definition_match"].all()):
        raise AssertionError("Engine 1.1-B winner definition diverged from frozen Engine 1.0")

    summary = {
        "season": int(season),
        "week": int(week),
        "games": int(len(d)),
        "v10_winner_accuracy": float(d["winner_correct"].mean()),
        "v11b_winner_accuracy": float(d["v11b_winner_correct"].mean()),
        "winner_definition_matches": bool(d["winner_definition_match"].all()),
        "v10_margin_mae": float(d["margin_error"].mean()),
        "v11b_margin_mae": float(d["v11b_margin_error"].mean()),
        "margin_mae_delta": float(d["v11b_margin_error"].mean() - d["margin_error"].mean()),
        "mean_v10_market_gap": float(d["v10_market_gap"].mean()),
        "mean_v11b_market_gap": float(d["v11b_market_gap"].mean()),
        "training_games": int(d["training_games"].iloc[0]),
    }
    return summary, d


def evaluate_multiweek(season: int = 2026, weeks: tuple[int, ...] = DEFAULT_WEEKS, simulations: int = 10000, seed: int = 42) -> dict:
    summaries = []
    frames = []
    for week in weeks:
        summary, frame = evaluate_week(season, int(week), simulations=simulations, seed=seed + int(week) * 100)
        summaries.append(summary)
        if not frame.empty:
            frame = frame.copy()
            frame["evaluation_week"] = int(week)
            frames.append(frame)

    usable = [s for s in summaries if s.get("games", 0) > 0]
    if not usable:
        raise RuntimeError("No completed weeks available for Engine 1.1-B evaluation")
    all_games = pd.concat(frames, ignore_index=True)
    aggregate = {
        "season": int(season),
        "weeks": [int(w) for w in weeks],
        "games": int(len(all_games)),
        "model_weight": MODEL_WEIGHT,
        "market_weight": MARKET_WEIGHT,
        "v10_winner_accuracy": float(all_games["winner_correct"].mean()),
        "v11b_winner_accuracy": float(all_games["v11b_winner_correct"].mean()),
        "winner_definition_matches": bool(all_games["winner_definition_match"].all()),
        "v10_margin_mae": float(all_games["margin_error"].mean()),
        "v11b_margin_mae": float(all_games["v11b_margin_error"].mean()),
        "margin_mae_delta": float(all_games["v11b_margin_error"].mean() - all_games["margin_error"].mean()),
        "weeks_margin_improved": int(sum(s["v11b_margin_mae"] < s["v10_margin_mae"] for s in usable)),
        "weeks_margin_worsened": int(sum(s["v11b_margin_mae"] > s["v10_margin_mae"] for s in usable)),
        "status": "EXPERIMENTAL_NOT_PROMOTED",
        "promotion_rule": "Requires a separately frozen holdout after challenger design is complete.",
    }
    return {"aggregate": aggregate, "by_week": pd.DataFrame(summaries), "results": all_games}


if __name__ == "__main__":
    report = evaluate_multiweek()
    print("ENGINE 1.1-B MULTI-WEEK SUMMARY")
    print(report["aggregate"])
    print("\nBY WEEK")
    print(report["by_week"].to_string(index=False))
