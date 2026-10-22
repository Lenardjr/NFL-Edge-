"""Diagnostics for developing NFL Edge Game Engine 1.1 against frozen Engine 1.0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from week6_backtest import run_week_retrospective


def diagnose_week6(simulations: int = 20000, seed: int = 42) -> dict:
    """Explain where frozen Engine 1.0 succeeded/failed without changing predictions."""
    games = run_week_retrospective(2026, 6, simulations=simulations, seed=seed)
    if games.empty:
        raise RuntimeError("Week 6 retrospective returned no games")

    d = games.copy()
    d["model_pick"] = np.where(d["projected_margin"] > 0, d["home_team"], d["away_team"])
    d["actual_winner"] = np.where(d["actual_margin"] > 0, d["home_team"], d["away_team"])
    d["model_confidence"] = np.maximum(d["home_win_probability"], 1.0 - d["home_win_probability"])
    d["projected_abs_margin"] = d["projected_margin"].abs()
    d["actual_abs_margin"] = d["actual_margin"].abs()
    # nflverse spread_line is already the expected HOME margin:
    # positive = home favored, negative = away favored.
    d["market_home_margin"] = pd.to_numeric(d["market_spread"], errors="coerce")
    d["market_pick"] = np.where(d["market_home_margin"] > 0, d["home_team"], d["away_team"])
    d["model_market_margin_gap"] = d["projected_margin"] - d["market_home_margin"]
    d["model_market_disagree"] = d["model_pick"] != d["market_pick"]
    d["actual_upset_vs_market"] = d["actual_winner"] != d["market_pick"]
    d["model_called_market_upset"] = d["model_pick"] != d["market_pick"]
    d["total_bias"] = d["projected_total"] - d["actual_total"]

    misses = d[~d["winner_correct"]].copy().sort_values("model_confidence", ascending=False)
    hits = d[d["winner_correct"]].copy().sort_values("model_confidence", ascending=False)

    confidence_bands = pd.cut(
        d["model_confidence"],
        bins=[0.0, 0.55, 0.65, 0.75, 0.85, 1.0],
        labels=["50-55%", "55-65%", "65-75%", "75-85%", "85%+"],
        include_lowest=True,
    )
    confidence = (
        d.assign(confidence_band=confidence_bands)
        .groupby("confidence_band", observed=True)
        .agg(games=("winner_correct", "size"), accuracy=("winner_correct", "mean"), margin_mae=("margin_error", "mean"))
        .reset_index()
    )

    market = {
        "games_with_line": int(d["market_spread"].notna().sum()),
        "model_market_disagreements": int(d["model_market_disagree"].sum()),
        "disagreement_accuracy": float(d.loc[d["model_market_disagree"], "winner_correct"].mean()) if d["model_market_disagree"].any() else np.nan,
        "market_upsets": int(d["actual_upset_vs_market"].sum()),
        "model_called_market_upsets": int(d["model_called_market_upset"].sum()),
        "successful_upset_calls": int((d["model_called_market_upset"] & d["winner_correct"]).sum()),
        "mean_abs_model_market_margin_gap": float(d["model_market_margin_gap"].abs().mean()),
    }

    summary = {
        "games": int(len(d)),
        "hits": int(d["winner_correct"].sum()),
        "misses": int((~d["winner_correct"]).sum()),
        "high_confidence_misses_65_plus": int(((~d["winner_correct"]) & (d["model_confidence"] >= 0.65)).sum()),
        "one_score_games_actual": int((d["actual_abs_margin"] <= 8).sum()),
        "one_score_game_accuracy": float(d.loc[d["actual_abs_margin"] <= 8, "winner_correct"].mean()),
        "projected_close_games_3_or_less": int((d["projected_abs_margin"] <= 3).sum()),
        "projected_close_accuracy": float(d.loc[d["projected_abs_margin"] <= 3, "winner_correct"].mean()) if (d["projected_abs_margin"] <= 3).any() else np.nan,
        "mean_total_bias": float(d["total_bias"].mean()),
        "median_total_bias": float(d["total_bias"].median()),
    }

    cols = [
        "gameday", "away_team", "home_team", "model_pick", "actual_winner",
        "home_win_probability", "model_confidence", "projected_margin", "actual_margin",
        "margin_error", "market_spread", "market_pick", "model_market_margin_gap",
        "projected_total", "actual_total", "total_error",
    ]
    return {
        "summary": summary,
        "market": market,
        "confidence_bands": confidence,
        "misses": misses[cols].reset_index(drop=True),
        "hits": hits[cols].reset_index(drop=True),
    }


if __name__ == "__main__":
    report = diagnose_week6()
    print("SUMMARY")
    print(report["summary"])
    print("\nMARKET DIAGNOSTICS")
    print(report["market"])
    print("\nCONFIDENCE BANDS")
    print(report["confidence_bands"].to_string(index=False))
    print("\nWINNER MISSES")
    print(report["misses"].to_string(index=False))
