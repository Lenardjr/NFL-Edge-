"""NFL Edge Game Engine 1.1 challenger.

Experimental only. Engine 1.0 remains frozen. This challenger anchors the model's
margin to the pregame market when available, then compares matched Week 6 results.
No Week 6 outcomes are used to fit or choose the anchor weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from week6_backtest import run_week_retrospective

# Conservative first challenger: keep most of NFL Edge's independent signal while
# shrinking extreme disagreement toward a strong external prior.
MODEL_WEIGHT = 0.70
MARKET_WEIGHT = 0.30


def market_home_margin(market_spread: pd.Series) -> pd.Series:
    """Return nflverse spread_line as expected home-team margin.

    Positive = home favored; negative = away favored.
    """
    return pd.to_numeric(market_spread, errors="coerce")


def anchored_margin(model_margin: pd.Series, market_spread: pd.Series) -> pd.Series:
    model = pd.to_numeric(model_margin, errors="coerce")
    market = market_home_margin(market_spread)
    blended = MODEL_WEIGHT * model + MARKET_WEIGHT * market
    return blended.where(market.notna(), model)


def evaluate_challenger(simulations: int = 20000, seed: int = 42) -> dict:
    base = run_week_retrospective(2026, 6, simulations=simulations, seed=seed)
    if base.empty:
        raise RuntimeError("Frozen Week 6 retrospective returned no games")

    d = base.copy()
    d["v1_margin"] = pd.to_numeric(d["projected_margin"], errors="coerce")
    d["v11_margin"] = anchored_margin(d["v1_margin"], d["market_spread"])
    d["v1_pick_home"] = d["v1_margin"] > 0
    d["v11_pick_home"] = d["v11_margin"] > 0
    d["actual_home_win"] = pd.to_numeric(d["actual_margin"], errors="coerce") > 0
    d["v11_winner_correct"] = d["v11_pick_home"] == d["actual_home_win"]
    d["v11_margin_error"] = (d["v11_margin"] - d["actual_margin"]).abs()
    d["pick_changed"] = d["v11_pick_home"] != d["v1_pick_home"]
    d["market_home_margin"] = market_home_margin(d["market_spread"])
    d["v1_market_gap"] = (d["v1_margin"] - d["market_home_margin"]).abs()
    d["v11_market_gap"] = (d["v11_margin"] - d["market_home_margin"]).abs()

    v1_accuracy = float(d["winner_correct"].mean())
    v11_accuracy = float(d["v11_winner_correct"].mean())
    v1_mae = float(d["margin_error"].mean())
    v11_mae = float(d["v11_margin_error"].mean())

    changed = d[d["pick_changed"]].copy()
    cols = [
        "gameday", "away_team", "home_team", "market_spread", "market_home_margin",
        "v1_margin", "v11_margin", "actual_margin", "winner_correct",
        "v11_winner_correct", "v11_margin_error",
    ]
    return {
        "summary": {
            "games": int(len(d)),
            "model_weight": MODEL_WEIGHT,
            "market_weight": MARKET_WEIGHT,
            "v1_winner_accuracy": v1_accuracy,
            "v11_winner_accuracy": v11_accuracy,
            "winner_accuracy_delta": v11_accuracy - v1_accuracy,
            "v1_margin_mae": v1_mae,
            "v11_margin_mae": v11_mae,
            "margin_mae_delta": v11_mae - v1_mae,
            "picks_changed": int(d["pick_changed"].sum()),
            "changed_pick_accuracy": float(changed["v11_winner_correct"].mean()) if not changed.empty else np.nan,
            "mean_v1_market_gap": float(d["v1_market_gap"].mean()),
            "mean_v11_market_gap": float(d["v11_market_gap"].mean()),
            "status": "EXPERIMENTAL_NOT_PROMOTED",
        },
        "changed_picks": changed[cols].reset_index(drop=True),
        "results": d,
    }


if __name__ == "__main__":
    report = evaluate_challenger()
    print("ENGINE 1.1 CHALLENGER SUMMARY")
    print(report["summary"])
    print("\nCHANGED PICKS")
    if report["changed_picks"].empty:
        print("None")
    else:
        print(report["changed_picks"].to_string(index=False))
