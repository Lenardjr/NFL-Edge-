"""Leakage-safe validation of NFL Edge probability/EV classifications.

Development evidence: 2023-2024.
Validation evidence: 2025.
2026 is deliberately never loaded by this module.

The goal is diagnostic: determine whether larger model-vs-market probability
edges correspond to stronger realized betting performance out of sample.
"""
from __future__ import annotations

import math
import pandas as pd

from edge_math import assess_two_way_side
from game_betting_lab import score_bets
from game_betting_multiseason import load_history

DEV_SEASONS = (2023, 2024)
VALIDATION_SEASON = 2025
ALLOWED_SEASONS = DEV_SEASONS + (VALIDATION_SEASON,)
BUCKET_ORDER = ["PASS", "LEAN", "EDGE", "STRONG_EDGE"]


def _side_rows(history: pd.DataFrame) -> pd.DataFrame:
    """Create one spread recommendation per game from the stronger model side."""
    rows = []
    required = {"home_cover_prob", "spread_line", "home_score", "away_score"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"history missing required columns: {sorted(missing)}")

    for _, g in history.iterrows():
        p_home = float(g.home_cover_prob)
        home = assess_two_way_side(p_home, -110, -110)
        away = assess_two_way_side(1.0 - p_home, -110, -110)
        side, assessment = ("HOME", home) if home.expected_value >= away.expected_value else ("AWAY", away)

        spread = float(g.spread_line)
        home_margin = float(g.home_score) - float(g.away_score)
        home_ats = home_margin + spread
        if home_ats == 0:
            result = "PUSH"
            units = 0.0
        else:
            won = home_ats > 0 if side == "HOME" else home_ats < 0
            result = "WIN" if won else "LOSS"
            units = 1.0 if won else -1.1

        rows.append({
            "season_eval": int(g.season_eval),
            "week_eval": int(g.week_eval),
            "gameday": g.gameday,
            "away_team": g.away_team,
            "home_team": g.home_team,
            "market": "SPREAD",
            "side": side,
            "model_probability": assessment.model_probability,
            "market_probability": assessment.market_probability,
            "edge": assessment.edge,
            "expected_value": assessment.expected_value,
            "classification": assessment.classification,
            "result": result,
            "units": units,
        })
    return pd.DataFrame(rows)


def _bucket_metrics(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket in BUCKET_ORDER:
        g = bets[bets.classification.eq(bucket)].copy()
        if g.empty:
            rows.append({"classification": bucket, "bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                         "win_rate": math.nan, "units": 0.0, "roi": math.nan,
                         "avg_edge": math.nan, "avg_ev": math.nan})
            continue
        wins = int((g.result == "WIN").sum())
        losses = int((g.result == "LOSS").sum())
        pushes = int((g.result == "PUSH").sum())
        decided = wins + losses
        risked = wins + losses * 1.1
        rows.append({
            "classification": bucket,
            "bets": int(len(g)),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": wins / decided if decided else math.nan,
            "units": float(g.units.sum()),
            "roi": float(g.units.sum()) / risked if risked else math.nan,
            "avg_edge": float(g.edge.mean()),
            "avg_ev": float(g.expected_value.mean()),
        })
    return pd.DataFrame(rows)


def evaluate_split(bets: pd.DataFrame, seasons: tuple[int, ...]) -> dict:
    sample = bets[bets.season_eval.isin(seasons)].copy()
    return {
        "seasons": list(seasons),
        "games": int(len(sample)),
        "buckets": _bucket_metrics(sample),
    }


def run_edge_bucket_validation(simulations: int = 5000) -> dict:
    # Explicitly request only 2023-2025. Never load 2026 here.
    history = load_history(seasons=ALLOWED_SEASONS, simulations=simulations)
    if not history.empty and history.season_eval.max() > VALIDATION_SEASON:
        raise RuntimeError("Protected 2026 holdout entered edge-bucket validation")
    bets = _side_rows(history)
    return {
        "method": "2023_2024_DEVELOPMENT__2025_VALIDATION__2026_UNTOUCHED",
        "development": evaluate_split(bets, DEV_SEASONS),
        "validation": evaluate_split(bets, (VALIDATION_SEASON,)),
        "all_bets": bets,
    }


if __name__ == "__main__":
    r = run_edge_bucket_validation()
    print("NFL EDGE PROBABILITY/EV BUCKET VALIDATION")
    print(r["method"])
    print("\n2023-2024 DEVELOPMENT")
    print(r["development"]["buckets"].to_string(index=False))
    print("\n2025 OUT-OF-SAMPLE VALIDATION")
    print(r["validation"]["buckets"].to_string(index=False))
    print("\n2026 FINAL HOLDOUT: UNTOUCHED")
