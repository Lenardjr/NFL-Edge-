"""Prospective holdout evaluator for frozen NFL Edge Game Engine candidates.

This module scores already-frozen predictions only. It is intentionally separate
from model fitting/tuning so future games can act as genuine holdout evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE_ID = "NFL_EDGE_ENGINE_1_1_B_CORRECTED_2026W6"
MODEL_WEIGHT = 0.70
MARKET_WEIGHT = 0.30


def market_home_margin(spread):
    """nflverse convention: positive = home favored, negative = away favored."""
    return pd.to_numeric(spread, errors="coerce")


def challenger_margin(engine_1_0_margin, market_spread):
    model = pd.to_numeric(engine_1_0_margin, errors="coerce")
    market = market_home_margin(market_spread)
    return np.where(pd.notna(market), MODEL_WEIGHT * model + MARKET_WEIGHT * market, model)


def score_frozen_predictions(predictions: pd.DataFrame, results: pd.DataFrame) -> dict:
    """Score frozen pregame rows against later final results.

    Required prediction columns:
      game_id, engine_1_0_margin, market_spread
    Required result columns:
      game_id, actual_home_score, actual_away_score
    """
    required_predictions = {"game_id", "engine_1_0_margin", "market_spread"}
    required_results = {"game_id", "actual_home_score", "actual_away_score"}
    missing_p = required_predictions - set(predictions.columns)
    missing_r = required_results - set(results.columns)
    if missing_p or missing_r:
        raise ValueError(f"Missing columns predictions={sorted(missing_p)} results={sorted(missing_r)}")

    d = predictions.merge(results, on="game_id", how="inner", validate="one_to_one").copy()
    if d.empty:
        raise ValueError("No matched frozen predictions/results")

    d["actual_margin"] = pd.to_numeric(d["actual_home_score"], errors="coerce") - pd.to_numeric(d["actual_away_score"], errors="coerce")
    d["engine_1_0_margin"] = pd.to_numeric(d["engine_1_0_margin"], errors="coerce")
    d["engine_1_1_b_margin"] = challenger_margin(d["engine_1_0_margin"], d["market_spread"])
    d["engine_1_0_error"] = (d["engine_1_0_margin"] - d["actual_margin"]).abs()
    d["engine_1_1_b_error"] = (d["engine_1_1_b_margin"] - d["actual_margin"]).abs()
    d["engine_1_0_home_pick"] = d["engine_1_0_margin"] > 0
    d["engine_1_1_b_home_pick"] = d["engine_1_0_home_pick"]
    d["actual_home_win"] = d["actual_margin"] > 0

    summary = {
        "candidate_id": CANDIDATE_ID,
        "games": int(len(d)),
        "engine_1_0_margin_mae": float(d["engine_1_0_error"].mean()),
        "engine_1_1_b_margin_mae": float(d["engine_1_1_b_error"].mean()),
        "margin_mae_delta": float(d["engine_1_1_b_error"].mean() - d["engine_1_0_error"].mean()),
        "engine_1_0_winner_accuracy": float((d["engine_1_0_home_pick"] == d["actual_home_win"]).mean()),
        "engine_1_1_b_winner_accuracy": float((d["engine_1_1_b_home_pick"] == d["actual_home_win"]).mean()),
        "winner_definition_matches": bool((d["engine_1_0_home_pick"] == d["engine_1_1_b_home_pick"]).all()),
    }
    return {"summary": summary, "results": d}


def load_json_rows(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    rows = payload if isinstance(payload, list) else payload.get("games", [])
    return pd.DataFrame(rows)
