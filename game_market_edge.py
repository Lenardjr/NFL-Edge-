"""Translate NFL Edge game-model probabilities into priced betting decisions.

This module is deliberately separate from game_engine.py so the validated model
remains unchanged. It consumes model output and sportsbook prices, then delegates
all pricing math to edge_math.
"""
from __future__ import annotations

from dataclasses import asdict

from edge_math import assess_two_way_side


def _opposite_probability(p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError("Probability must be between 0 and 1")
    return 1.0 - p


def assess_moneyline(prediction: dict, home_odds: float, away_odds: float) -> dict:
    """Price both moneyline sides from a game_engine prediction."""
    home_p = float(prediction["home_win_prob"])
    return {
        "home": asdict(assess_two_way_side(home_p, home_odds, away_odds)),
        "away": asdict(assess_two_way_side(_opposite_probability(home_p), away_odds, home_odds)),
    }


def assess_spread(prediction: dict, home_odds: float = -110, away_odds: float = -110) -> dict:
    """Price both spread sides. prediction must include home_cover_prob."""
    home_p = float(prediction["home_cover_prob"])
    return {
        "home": asdict(assess_two_way_side(home_p, home_odds, away_odds)),
        "away": asdict(assess_two_way_side(_opposite_probability(home_p), away_odds, home_odds)),
    }


def assess_total(prediction: dict, over_odds: float = -110, under_odds: float = -110) -> dict:
    """Price over and under. prediction must include over_prob."""
    over_p = float(prediction["over_prob"])
    return {
        "over": asdict(assess_two_way_side(over_p, over_odds, under_odds)),
        "under": asdict(assess_two_way_side(_opposite_probability(over_p), under_odds, over_odds)),
    }


def best_available_edge(markets: dict) -> dict | None:
    """Return the highest-EV non-PASS side across supplied market assessments."""
    candidates = []
    for market_name, sides in markets.items():
        if sides is None:
            continue
        for side_name, assessment in sides.items():
            if assessment["classification"] != "PASS":
                candidates.append({"market": market_name, "side": side_name, **assessment})
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x["expected_value"], x["edge"]))


def assess_game_markets(
    prediction: dict,
    *,
    home_moneyline: float | None = None,
    away_moneyline: float | None = None,
    home_spread_odds: float = -110,
    away_spread_odds: float = -110,
    over_odds: float = -110,
    under_odds: float = -110,
) -> dict:
    """Create a complete priced snapshot for all markets available in prediction."""
    markets = {}
    if home_moneyline is not None and away_moneyline is not None:
        markets["moneyline"] = assess_moneyline(prediction, home_moneyline, away_moneyline)
    if "home_cover_prob" in prediction:
        markets["spread"] = assess_spread(prediction, home_spread_odds, away_spread_odds)
    if "over_prob" in prediction:
        markets["total"] = assess_total(prediction, over_odds, under_odds)
    return {"markets": markets, "best_edge": best_available_edge(markets)}
