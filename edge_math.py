"""Core market probability and expected-value math for NFL Edge.

This module is intentionally sport-agnostic so the same pricing layer can later
support NBA, MLB, and NHL models without changing the underlying mathematics.
"""
from __future__ import annotations

from dataclasses import dataclass


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def implied_probability(odds: float) -> float:
    """Return the sportsbook's raw implied probability, including vig."""
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def remove_vig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Normalize a two-way market into fair/no-vig probabilities."""
    pa = implied_probability(odds_a)
    pb = implied_probability(odds_b)
    total = pa + pb
    if total <= 0:
        raise ValueError("Invalid two-way market")
    return pa / total, pb / total


def expected_value(model_probability: float, american_odds: float) -> float:
    """Expected profit per $1 risked.

    Example: 0.60 probability at +110 => +0.26 EV per $1 risked.
    """
    _validate_probability(model_probability)
    decimal = american_to_decimal(american_odds)
    net_win = decimal - 1.0
    return model_probability * net_win - (1.0 - model_probability)


def probability_edge(model_probability: float, market_probability: float) -> float:
    """Model probability minus fair market probability."""
    _validate_probability(model_probability)
    _validate_probability(market_probability)
    return model_probability - market_probability


def full_kelly_fraction(model_probability: float, american_odds: float) -> float:
    """Full Kelly bankroll fraction, floored at zero for non-positive edges."""
    _validate_probability(model_probability)
    b = american_to_decimal(american_odds) - 1.0
    q = 1.0 - model_probability
    return max(0.0, (b * model_probability - q) / b)


def fractional_kelly(model_probability: float, american_odds: float, fraction: float = 0.25) -> float:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Kelly fraction must be between 0 and 1")
    return full_kelly_fraction(model_probability, american_odds) * fraction


def classify_edge(edge: float, ev: float) -> str:
    """Conservative v1 display classification; thresholds can later be calibrated."""
    if edge < 0.02 or ev <= 0:
        return "PASS"
    if edge < 0.04:
        return "LEAN"
    if edge < 0.07:
        return "EDGE"
    return "STRONG_EDGE"


@dataclass(frozen=True)
class EdgeAssessment:
    model_probability: float
    market_probability: float
    edge: float
    expected_value: float
    full_kelly: float
    quarter_kelly: float
    classification: str


def assess_two_way_side(
    model_probability: float,
    selected_odds: float,
    opposite_odds: float,
) -> EdgeAssessment:
    """Evaluate one side of a two-way market against its no-vig market probability."""
    fair_selected, _ = remove_vig_two_way(selected_odds, opposite_odds)
    edge = probability_edge(model_probability, fair_selected)
    ev = expected_value(model_probability, selected_odds)
    return EdgeAssessment(
        model_probability=model_probability,
        market_probability=fair_selected,
        edge=edge,
        expected_value=ev,
        full_kelly=full_kelly_fraction(model_probability, selected_odds),
        quarter_kelly=fractional_kelly(model_probability, selected_odds, 0.25),
        classification=classify_edge(edge, ev),
    )


def _validate_probability(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError("Probability must be between 0 and 1")
