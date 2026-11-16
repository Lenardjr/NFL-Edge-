"""Research-only market arithmetic and paper-trade validation.

No live odds retrieval or wager placement. Inputs must be timestamped, sourced,
and frozen before game start. All probabilities are decimal fractions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math


def american_to_decimal(odds: int) -> float:
    if isinstance(odds, bool) or not isinstance(odds, int) or odds == 0 or abs(odds) < 100:
        raise ValueError("American odds must be integers <= -100 or >= +100")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def implied_probability(odds: int) -> float:
    return 1.0 / american_to_decimal(odds)


def no_vig_two_way(odds_a: int, odds_b: int) -> tuple[float, float]:
    """Proportional no-vig estimate for an exhaustive two-outcome market."""
    a, b = implied_probability(odds_a), implied_probability(odds_b)
    return a / (a + b), b / (a + b)


def expected_roi(probability: float, american_odds: int) -> float:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    return probability * american_to_decimal(american_odds) - 1.0


def brier_score(probabilities, outcomes) -> float:
    p, y = list(probabilities), list(outcomes)
    if not p or len(p) != len(y) or any(not 0 <= v <= 1 for v in p) or any(v not in (0, 1) for v in y):
        raise ValueError("aligned binary outcomes and probabilities required")
    return sum((a - b) ** 2 for a, b in zip(p, y)) / len(p)


def log_loss(probabilities, outcomes, eps: float = 1e-12) -> float:
    p, y = list(probabilities), list(outcomes)
    if not p or len(p) != len(y) or any(not 0 <= v <= 1 for v in p) or any(v not in (0, 1) for v in y):
        raise ValueError("aligned binary outcomes and probabilities required")
    return -sum(b * math.log(max(eps, min(1-eps, a))) + (1-b)*math.log(max(eps, min(1-eps, 1-a))) for a,b in zip(p,y)) / len(p)


def validate_snapshot(*, captured_at: str, kickoff_at: str, sportsbook: str, source: str) -> None:
    """Require explicit UTC offsets; quote must precede scheduled event."""
    if not sportsbook.strip() or not source.strip():
        raise ValueError("book and price source required")
    def parse(s):
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is None or d.utcoffset() is None:
            raise ValueError("timestamps must include timezone")
        return d.astimezone(timezone.utc)
    if parse(captured_at) >= parse(kickoff_at):
        raise ValueError("odds snapshot must precede kickoff")


def settle_moneyline(*, stake: float, american_odds: int, result: str) -> float:
    """Return net profit in stake units/currency; result: win, loss, void."""
    if not math.isfinite(stake) or stake <= 0:
        raise ValueError("stake must be positive")
    decimal = american_to_decimal(american_odds)
    if result == "win":
        return stake * (decimal - 1)
    if result == "loss":
        return -stake
    if result == "void":
        return 0.0
    raise ValueError("result must be win/loss/void")
