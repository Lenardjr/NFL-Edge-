import math
import pytest

from edge_math import (
    american_to_decimal,
    assess_two_way_side,
    expected_value,
    implied_probability,
    remove_vig_two_way,
)


def test_positive_american_odds():
    assert american_to_decimal(110) == pytest.approx(2.10)
    assert implied_probability(110) == pytest.approx(100 / 210)


def test_negative_american_odds():
    assert american_to_decimal(-150) == pytest.approx(1.6666666667)
    assert implied_probability(-150) == pytest.approx(0.60)


def test_ev_example_from_design():
    assert expected_value(0.60, 110) == pytest.approx(0.26)


def test_remove_vig_sums_to_one():
    a, b = remove_vig_two_way(-110, -110)
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(0.5)
    assert a + b == pytest.approx(1.0)


def test_assessment_exposes_market_edge_ev_and_kelly():
    result = assess_two_way_side(0.60, 110, -130)
    assert 0 <= result.market_probability <= 1
    assert result.edge == pytest.approx(result.model_probability - result.market_probability)
    assert result.expected_value == pytest.approx(0.26)
    assert result.full_kelly >= result.quarter_kelly >= 0
    assert result.classification in {"PASS", "LEAN", "EDGE", "STRONG_EDGE"}


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):
        expected_value(1.01, -110)
