"""Deterministic arithmetic and integrity tests; run: python -m unittest discover -s research -p 'test_*.py'."""
import unittest
from research.market_validation import (
    american_to_decimal, implied_probability, no_vig_two_way,
    expected_roi, brier_score, log_loss, validate_snapshot, settle_moneyline
)


class MarketValidationTests(unittest.TestCase):
    def test_american_odds(self):
        self.assertAlmostEqual(american_to_decimal(120), 2.2)
        self.assertAlmostEqual(american_to_decimal(-110), 1.90909090909)
        self.assertAlmostEqual(implied_probability(120), 1/2.2)
        for invalid in (0, 99, -99):
            with self.assertRaises(ValueError):
                american_to_decimal(invalid)

    def test_no_vig(self):
        a,b = no_vig_two_way(-110,-110)
        self.assertAlmostEqual(a, 0.5)
        self.assertAlmostEqual(a+b, 1)

    def test_ev(self):
        self.assertAlmostEqual(expected_roi(.52,120), .144)
        self.assertAlmostEqual(expected_roi(.5,100), 0)

    def test_scores(self):
        self.assertAlmostEqual(brier_score([.8,.2],[1,0]), .04)
        self.assertGreater(log_loss([.5,.5],[1,0]), 0)

    def test_timestamp_guard(self):
        args=dict(captured_at="2026-11-16T18:00:00-05:00", kickoff_at="2026-11-16T20:00:00-05:00", sportsbook="Example Book", source="archived quote")
        validate_snapshot(**args)
        with self.assertRaises(ValueError):
            validate_snapshot(**{**args, "captured_at":"2026-11-16T21:00:00-05:00"})
        with self.assertRaises(ValueError):
            validate_snapshot(**{**args, "captured_at":"2026-11-16T18:00:00"})

    def test_settlement(self):
        self.assertAlmostEqual(settle_moneyline(stake=100, american_odds=120, result="win"),120)
        self.assertAlmostEqual(settle_moneyline(stake=100, american_odds=-110, result="loss"),-100)
        self.assertAlmostEqual(settle_moneyline(stake=100, american_odds=120, result="void"),0)


if __name__ == "__main__":
    unittest.main()
