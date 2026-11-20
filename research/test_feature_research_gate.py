import unittest
import pandas as pd
from research.feature_research_gate import eligible_feature_snapshot,promotion_decision
class FeatureResearchGateTests(unittest.TestCase):
    def setUp(self):
        self.f=pd.DataFrame([dict(game_id="g1",feature_name="wind_gust_mph",
            feature_value=15,category="weather",published_at_utc="2026-11-19T12:00:00Z",
            source_id="forecast_1")])
    def test_accept_prior_published(self):
        self.assertEqual(len(eligible_feature_snapshot(self.f,asof_utc="2026-11-19T13:00:00Z")),1)
    def test_reject_future_published(self):
        with self.assertRaisesRegex(ValueError,"after prediction"):
            eligible_feature_snapshot(self.f,asof_utc="2026-11-19T11:00:00Z")
    def test_reject_market_by_default(self):
        self.f.loc[0,"category"]="market"
        with self.assertRaisesRegex(ValueError,"opt-in"):
            eligible_feature_snapshot(self.f,asof_utc="2026-11-19T13:00:00Z")
    def test_reject_duplicate(self):
        self.f=pd.concat([self.f,self.f],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            eligible_feature_snapshot(self.f,asof_utc="2026-11-19T13:00:00Z")
    def test_reject_unverified_promotion(self):
        result=promotion_decision(baseline_brier=.22,candidate_brier=.20,
           market_brier=.21,paired_games=1000,ci_upper_delta=-.001,source_verified=False)
        self.assertFalse(result["promote"])
    def test_reject_market_underperformance(self):
        result=promotion_decision(baseline_brier=.22,candidate_brier=.215,
           market_brier=.21,paired_games=1000,ci_upper_delta=-.001,source_verified=True)
        self.assertFalse(result["promote"])
