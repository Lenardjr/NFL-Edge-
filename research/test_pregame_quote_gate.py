import unittest
import pandas as pd
from research.pregame_quote_gate import validate_quotes,evaluate_paper_candidates
class PregameQuoteGateTests(unittest.TestCase):
    def quote(self):
        return pd.DataFrame([{"game_id":"g1","side":"home","american_odds":150,"sportsbook":"testbook",
          "source":"archive","source_url":"https://example.org/quote","captured_at":"2025-10-01T18:00:00Z",
          "kickoff_at":"2025-10-01T20:00:00Z","verification":"independently_verified_pregame"}])
    def test_declared_verified_quote(self):
        _,s=validate_quotes(self.quote());self.assertEqual(s["eligible"],1)
    def test_unverified_rejected(self):
        q=self.quote();q.loc[0,"verification"]="unreviewed"
        _,s=validate_quotes(q);self.assertEqual(s["eligible"],0)
    def test_postkickoff_rejected(self):
        q=self.quote();q.loc[0,"captured_at"]="2025-10-01T21:00:00Z"
        _,s=validate_quotes(q);self.assertEqual(s["eligible"],0)
    def test_invalid_odds_rejected(self):
        q=self.quote();q.loc[0,"american_odds"]=50
        _,s=validate_quotes(q);self.assertEqual(s["eligible"],0)
    def test_prediction_after_quote_rejected(self):
        p=pd.DataFrame([{"game_id":"g1","home_probability":.5,"prediction_at":"2025-10-01T19:00:00Z"}])
        with self.assertRaisesRegex(ValueError,"not frozen"):evaluate_paper_candidates(p,self.quote())
    def test_expected_roi_without_outcomes(self):
        p=pd.DataFrame([{"game_id":"g1","home_probability":.5,"prediction_at":"2025-10-01T17:00:00Z"}])
        result,s=evaluate_paper_candidates(p,self.quote())
        self.assertAlmostEqual(result.expected_roi.iloc[0],.25)
        self.assertEqual(s["paired_candidates"],1)
    def test_duplicate_quote_rejected(self):
        q=pd.concat([self.quote(),self.quote()],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Duplicate"):validate_quotes(q)
