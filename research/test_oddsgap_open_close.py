import unittest
import pandas as pd
from research.oddsgap_open_close import normalize
class OddsGapOpenCloseTests(unittest.TestCase):
    def row(self):
        return {"sport":"americanfootball_nfl","away":"New York Jets","home":"Chicago Bears",
         "commence_time":"2026-10-04T17:00:00Z","market":"ml","side":"home",
         "book":"betmgm","game_id":"nfl-test","line":None,
         "earliest_ts":"2026-09-28T00:00:27Z","earliest_american_odds":-160,
         "close_ts":"2026-10-04T16:00:41Z","close_american_odds":-190}
    def test_two_quotes(self):
        d,s=normalize(pd.DataFrame([self.row()]))
        self.assertEqual(s["valid_pregame_quotes"],2)
        self.assertEqual(set(d.quote_phase),{"earliest","close"})
        self.assertFalse(d.model_eligible.any())
    def test_rejects_after_kickoff(self):
        r=self.row();r["close_ts"]="2026-10-04T18:00:00Z"
        d,s=normalize(pd.DataFrame([r]));self.assertEqual(len(d),1)
    def test_rejects_invalid_price(self):
        r=self.row();r["earliest_american_odds"]=0
        d,s=normalize(pd.DataFrame([r]));self.assertEqual(len(d),1)
    def test_market_filter(self):
        r=self.row();r["market"]="player_prop"
        d,s=normalize(pd.DataFrame([r]));self.assertEqual(len(d),0)
    def test_required_fields(self):
        with self.assertRaisesRegex(ValueError,"Missing columns"):normalize(pd.DataFrame([{"sport":"americanfootball_nfl"}]))
