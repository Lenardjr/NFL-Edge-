import unittest
import pandas as pd
from research.paper_ledger import validate_predictions,settle

def snapshot():
    return pd.DataFrame([dict(game_id="g1",kickoff_utc="2026-11-22T18:00:00Z",
      captured_at_utc="2026-11-22T17:00:00Z",odds_captured_at_utc="2026-11-22T16:55:00Z",
      home_team="A",away_team="B",model_version="v2",model_home_prob=.7,
      home_moneyline=-110,away_moneyline=100,sportsbook="sample")])

class PaperLedgerTests(unittest.TestCase):
    def test_valid_and_settled(self):
        d=snapshot()
        self.assertEqual(len(validate_predictions(d)),1)
        picks,report=settle(d,pd.DataFrame([{"game_id":"g1","home_win":1}]))
        self.assertEqual(picks.iloc[0].selection,"home")
        self.assertEqual(report["completed_bets"],1)
        self.assertGreater(report["total_profit_units"],0)
    def test_after_kickoff_rejected(self):
        d=snapshot();d.loc[0,"captured_at_utc"]="2026-11-22T19:00:00Z"
        with self.assertRaises(ValueError):validate_predictions(d)
    def test_future_odds_rejected(self):
        d=snapshot();d.loc[0,"odds_captured_at_utc"]="2026-11-22T17:01:00Z"
        with self.assertRaises(ValueError):validate_predictions(d)
    def test_duplicate_snapshot_rejected(self):
        d=snapshot();d=pd.concat([d,d],ignore_index=True)
        with self.assertRaises(ValueError):validate_predictions(d)
if __name__=="__main__":unittest.main()
