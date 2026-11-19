import unittest
from research.historical_odds_import import parse_snapshot
class HistoricalOddsImportTests(unittest.TestCase):
    def payload(self):
        return {"timestamp":"2025-01-01T19:00:00Z","data":[{"id":"event1",
          "home_team":"Denver Broncos","away_team":"Kansas City Chiefs",
          "commence_time":"2025-01-05T21:25:00Z",
          "bookmakers":[{"key":"book1","last_update":"2025-01-01T18:59:00Z",
          "markets":[{"key":"h2h","outcomes":[{"name":"Denver Broncos","price":-110},
          {"name":"Kansas City Chiefs","price":100}]}]}]}]}
    def test_extracts_two_sides(self):
        d=parse_snapshot(self.payload(),"2025-01-01T19:01:00Z")
        self.assertEqual(len(d),2)
        self.assertFalse(d.model_eligible.any())
    def test_no_future_snapshot(self):
        with self.assertRaisesRegex(ValueError,"after requested"):
            parse_snapshot(self.payload(),"2025-01-01T18:00:00Z")
    def test_requires_data(self):
        with self.assertRaises(ValueError):parse_snapshot({},"2025-01-01T19:01:00Z")
    def test_missing_team_quote_excluded(self):
        p=self.payload();p["data"][0]["bookmakers"][0]["markets"][0]["outcomes"].pop()
        self.assertEqual(len(parse_snapshot(p,"2025-01-01T19:01:00Z")),0)
    def test_invalid_price_rejected(self):
        p=self.payload();p["data"][0]["bookmakers"][0]["markets"][0]["outcomes"][0]["price"]=0
        with self.assertRaisesRegex(ValueError,"Invalid American odds"):
            parse_snapshot(p,"2025-01-01T19:01:00Z")
