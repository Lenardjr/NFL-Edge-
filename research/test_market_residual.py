import unittest
import pandas as pd
from research.market_residual import run

class MarketResidualTests(unittest.TestCase):
    def test_uses_prior_seasons_only(self):
        rows=[]
        for season in (2022,2023):
            for i in range(220):
                rows.append(dict(season=season,gameday=f"{season}-10-{i%25+1:02d}",
                    game_id=f"{season}-{i}",home_win=i%2,
                    model_home_prob=.3 if i%2 else .7,
                    market_home_prob=.45 if i%2 else .55))
        report,p=run(pd.DataFrame(rows))
        self.assertEqual(report["games"],220)
        self.assertEqual([r["season"] for r in report["seasons"]],[2023])
        self.assertEqual(len(p),220)

    def test_invalid_probability(self):
        d=pd.DataFrame([dict(season=2022,gameday="2022-01-01",home_win=1,model_home_prob=1.2,market_home_prob=.6)])
        with self.assertRaises(ValueError):run(d)

if __name__=="__main__":unittest.main()
