import tempfile,unittest
from pathlib import Path
import pandas as pd
from research.run_upcoming_v3 import run
class UpcomingV3RunnerTests(unittest.TestCase):
    def test_reject_untrained_snapshot_without_logging(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);g=p/"games.csv";e=p/"epa.csv";ledger=p/"ledger.csv"
            pd.DataFrame([dict(game_id="2026_11_CHI_NYJ",season=2026,week=11,
                gameday="2026-11-22",gametime="13:00",home_team="CHI",
                away_team="NYJ",home_score=None,away_score=None,
                kickoff_utc="2026-11-22T18:00:00Z")]).to_csv(g,index=False)
            pd.DataFrame(columns=["game_id","team","offense_epa_per_play","offense_success_rate"]).to_csv(e,index=False)
            with self.assertRaisesRegex(ValueError,"Insufficient"):
                run(games_csv=g,epa_csv=e,asof_utc="2026-11-19T12:00:00Z",
                    now_utc="2026-11-19T13:00:00Z",output_dir=p/"out",ledger=ledger)
            self.assertFalse(ledger.exists())
