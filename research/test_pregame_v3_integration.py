"""Positive end-to-end test: a frozen pregame bundle produces a real probability.

All matchups, scores and EPA here are synthetic; no real-world performance
claim is implied. The test catches integration bugs missed by rejection tests.
"""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from research.pregame_v3_bundle import run_bundle

class PregameBundleIntegrationTests(unittest.TestCase):
    def test_synthetic_full_pipeline(self):
        teams=["CHI","NYJ","GB","DET"]
        rows=[];epa=[]
        start=pd.Timestamp("2025-01-01")
        for day in range(165):
            date=(start+pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            for k,(home,away) in enumerate((("CHI","NYJ"),("GB","DET")) if day%2==0
                                             else (("NYJ","GB"),("DET","CHI"))):
                gid=f"2025_{day:03d}_{k}_{away}_{home}"
                future=day==164
                hs=None if future else 17+(day+k*3)%21
                aws=None if future else 14+(day*3+k)%20
                rows.append(dict(game_id=gid,season=2025,week=day//7+1,
                    gameday=date,gametime="13:00",home_team=home,away_team=away,
                    home_score=hs,away_score=aws))
                if not future:
                    for team in (home,away):
                        epa.append(dict(game_id=gid,team=team,
                            offense_epa_per_play=((day+k)%9-4)/30,
                            offense_success_rate=.4+((day+k)%5)/40))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);games=p/"games.csv";team_epa=p/"epa.csv"
            pd.DataFrame(rows).to_csv(games,index=False)
            pd.DataFrame(epa).to_csv(team_epa,index=False)
            now="2025-06-13T12:00:00Z" # June 13 = day 163; day 164 June 14
            # Use snapshot on day 164 before 13:00 ET kickoff; same-day scores
            # suppressed and future kickoff remains in the future.
            now="2025-06-14T12:00:00Z"
            result=run_bundle(schedule_csv=games,epa_csv=team_epa,
                output_dir=p/"out",now_utc=now,minimum_train=300)
            self.assertEqual(result["predictions"],2)
            predicted=pd.read_csv(p/"out"/"nfl_upcoming_v3_research_predictions.csv")
            self.assertEqual(len(predicted),2)
            self.assertTrue(predicted.home_win_probability.between(0,1).all())
            self.assertEqual(result["logged"],0)
