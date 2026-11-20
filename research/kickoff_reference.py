"""Research-only kickoff reference built from nflverse schedule game IDs.

Schedule kickoff times are historical metadata, NOT independently archived
pregame timestamps. Do not use these rows to verify quote executability.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from research.run_nfl_v3 import SCHEDULE_URL

ET=ZoneInfo("America/New_York")
def build_kickoff_reference(predictions:pd.DataFrame,schedule:pd.DataFrame):
    if "game_id" not in predictions or predictions.game_id.isna().any() or predictions.game_id.duplicated().any():
        raise ValueError("Invalid prediction game IDs")
    fields=["game_id","gameday","gametime","home_team","away_team","season","week"]
    if set(fields)-set(schedule):raise ValueError("Missing schedule fields")
    s=schedule[fields].copy()
    if s.game_id.isna().any() or s.game_id.duplicated().any():raise ValueError("Invalid schedule IDs")
    p=predictions[["game_id"]].merge(s,on="game_id",how="left",validate="one_to_one",indicator=True)
    if (p._merge!="both").any():raise ValueError("Unmatched prediction games")
    def convert(r):
        day=str(r.gameday)
        time=str(r.gametime)
        try:
            d=datetime.strptime(day+" "+time,"%Y-%m-%d %H:%M").replace(tzinfo=ET)
            # Check wall time round trip to catch invalid daylight-transition times.
            from datetime import timezone
            u=d.astimezone(timezone.utc)
            if u.astimezone(ET).replace(tzinfo=None)!=d.replace(tzinfo=None):
                raise ValueError("Nonexistent Eastern wall time")
            return u.isoformat()
        except (ValueError,TypeError) as e:
            raise ValueError("Invalid game kickoff for "+str(r.game_id)+": "+str(e)) from e
    if p[["home_team","away_team","season","week"]].isna().any().any():
        raise ValueError("Missing matchup metadata")
    p["kickoff_utc"]=p.apply(convert,axis=1)
    p["kickoff_source"]="nflverse historical schedule; not authenticated pregame snapshot"
    p["verified_pregame"]=False
    p=p.drop(columns="_merge")
    return p,{"prediction_games":len(predictions),"matched_kickoffs":len(p),
              "verified_pregame_kickoffs":0,
              "caveat":"Historical schedule timestamps are reference metadata, not archived pregame evidence; kickoff revisions possible."}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="research_outputs/nfl_v3_predictions.csv")
    ap.add_argument("--schedule",default=SCHEDULE_URL)
    ap.add_argument("--output-dir",default="research_outputs")
    a=ap.parse_args()
    result,summary=build_kickoff_reference(pd.read_csv(a.predictions),pd.read_csv(a.schedule))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"nfl_kickoff_reference.csv",index=False)
    (out/"nfl_kickoff_reference_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
