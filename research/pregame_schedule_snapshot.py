"""Prepare a conservative NFLverse schedule snapshot for pregame research.

NFLverse schedule 'gametime' is interpreted as America/New_York wall time,
including DST; rows with missing or invalid time are rejected, never guessed.
Snapshot timestamp is the actual capture time, not the requested historical
as-of date. No claim of independent source publication verification.
"""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

EASTERN=ZoneInfo("America/New_York")
def convert_schedule(raw:pd.DataFrame,*,captured_at_utc,asof_utc):
    captured=pd.Timestamp(captured_at_utc)
    asof=pd.Timestamp(asof_utc)
    if captured.tzinfo is None or asof.tzinfo is None:raise ValueError("UTC-aware timestamps required")
    captured=captured.tz_convert("UTC");asof=asof.tz_convert("UTC")
    if asof>captured:raise ValueError("As-of cannot be later than capture")
    required={"game_id","season","week","gameday","gametime","home_team","away_team","home_score","away_score"}
    if required-set(raw):raise ValueError("Missing schedule columns: "+str(sorted(required-set(raw))))
    g=raw.copy()
    if g.game_id.isna().any() or g.game_id.duplicated().any():raise ValueError("Duplicate or missing game IDs")
    def kickoff(r):
        if pd.isna(r.gametime) or not str(r.gametime).strip():raise ValueError("Missing gametime")
        date=pd.Timestamp(r.gameday).strftime("%Y-%m-%d")
        stamp=pd.Timestamp(date+" "+str(r.gametime)).tz_localize(EASTERN,ambiguous="raise",nonexistent="raise")
        return stamp.tz_convert("UTC").isoformat()
    g["kickoff_utc"]=g.apply(kickoff,axis=1)
    g["home_score"]=pd.to_numeric(g.home_score,errors="coerce")
    g["away_score"]=pd.to_numeric(g.away_score,errors="coerce")
    if (g.home_score.notna()!=g.away_score.notna()).any():
        raise ValueError("Incomplete score pair")
    # A snapshot captured now cannot reconstruct the data as it appeared in
    # the past. Explicitly reject backdating to another UTC calendar date.
    if asof.date()!=captured.date():raise ValueError("Cannot backdate a current schedule snapshot")
    # Scores from same-day games are conservatively removed, because we have
    # no independently verified final-whistle timestamp.
    gamedays=pd.to_datetime(g.gameday,errors="raise").dt.date
    g.loc[gamedays>=asof.date(),["home_score","away_score"]]=float("nan")
    return g
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--schedule-csv",required=True)
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    now=datetime.now(timezone.utc).isoformat()
    raw=pd.read_csv(a.schedule_csv)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    g=convert_schedule(raw,captured_at_utc=now,asof_utc=now)
    g.to_csv(out/"nfl_pregame_schedule_snapshot.csv",index=False)
    (out/"nfl_pregame_schedule_snapshot_metadata.json").write_text(json.dumps({
      "captured_at_utc":now,"asof_utc":now,"source_file":str(a.schedule_csv),
      "rows":len(g),"source_publication_verified":False,
      "notes":"NFLverse gametime interpreted as US Eastern local; same-day scores suppressed; historical backdating forbidden."},indent=2))
    print(f"Prepared {len(g)} schedule rows")
if __name__=="__main__":main()
