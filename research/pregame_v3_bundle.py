"""Assemble a pregame-safe V3 input bundle from a frozen schedule and team EPA.

EPA must be derived only from completed games dated strictly before the UTC
snapshot date. Incomplete or future game aggregates are rejected, not shifted.
This does not independently authenticate when source files were published.
"""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from research.pregame_schedule_snapshot import convert_schedule
from research.run_upcoming_v3 import run

def filter_pregame_epa(schedule:pd.DataFrame,team_epa:pd.DataFrame,*,asof_utc):
    cutoff=pd.Timestamp(asof_utc)
    if cutoff.tzinfo is None:raise ValueError("asof_utc must be timezone-aware")
    required={"game_id","team","offense_epa_per_play","offense_success_rate"}
    if required-set(team_epa):raise ValueError("Missing EPA columns")
    g=schedule.copy()
    if g.game_id.duplicated().any():raise ValueError("Duplicate schedule games")
    g["gameday"]=pd.to_datetime(g.gameday,errors="raise")
    done=g.home_score.notna() & g.away_score.notna()
    eligible=set(g.loc[done & (g.gameday.dt.date<cutoff.date()),"game_id"])
    t=team_epa.copy()
    if t.game_id.isna().any() or t.team.isna().any():raise ValueError("Missing EPA identity")
    if t.duplicated(["game_id","team"]).any():raise ValueError("Duplicate EPA team-game")
    if not set(t.game_id).issubset(eligible):
        raise ValueError("EPA contains uncompleted, same-day, or future games")
    # Each game needs exactly its two participating teams; partial EPA could
    # otherwise silently bias prior-game rolling metrics.
    teams={r.game_id:{str(r.home_team),str(r.away_team)} for _,r in g.iterrows()}
    for game_id,part in t.groupby("game_id"):
        if set(part.team.astype(str))!=teams[game_id]:
            raise ValueError("EPA missing team or includes nonparticipant")
    return t

def run_bundle(*,schedule_csv,epa_csv,output_dir,ledger=None,now_utc=None,minimum_train=300):
    now=now_utc or datetime.now(timezone.utc).isoformat()
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    games=convert_schedule(pd.read_csv(schedule_csv),captured_at_utc=now,asof_utc=now)
    epa=filter_pregame_epa(games,pd.read_csv(epa_csv),asof_utc=now)
    games_file=out/"frozen_games.csv";epa_file=out/"frozen_team_epa.csv"
    games.to_csv(games_file,index=False);epa.to_csv(epa_file,index=False)
    result=run(games_csv=games_file,epa_csv=epa_file,asof_utc=now,
               now_utc=now,output_dir=out,ledger=ledger,minimum_train=minimum_train)
    result["epa_team_games"]=len(epa)
    (out/"bundle_metadata.json").write_text(json.dumps({
        "captured_at_utc":now,"schedule_source":str(schedule_csv),
        "epa_source":str(epa_csv),"source_publication_verified":False,
        "notes":"Research snapshot; source publication and completion timestamps not independently authenticated"
    },indent=2))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--schedule-csv",required=True)
    p.add_argument("--epa-csv",required=True)
    p.add_argument("--output-dir",default="research_outputs")
    p.add_argument("--ledger",default=None)
    p.add_argument("--minimum-train",type=int,default=300)
    a=p.parse_args()
    print(json.dumps(run_bundle(schedule_csv=a.schedule_csv,epa_csv=a.epa_csv,
        output_dir=a.output_dir,ledger=a.ledger,minimum_train=a.minimum_train),indent=2))
if __name__=="__main__":main()
