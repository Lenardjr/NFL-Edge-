"""Inspect actual nflverse injury CSV releases without using them as pregame features.

The files are retrospectively updated season releases. Report dates and schema
do not prove any individual record was available before a given kickoff.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

BASE="https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
def inspect(season:int,frame:pd.DataFrame):
    if frame.empty:raise ValueError("Empty injury release")
    cols=list(frame.columns)
    if len(cols)!=len(set(cols)):raise ValueError("Duplicate columns")
    date_cols=[c for c in cols if any(k in c.lower() for k in ("date","timestamp","updated","reported"))]
    week_cols=[c for c in cols if "week" in c.lower()]
    team_cols=[c for c in cols if "team" in c.lower()]
    return {"season":season,"rows":len(frame),"columns":cols,"date_columns":date_cols,
            "week_columns":week_cols,"team_columns":team_cols,
            "missing_fraction":{c:round(float(frame[c].isna().mean()),4) for c in date_cols+week_cols+team_cols},
            "verified_pregame_records":0,
            "eligible_for_model":False,
            "warning":"Season-release rows are retrospective; date/week fields do not authenticate an archived pregame snapshot."}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seasons",nargs="+",type=int,default=[2024,2025])
    p.add_argument("--output",default="research_outputs/nfl_injury_release_schema_audit.json")
    a=p.parse_args()
    if any(y<2009 or y>2026 for y in a.seasons) or len(a.seasons)!=len(set(a.seasons)):
        raise ValueError("Invalid or duplicate seasons")
    reports=[]
    for y in a.seasons:
        frame=pd.read_csv(BASE.format(season=y),low_memory=False)
        reports.append(inspect(y,frame))
    result={"releases_inspected":len(reports),"seasons":reports,"verified_pregame_records":0,
            "model_eligible":False,
            "next_step":"Review actual columns and source provenance before constructing pregame features"}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
