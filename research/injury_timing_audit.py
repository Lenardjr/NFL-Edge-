"""Join retrospective injury records to historical games and audit timestamp usability.

No injury record is declared independently verified or eligible for modeling.
"""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from research.injury_release_schema_audit import BASE

def parse_aware(value):
    if pd.isna(value):return None
    try:
        stamp=pd.Timestamp(str(value).strip())
        if pd.isna(stamp) or stamp.tzinfo is None:return None
        return stamp.tz_convert("UTC")
    except (ValueError,TypeError,OverflowError):return None

def audit(kickoffs:pd.DataFrame,injuries:pd.DataFrame):
    required_k={"game_id","season","week","home_team","away_team","kickoff_utc"}
    required_i={"season","week","team","date_modified"}
    if required_k-set(kickoffs):raise ValueError("Missing kickoff columns")
    if required_i-set(injuries):raise ValueError("Missing injury columns")
    if kickoffs.game_id.isna().any() or kickoffs.game_id.duplicated().any():
        raise ValueError("Invalid kickoff game IDs")
    if injuries.empty:raise ValueError("Empty injury data")
    if injuries[["season","week","team"]].isna().any().any():
        raise ValueError("Missing injury join identifiers")
    k=kickoffs.copy()
    k["kickoff_parsed"]=k.kickoff_utc.map(parse_aware)
    if k.kickoff_parsed.isna().any():raise ValueError("Invalid timezone-aware kickoff")
    sides=pd.concat([
        k[["game_id","season","week","home_team","kickoff_utc","kickoff_parsed"]].rename(columns={"home_team":"team"}).assign(side="home"),
        k[["game_id","season","week","away_team","kickoff_utc","kickoff_parsed"]].rename(columns={"away_team":"team"}).assign(side="away")
    ],ignore_index=True)
    if sides.duplicated(["season","week","team"]).any():
        raise ValueError("Ambiguous team-week schedule")
    data=injuries.copy()
    data["injury_row_id"]=range(len(data))
    m=data.merge(sides,on=["season","week","team"],how="left",validate="many_to_one",indicator=True)
    m["date_modified_parsed"]=m.date_modified.map(parse_aware)
    def classify(r):
        if r["_merge"]!="both":return "unmatched_team_week"
        if pd.isna(r.date_modified_parsed):return "timestamp_timezone_unverified"
        if r.date_modified_parsed>=r.kickoff_parsed:return "modified_at_or_after_kickoff"
        return "modified_before_kickoff_unverified"
    m["timing_class"]=m.apply(classify,axis=1)
    m["pregame_source_verified"]=False
    m["eligible_for_model"]=False
    counts={str(x):int(y) for x,y in m.timing_class.value_counts().items()}
    summary={"injury_rows":len(m),"matched_team_week_rows":int((m._merge=="both").sum()),
             "matched_games":int(m.game_id.nunique()),"reference_games":len(k),
             "timing_class_counts":counts,"verified_pregame_rows":0,"model_eligible_rows":0,
             "caution":"Date_modified may reflect retrospective updates; even a pre-kickoff value is not authenticated as contemporaneously available. Naive timestamps are not assigned an assumed timezone."}
    return m.drop(columns=["_merge","kickoff_parsed","date_modified_parsed"]),summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--kickoffs",default="research_outputs/nfl_kickoff_reference.csv")
    p.add_argument("--seasons",type=int,nargs="+",default=[2024,2025])
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    if len(a.seasons)!=len(set(a.seasons)) or any(y<2009 or y>2026 for y in a.seasons):
        raise ValueError("Invalid seasons")
    k=pd.read_csv(a.kickoffs)
    injury=pd.concat([pd.read_csv(BASE.format(season=y),low_memory=False) for y in a.seasons],ignore_index=True)
    result,summary=audit(k,injury)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"nfl_injury_timing_join.csv",index=False)
    (out/"nfl_injury_timing_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
