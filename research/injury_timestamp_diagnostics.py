"""Diagnose injury date_modified formats and QB availability coverage.

Descriptive audit only. No timezone inference, no pregame feature promotion.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import pandas as pd

def timestamp_format(value):
    if pd.isna(value) or not str(value).strip():return "missing"
    s=str(value).strip()
    if re.search(r"(?:Z|[+-]\d{2}:?\d{2})$",s,re.I):return "explicit_offset"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?",s):
        return "naive_iso"
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?",s):
        return "naive_us"
    return "other_or_unparseable"

def analyze(joined):
    required={"season","date_modified","timing_class","game_id","team","week"}
    if required-set(joined):raise ValueError("Missing injury join columns")
    if joined.empty:raise ValueError("Empty injury join")
    d=joined.copy()
    d["timestamp_format"]=d.date_modified.map(timestamp_format)
    d["position_clean"]=d.position.fillna("").astype(str).str.strip().str.upper() if "position" in d else ""
    by_season=[]
    for year,g in d.groupby("season",sort=True):
        qb=g[g.position_clean=="QB"]
        by_season.append({"season":int(year),"rows":len(g),
            "timestamp_formats":{str(k):int(v) for k,v in g.timestamp_format.value_counts().items()},
            "timing_classes":{str(k):int(v) for k,v in g.timing_class.value_counts().items()},
            "qb_rows":len(qb),"qb_matched_rows":int(qb.game_id.notna().sum()),
            "qb_matched_games":int(qb.game_id.nunique()),
            "qb_before_kickoff_unverified":int((qb.timing_class=="modified_before_kickoff_unverified").sum())})
    result={"injury_rows":len(d),"timestamp_formats":{str(k):int(v) for k,v in d.timestamp_format.value_counts().items()},
            "seasons":by_season,"verified_pregame_rows":0,"model_eligible_rows":0,
            "interpretation":"Naive dates have no authenticated timezone. Explicit offsets establish parseability, not historical availability. QB reports may omit healthy players."}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--joined",default="research_outputs/nfl_injury_timing_join.csv")
    p.add_argument("--output",default="research_outputs/nfl_injury_timestamp_diagnostics.json")
    a=p.parse_args()
    report=analyze(pd.read_csv(a.joined,low_memory=False))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
