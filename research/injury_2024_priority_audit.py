"""2024 injury timestamp lead-time audit and QB extreme-gap prioritization.

Descriptive only. Retrospective releases cannot authenticate historical availability.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

BINS=[-1e9,0,6,24,48,72,168,1e9]
LABELS=["at_or_after_kickoff","0_to_6h","6_to_24h","1_to_2d","2_to_3d","3_to_7d","over_7d"]
def analyze(joined:pd.DataFrame,extreme:pd.DataFrame):
    need={"season","game_id","team","side","position","date_modified","kickoff_utc","timing_class"}
    if need-set(joined):raise ValueError("Missing injury columns")
    if "game_id" not in extreme:raise ValueError("Missing extreme game_id")
    if extreme.game_id.isna().any() or extreme.game_id.duplicated().any():
        raise ValueError("Invalid extreme IDs")
    d=joined.loc[joined.season==2024].copy()
    if d.empty:raise ValueError("No 2024 injury rows")
    d["kickoff_dt"]=pd.to_datetime(d.kickoff_utc,utc=True,errors="coerce")
    d["modified_dt"]=pd.to_datetime(d.date_modified,utc=True,errors="coerce")
    d["lead_hours"]=(d.kickoff_dt-d.modified_dt).dt.total_seconds()/3600
    d["lead_bucket"]=pd.cut(d.lead_hours,bins=BINS,labels=LABELS,right=True)
    d["is_qb"]=d.position.fillna("").astype(str).str.upper().str.strip().eq("QB")
    d["extreme_gap_game"]=d.game_id.isin(set(extreme.game_id))
    qb=d[d.is_qb & d.extreme_gap_game & d.game_id.notna()].copy()
    cols=["game_id","team","side","date_modified","kickoff_utc","lead_hours","timing_class"]
    for optional in ["full_name","report_status","practice_status","report_primary_injury"]:
        if optional in qb:cols.append(optional)
    qb=qb[cols].sort_values(["game_id","team","date_modified"])
    buckets={str(x):int((d.lead_bucket==x).sum()) for x in LABELS}
    result={"season":2024,"injury_rows":len(d),"matched_rows":int(d.game_id.notna().sum()),
            "timestamp_parseable_rows":int(d.modified_dt.notna().sum()),
            "lead_time_buckets":buckets,
            "qb_rows":int(d.is_qb.sum()),"qb_extreme_gap_rows":len(qb),
            "qb_extreme_gap_games":int(qb.game_id.nunique()),
            "qb_extreme_gap_records_with_prior_timestamp":int((qb.lead_hours>0).sum()),
            "verified_pregame_rows":0,"model_eligible_rows":0,
            "warning":"Modification timestamps from retrospective injury releases are not independently authenticated capture times; no model feature eligibility inferred."}
    return qb,result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--joined",default="research_outputs/nfl_injury_timing_join.csv")
    p.add_argument("--extreme",default="research_outputs/nfl_v5_extreme_case_review.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    qb,summary=analyze(pd.read_csv(a.joined,low_memory=False),pd.read_csv(a.extreme))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    qb.to_csv(out/"nfl_2024_qb_extreme_injury_review.csv",index=False)
    (out/"nfl_2024_injury_leadtime_audit.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
