"""Build an evidence timeline without inferring historical decision times or odds."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def timeline(evidence:pd.DataFrame,kickoffs:pd.DataFrame):
    needed={"game_id","side","qb_name","source_url","published_date","game_date","in_priority_queue"}
    if needed-set(evidence):raise ValueError("Missing evidence columns")
    if not {"game_id","kickoff_utc"}<=set(kickoffs):raise ValueError("Missing kickoff columns")
    if evidence.duplicated(["game_id","side"]).any() or kickoffs.game_id.duplicated().any():
        raise ValueError("Duplicate game or side")
    k=kickoffs[["game_id","kickoff_utc"]].copy()
    d=evidence.merge(k,on="game_id",how="left",validate="many_to_one")
    publication=pd.to_datetime(d.published_date,errors="coerce",utc=True)
    kickoff=pd.to_datetime(d.kickoff_utc,errors="coerce",utc=True)
    # Dates alone do not establish publication hour. Do not compare midnight to a betting cutoff.
    d["published_date_before_kickoff_day"]=publication.notna() & kickoff.notna() & (publication.dt.date<kickoff.dt.date)
    d["source_publication_timestamp_utc"]=""
    d["prediction_decision_timestamp_utc"]=""
    d["market_quote_timestamp_utc"]=""
    d["market_quote_source"]=""
    d["market_quote_american_odds"]=""
    d["independent_capture_url"]=""
    d["independent_capture_timestamp_utc"]=""
    d["source_verified_at_decision"]=False
    d["quote_verified_at_decision"]=False
    d["model_eligible"]=False
    d["research_status"]="requires_independent_timestamps_and_market_quote"
    summary={"cases":len(d),"priority_matches":int(d.in_priority_queue.astype(str).str.lower().eq("true").sum()),
             "publication_date_before_kickoff_day":int(d.published_date_before_kickoff_day.sum()),
             "kickoff_missing":int(kickoff.isna().sum()),"verified_source_at_decision":0,
             "verified_market_quotes_at_decision":0,"model_eligible":0,
             "warning":"A dated article is not a timestamped pre-decision source. No historical quote or prediction time has been inferred."}
    return d,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--evidence",default="research_outputs/nfl_qb_documented_priority_cross_reference.csv")
    p.add_argument("--kickoffs",default="research_outputs/nfl_kickoff_reference.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=timeline(pd.read_csv(a.evidence),pd.read_csv(a.kickoffs))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_qb_announcement_decision_timeline.csv",index=False)
    (out/"nfl_qb_announcement_decision_timeline_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
