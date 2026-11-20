"""Audit QB injury candidate completeness and timestamp ambiguity; never certify pregame."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

FIELDS=("player","team","injury","practice_status","game_status","report_date")
def audit(candidates:pd.DataFrame, timing:pd.DataFrame):
    if not {"game_id",*FIELDS}<=set(candidates):raise ValueError("Missing candidate columns")
    if not {"game_id","kickoff_utc"}<=set(timing):raise ValueError("Missing kickoff reference")
    if timing.game_id.isna().any() or timing.game_id.duplicated().any():raise ValueError("Invalid kickoff keys")
    d=candidates.copy()
    if d.game_id.isna().any():raise ValueError("Missing candidate game ID")
    d=d.merge(timing[["game_id","kickoff_utc"]],on="game_id",how="left",validate="many_to_one",indicator=True)
    for f in FIELDS:
        d[f]=d[f].fillna("").astype(str).str.strip()
        d[f+"_present"]=d[f].ne("")
    d["identity_complete"]=d.player_present & d.team_present
    d["designation_present"]=d.game_status_present | d.practice_status_present
    d["exact_duplicate"]=d.duplicated(subset=["game_id",*FIELDS],keep=False)
    report=pd.to_datetime(d.report_date,utc=True,errors="coerce")
    kickoff=pd.to_datetime(d.kickoff_utc,utc=True,errors="coerce")
    # Explicit timezone offsets only; a date-only value is not a verified timestamp.
    offset=d.report_date.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$",regex=True,case=False)
    d["report_timestamp_parseable"]=report.notna() & offset
    d["report_timestamp_before_kickoff"]=d.report_timestamp_parseable & kickoff.notna() & report.lt(kickoff)
    d["source_verified"]=False
    d["prediction_cutoff_verified"]=False
    d["model_eligible"]=False
    d["review_status"]="needs_independent_contemporaneous_source"
    d.loc[~d.identity_complete,"review_status"]="identity_fields_incomplete"
    d.loc[d._merge.ne("both"),"review_status"]="kickoff_reference_missing"
    d=d.drop(columns=["_merge"])
    s={"qb_candidate_rows":len(d),"identity_complete":int(d.identity_complete.sum()),
       "designation_present":int(d.designation_present.sum()),
       "report_timestamp_parseable_with_offset":int(d.report_timestamp_parseable.sum()),
       "report_timestamp_before_kickoff_unverified":int(d.report_timestamp_before_kickoff.sum()),
       "exact_duplicate_rows":int(d.exact_duplicate.sum()),
       "kickoff_reference_missing":int(d.kickoff_utc.isna().sum()),
       "independently_verified":0,"model_eligible":0,
       "warning":"File contents and timestamps are retrospective. Even before-kickoff timestamps do not prove prediction-time publication or independent capture."}
    return d,s

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--candidates",default="research_outputs/nfl_2024_qb_rds_extracted_candidates.csv")
    p.add_argument("--timing",default="research_outputs/nfl_2024_qb_asset_kickoff_timing.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=audit(pd.read_csv(a.candidates),pd.read_csv(a.timing))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_2024_qb_rds_record_audit.csv",index=False)
    (out/"nfl_2024_qb_rds_record_audit_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
