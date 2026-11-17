"""Inspect historical injurybot RDS schema without treating rows as verified pregame facts."""
from __future__ import annotations
import argparse,io,json,tempfile
from pathlib import Path
import pandas as pd
from research.injury_asset_fingerprint import download,MAX_BYTES
def inspect(assets:pd.DataFrame,fetcher=None,reader=None):
    required={"game_id","asset_found","asset_url","uploaded_before_kickoff"}
    if required-set(assets):raise ValueError("Missing asset columns")
    if assets.game_id.isna().any() or assets.game_id.duplicated().any():raise ValueError("Invalid game IDs")
    fetcher=fetcher or download
    reader=reader or read_rds
    rows=[]
    for a in assets.to_dict("records"):
        found=str(a["asset_found"]).lower()=="true"
        pre=str(a["uploaded_before_kickoff"]).lower()=="true"
        row={"game_id":a["game_id"],"status":"skipped","record_count":0,
          "columns_json":"[]","qb_candidate_columns_json":"[]","sample_values_excluded":True,
          "independent_capture_verified":False,"prediction_cutoff_verified":False,
          "model_eligible":False}
        if found and pre:
            try:
                from urllib.parse import urlparse
                u=a["asset_url"]
                if not isinstance(u,str) or urlparse(u).scheme!="https" or urlparse(u).hostname not in {"github.com","objects.githubusercontent.com","release-assets.githubusercontent.com"}:
                    raise ValueError("Untrusted asset URL")
                data=fetcher(u)
                if not isinstance(data,bytes) or not data or len(data)>MAX_BYTES:raise ValueError("Invalid asset bytes")
                frame=reader(data)
                if not isinstance(frame,pd.DataFrame):raise TypeError("RDS is not a data frame")
                cols=[str(c) for c in frame.columns]
                candidates=[c for c in cols if any(t in c.lower() for t in ("position","player","injury","status","report","date","game","team"))]
                row.update(status="schema_inspected",record_count=len(frame),
                    columns_json=json.dumps(cols),qb_candidate_columns_json=json.dumps(candidates))
            except Exception as exc:
                row["status"]="error_"+type(exc).__name__
        rows.append(row)
    d=pd.DataFrame(rows)
    s={"priority_games":len(d),"schema_inspected":int(d.status.eq("schema_inspected").sum()),
       "schema_errors":int(d.status.str.startswith("error_").sum()),
       "total_rows_in_inspected_assets":int(d.record_count.sum()),
       "qb_records_verified":0,"model_eligible":0,
       "warning":"Schema and row counts only; no QB identities, injury status, historical snapshot authenticity, or prediction-time eligibility established."}
    return d,s

def read_rds(data:bytes):
    import pyreadr
    with tempfile.NamedTemporaryFile(suffix=".rds") as f:
        f.write(data);f.flush()
        objects=pyreadr.read_r(f.name)
    frames=[v for v in objects.values() if isinstance(v,pd.DataFrame)]
    if len(frames)!=1:raise ValueError("Expected exactly one data frame in RDS")
    return frames[0]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",default="research_outputs/nfl_2024_qb_asset_kickoff_timing.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=inspect(pd.read_csv(a.assets))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_2024_qb_rds_schema.csv",index=False)
    (out/"nfl_2024_qb_rds_schema_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
