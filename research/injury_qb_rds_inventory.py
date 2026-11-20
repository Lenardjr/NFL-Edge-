"""Inventory quarterback rows in injurybot RDS assets, without asserting pregame validity."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from research.injury_rds_schema import read_rds
from research.injury_asset_fingerprint import download,MAX_BYTES

POSITION_NAMES=("position","pos","player_position")
def summarize(assets:pd.DataFrame,fetcher=None,reader=None):
    required={"game_id","asset_found","asset_url","uploaded_before_kickoff"}
    if required-set(assets):raise ValueError("Missing asset fields")
    if assets.game_id.isna().any() or assets.game_id.duplicated().any():raise ValueError("Duplicate or null games")
    fetcher=fetcher or download;reader=reader or read_rds
    rows=[]
    for a in assets.to_dict("records"):
        row={"game_id":a["game_id"],"status":"skipped","total_records":0,"qb_candidate_records":0,
             "position_column":"","available_columns":"[]","model_eligible":False}
        if str(a["asset_found"]).lower()=="true" and str(a["uploaded_before_kickoff"]).lower()=="true":
            try:
                from urllib.parse import urlparse
                u=a["asset_url"]
                if not isinstance(u,str) or urlparse(u).scheme!="https" or urlparse(u).hostname not in {"github.com","objects.githubusercontent.com","release-assets.githubusercontent.com"}:raise ValueError("Untrusted URL")
                raw=fetcher(u)
                if not isinstance(raw,bytes) or not raw or len(raw)>MAX_BYTES:raise ValueError("Invalid bytes")
                frame=reader(raw)
                if not isinstance(frame,pd.DataFrame):raise TypeError("Not a dataframe")
                columns={str(c).strip().lower():c for c in frame.columns}
                matched=[columns[x] for x in POSITION_NAMES if x in columns]
                row["total_records"]=len(frame)
                row["available_columns"]=json.dumps([str(x) for x in frame.columns])
                if matched:
                    pos=matched[0]
                    row["position_column"]=str(pos)
                    row["qb_candidate_records"]=int(frame[pos].astype("string").str.strip().str.upper().eq("QB").fillna(False).sum())
                    row["status"]="qb_candidates_counted"
                else:row["status"]="position_schema_unmapped"
            except Exception as exc:row["status"]="error_"+type(exc).__name__
        rows.append(row)
    d=pd.DataFrame(rows)
    s={"priority_games":len(d),"qb_schema_mapped_files":int(d.status.eq("qb_candidates_counted").sum()),
       "unmapped_position_schema_files":int(d.status.eq("position_schema_unmapped").sum()),
       "read_errors":int(d.status.str.startswith("error_").sum()),
       "qb_candidate_rows":int(d.qb_candidate_records.sum()),
       "qb_records_pregame_verified":0,"model_eligible":0,
       "warning":"QB candidate counts are retrospective file contents; no identities or availability claims are validated."}
    return d,s

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",default="research_outputs/nfl_2024_qb_asset_kickoff_timing.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=summarize(pd.read_csv(a.assets))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_2024_qb_rds_candidate_inventory.csv",index=False)
    (out/"nfl_2024_qb_rds_candidate_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
