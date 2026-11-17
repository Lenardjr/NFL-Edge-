"""Extract QB injury candidates as unverified research evidence only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from research.injury_rds_schema import read_rds
from research.injury_asset_fingerprint import download,MAX_BYTES

def extract(assets,fetcher=None,reader=None):
    required={"game_id","asset_found","asset_url","uploaded_before_kickoff"}
    if required-set(assets):raise ValueError("Missing asset fields")
    if assets.game_id.isna().any() or assets.game_id.duplicated().any():raise ValueError("Invalid game IDs")
    fetcher=fetcher or download;reader=reader or read_rds
    rows=[];issues=[]
    for a in assets.to_dict("records"):
        game=a["game_id"]
        if str(a["asset_found"]).lower()!="true" or str(a["uploaded_before_kickoff"]).lower()!="true":
            issues.append({"game_id":game,"status":"not_available_for_review"});continue
        try:
            url=a["asset_url"];p=urlparse(url) if isinstance(url,str) else None
            if p is None or p.scheme!="https" or p.hostname not in {"github.com","objects.githubusercontent.com","release-assets.githubusercontent.com"}:
                raise ValueError("Untrusted asset URL")
            data=fetcher(url)
            if not isinstance(data,bytes) or not data or len(data)>MAX_BYTES:raise ValueError("Invalid bytes")
            frame=reader(data)
            if not isinstance(frame,pd.DataFrame):raise TypeError("Not a data frame")
            names={str(c).strip().lower():c for c in frame.columns}
            pos=next((names[k] for k in ("position","pos","player_position") if k in names),None)
            if pos is None:
                issues.append({"game_id":game,"status":"position_column_unmapped"});continue
            qb=frame[frame[pos].astype("string").str.strip().str.upper().eq("QB").fillna(False)]
            fields={"player":("player_name","player","full_name","name"),
                    "team":("team","team_abbr","team_abbreviation"),
                    "injury":("injury","injury_type","injury_description"),
                    "practice_status":("practice_status","practice"),
                    "game_status":("game_status","status","report_status"),
                    "report_date":("report_date","date","date_modified")}
            mapped={k:next((names[n] for n in variants if n in names),None) for k,variants in fields.items()}
            for _,v in qb.iterrows():
                row={"game_id":game,"asset_url":url,"position":"QB"}
                for key,col in mapped.items():
                    value=v[col] if col is not None else None
                    row[key]="" if value is None or pd.isna(value) else str(value)
                row.update(source_verified=False,prediction_cutoff_verified=False,model_eligible=False)
                rows.append(row)
            issues.append({"game_id":game,"status":"qb_rows_extracted","qb_candidate_rows":len(qb),
                           "mapped_columns_json":json.dumps({k:str(v) if v is not None else "" for k,v in mapped.items()})})
        except Exception as exc:
            issues.append({"game_id":game,"status":"error_"+type(exc).__name__})
    columns=["game_id","asset_url","position","player","team","injury","practice_status","game_status","report_date","source_verified","prediction_cutoff_verified","model_eligible"]
    d=pd.DataFrame(rows,columns=columns);audit=pd.DataFrame(issues)
    s={"priority_games":len(assets),"files_with_extraction":int(audit.status.eq("qb_rows_extracted").sum()),
       "qb_candidate_rows":len(d),"files_with_errors":int(audit.status.str.startswith("error_").sum()),
       "files_with_unmapped_position":int(audit.status.eq("position_column_unmapped").sum()),
       "pregame_verified_qb_rows":0,"model_eligible":0,
       "warning":"Extracted records are unverified retrospective candidates; do not use as model features."}
    return d,audit,s

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",default="research_outputs/nfl_2024_qb_asset_kickoff_timing.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,audit,s=extract(pd.read_csv(a.assets))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_2024_qb_rds_extracted_candidates.csv",index=False)
    audit.to_csv(out/"nfl_2024_qb_rds_extraction_audit.csv",index=False)
    (out/"nfl_2024_qb_rds_extraction_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
