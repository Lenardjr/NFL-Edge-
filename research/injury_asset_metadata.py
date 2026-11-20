"""Inspect GitHub release asset metadata for historical game injury files.

Metadata is source discovery only; never treat an asset upload as proof of
public injury information at prediction time.
"""
from __future__ import annotations
import argparse,json,urllib.request
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd

API="https://api.github.com/repos/nflverse/nflverse-injurybot/releases/tags/injuries_2024"
def inspect(priority:pd.DataFrame,release:dict):
    if not {"game_id","priority_rank","season","week"}<=set(priority):
        raise ValueError("Missing priority columns")
    if priority.game_id.isna().any() or priority.game_id.duplicated().any():
        raise ValueError("Invalid game IDs")
    assets=release.get("assets",[])
    if not isinstance(assets,list):raise ValueError("Invalid release assets")
    index={}
    for asset in assets:
        name=asset.get("name")
        if name in index:raise ValueError("Duplicate release asset")
        index[name]=asset
    rows=[]
    for g in priority.itertuples(index=False):
        name=str(g.game_id)+".rds"
        asset=index.get(name)
        uploaded=asset.get("created_at","") if asset else ""
        url=asset.get("browser_download_url","") if asset else ""
        if uploaded:
            t=pd.to_datetime(uploaded,utc=True,errors="coerce")
            if pd.isna(t):raise ValueError("Invalid upload timestamp")
        rows.append({"game_id":g.game_id,"priority_rank":int(g.priority_rank),
            "asset_name":name,"asset_found":bool(asset),"asset_url":url,
            "asset_uploaded_at":uploaded,"asset_id":asset.get("id","") if asset else "",
            "content_inspected":False,"independent_archive_verified":False,
            "prediction_cutoff_verified":False,"model_eligible":False})
    d=pd.DataFrame(rows,columns=["game_id","priority_rank","asset_name","asset_found","asset_url",
        "asset_uploaded_at","asset_id","content_inspected","independent_archive_verified",
        "prediction_cutoff_verified","model_eligible"])
    summary={"priority_games":len(d),"assets_found":int(d.asset_found.sum()),
        "assets_missing":int((~d.asset_found).sum()),"verified_pregame":0,"model_eligible":0,
        "release_api":API,
        "warning":"Asset metadata is not proof of content, contemporaneous publication, or availability before prediction."}
    return d,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--priority",default="research_outputs/nfl_2024_qb_priority_games.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    req=urllib.request.Request(API,headers={"Accept":"application/vnd.github+json","User-Agent":"NFL-Edge-research"})
    with urllib.request.urlopen(req,timeout=30) as response:
        release=json.load(response)
    d,summary=inspect(pd.read_csv(a.priority),release)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_2024_qb_asset_metadata.csv",index=False)
    (out/"nfl_2024_qb_asset_metadata_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
