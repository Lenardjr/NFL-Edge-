"""Fetch current public NFLverse schedule and PBP, then run pregame V3 research.

This is an on-demand runner, not a historical reconstruction. It never
automatically records ledger entries, trades, or claims executable odds.
Downloaded source bytes are hashed for reproducibility. Data provenance
(publication timing and completeness) is NOT independently verified.
"""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from research.run_nfl_v3 import SCHEDULE_URL,PBP_URL
from research.v3_pbp_features import aggregate_games
from research.pregame_v3_bundle import run_bundle

def download_public_snapshot(*,output_dir,first_season=2018,last_season=None):
    now=datetime.now(timezone.utc).isoformat()
    year=pd.Timestamp(now).year
    last_season=year if last_season is None else last_season
    if first_season>last_season or first_season<1999 or last_season>year:
        raise ValueError("Invalid season range")
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    schedule=out/"nflverse_schedule_source.csv"
    # Retain the exact downloaded source bytes and a SHA256 fingerprint.
    import urllib.request
    def fetch(url,path):
        with urllib.request.urlopen(url,timeout=90) as response,open(path,"wb") as f:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                f.write(chunk)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes={"schedule":fetch(SCHEDULE_URL,schedule)}
    parts=[]
    for season in range(first_season,last_season+1):
        source=out/f"nflverse_pbp_{season}.parquet"
        hashes[str(season)]=fetch(PBP_URL.format(season=season),source)
        part=pd.read_parquet(source,columns=["game_id","posteam","defteam","epa","play_type","season","week"])
        parts.append(aggregate_games(part))
    epa=out/"nflverse_team_game_epa.csv"
    pd.concat(parts,ignore_index=True).to_csv(epa,index=False)
    (out/"nflverse_download_manifest.json").write_text(json.dumps({
        "captured_at_utc":now,"schedule_url":SCHEDULE_URL,"pbp_url_pattern":PBP_URL,
        "first_season":first_season,"last_season":last_season,"sha256":hashes,
        "independent_publication_time_verified":False},indent=2))
    return schedule,epa

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default="research_outputs/live_v3")
    p.add_argument("--first-season",type=int,default=2018)
    p.add_argument("--last-season",type=int,default=None)
    a=p.parse_args()
    out=Path(a.output_dir)
    schedule,epa=download_public_snapshot(output_dir=out/"sources",
        first_season=a.first_season,last_season=a.last_season)
    # Deliberately no --ledger option: first verify outputs and source coverage.
    result=run_bundle(schedule_csv=schedule,epa_csv=epa,output_dir=out/"predictions")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
