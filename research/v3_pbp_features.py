"""Convert nflverse play-by-play to game-level EPA features.

Produces *completed-game* aggregates only. These are NOT safe as pregame
features until shifted to strictly prior games by a separate date-safe join.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REQUIRED={"game_id","posteam","defteam","epa","play_type","season","week"}
def aggregate_games(pbp:pd.DataFrame)->pd.DataFrame:
    missing=REQUIRED-set(pbp.columns)
    if missing:raise ValueError(f"Missing play-by-play columns: {sorted(missing)}")
    p=pbp.copy()
    p["epa"]=pd.to_numeric(p["epa"],errors="coerce")
    p=p[p["play_type"].isin(["pass","run"]) & p.epa.notna() & p.posteam.notna() & p.defteam.notna()]
    if p.empty:raise ValueError("No eligible scrimmage plays with EPA")
    if (p.posteam==p.defteam).any():raise ValueError("Offense and defense cannot be same team")
    p["success"]=(p.epa>0).astype(float)
    out=p.groupby(["game_id","season","week","posteam"],as_index=False).agg(
        offense_epa_per_play=("epa","mean"),offense_success_rate=("success","mean"),
        eligible_plays=("epa","size"))
    if out.duplicated(["game_id","posteam"]).any():raise ValueError("Duplicate team-game aggregates")
    return out.rename(columns={"posteam":"team"})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--pbp",required=True)
    parser.add_argument("--output",default="research_outputs/nfl_v3_game_epa.csv")
    args=parser.parse_args()
    source=Path(args.pbp)
    if source.suffix==".parquet":pbp=pd.read_parquet(source)
    else:pbp=pd.read_csv(source,low_memory=False)
    result=aggregate_games(pbp)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(path,index=False)
    print(f"Exported {len(result)} team-game EPA records; not pregame-safe until lagged")
if __name__=="__main__":main()
