"""Join extreme-case reviews to nflverse schedule metadata, with prior-game rest.

Historical schedule metadata is descriptive. It does not establish availability of
injury reports, quarterback news, betting quotes, or weather at prediction time.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

SCHEDULE_URL="https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
MATCH_FIELDS=["game_id","gameday","home_team","away_team","home_rest_days","away_rest_days","rest_advantage_home"]
def enrich(cases:pd.DataFrame,schedules:pd.DataFrame):
    if cases.game_id.isna().any() or cases.game_id.duplicated().any():
        raise ValueError("Invalid case game IDs")
    required={"game_id","gameday","home_team","away_team","season","week"}
    if required-set(schedules):raise ValueError("Missing schedule fields: "+str(sorted(required-set(schedules))))
    s=schedules[list(required)].copy()
    if s.game_id.isna().any() or s.game_id.duplicated().any():raise ValueError("Invalid schedule game IDs")
    s["gameday"]=pd.to_datetime(s.gameday,errors="coerce")
    if s.gameday.isna().any():raise ValueError("Invalid game dates")
    if s[["home_team","away_team"]].isna().any().any():raise ValueError("Missing teams")
    # Only earlier games in the same season count; no future outcomes are used.
    long=pd.concat([s[["game_id","season","gameday","home_team"]].rename(columns={"home_team":"team"}),
                    s[["game_id","season","gameday","away_team"]].rename(columns={"away_team":"team"})],ignore_index=True)
    long=long.sort_values(["season","team","gameday","game_id"],kind="stable")
    # Previous distinct calendar date, excluding same-day games.
    distinct=long[["season","team","gameday"]].drop_duplicates().sort_values(["season","team","gameday"])
    distinct["previous_date"]=distinct.groupby(["season","team"]).gameday.shift(1)
    distinct["rest_days"]=(distinct.gameday-distinct.previous_date).dt.days-1
    rest=long.merge(distinct,on=["season","team","gameday"],validate="many_to_one")
    home=rest[rest.game_id.isin(s.game_id)].merge(s[["game_id","home_team"]],on="game_id")
    home=home[home.team==home.home_team][["game_id","rest_days"]].rename(columns={"rest_days":"home_rest_days"})
    away=rest.merge(s[["game_id","away_team"]],on="game_id")
    away=away[away.team==away.away_team][["game_id","rest_days"]].rename(columns={"rest_days":"away_rest_days"})
    s=s.merge(home,on="game_id",validate="one_to_one").merge(away,on="game_id",validate="one_to_one")
    s["rest_advantage_home"]=s.home_rest_days-s.away_rest_days
    result=cases.merge(s[MATCH_FIELDS],on="game_id",how="left",validate="one_to_one",indicator=True)
    if not (result["_merge"]=="both").all():raise ValueError("Cases missing from schedules")
    result=result.drop(columns="_merge")
    if "season" in cases:
        check=cases[["game_id","season"]].merge(s[["game_id","season"]],on="game_id",suffixes=("_case","_schedule"))
        if not (check.season_case==check.season_schedule).all():raise ValueError("Season mismatch")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cases",default="research_outputs/nfl_v5_extreme_case_review.csv")
    p.add_argument("--schedules",default=SCHEDULE_URL)
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d=enrich(pd.read_csv(a.cases),pd.read_csv(a.schedules))
    d.to_csv(out/"nfl_v5_extreme_cases_with_matchups.csv",index=False)
    summary={"cases":len(d),"matched":int(d.gameday.notna().sum()),
             "with_both_rest_values":int(d[["home_rest_days","away_rest_days"]].notna().all(axis=1).sum()),
             "extreme_20pt_cases":int((d.absolute_gap>=.2).sum()),
             "caution":"Rest is calendar days since last scheduled game within season; not injury/QB evidence. Matchups are historical metadata. Verify contemporaneous pregame sources before making causal claims."}
    (out/"nfl_v5_matchup_enrichment_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
