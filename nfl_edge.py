from __future__ import annotations
# NFL Edge v2.0 — standalone Streamlit app; Game Engine 1.0 protected
# Data source attribution: nflverse (CC-BY 4.0 where applicable).

import math
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEAM_ALIASES={"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}
FEATURES=["elo_diff","pf_matchup","form_diff","winform_diff",
          "home_pf5","away_pf5","home_pa5","away_pa5",
          "home_games_prior","away_games_prior","rest_diff","home_rest","away_rest"]

def load_games(url):
    df=pd.read_csv(url)
    for c in ["home_team","away_team"]:
        df[c]=df[c].map(lambda x: TEAM_ALIASES.get(str(x),str(x)))
    df["gameday"]=pd.to_datetime(df["gameday"],errors="coerce")
    for c in ["home_score","away_score","spread_line","total_line"]:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.sort_values(["gameday","gametime"],na_position="last").reset_index(drop=True)

def pregame_features(df):
    rows=[]; state={}; last_season=None
    def team(t):
        return state.setdefault(t,{"games":0,"pf":[],"pa":[],"margin":[],"wins":[],"elo":1500.,"last_date":None})
    def avg(v,n,d): return float(np.mean(v[-n:])) if v else d
    for _,g in df.iterrows():
        season=int(g.season) if pd.notna(g.get("season",np.nan)) else None
        if last_season is not None and season is not None and season!=last_season:
            for s in state.values():
                s["elo"]=1500.+0.67*(s["elo"]-1500.)
                s["games"]=0; s["pf"]=[]; s["pa"]=[]; s["margin"]=[]; s["wins"]=[]; s["last_date"]=None
        last_season=season if season is not None else last_season
        h,a=team(g.home_team),team(g.away_team)
        gd=pd.to_datetime(g.get("gameday"),errors="coerce")
        def rest_days(s):
            if s["last_date"] is None or pd.isna(gd): return 7.0
            return float(np.clip((gd-s["last_date"]).days,3,21))
        hr,ar=rest_days(h),rest_days(a)
        r=g.to_dict()
        r.update({
            "home_elo":h["elo"],"away_elo":a["elo"],"elo_diff":h["elo"]-a["elo"],
            "home_pf5":avg(h["pf"],5,22),"away_pf5":avg(a["pf"],5,22),
            "home_pa5":avg(h["pa"],5,22),"away_pa5":avg(a["pa"],5,22),
            "home_margin5":avg(h["margin"],5,0),"away_margin5":avg(a["margin"],5,0),
            "home_win5":avg(h["wins"],5,.5),"away_win5":avg(a["wins"],5,.5),
            "home_games_prior":h["games"],"away_games_prior":a["games"],
            "home_rest":hr,"away_rest":ar,"rest_diff":hr-ar})
        rows.append(r)
        if pd.notna(g.home_score) and pd.notna(g.away_score):
            hs,aw=float(g.home_score),float(g.away_score); margin=hs-aw
            exp=1/(1+10**((a["elo"]-(h["elo"]+55))/400))
            actual=1 if margin>0 else (.5 if margin==0 else 0)
            delta=20*math.log(abs(margin)+1)*(actual-exp)
            h["elo"]+=delta; a["elo"]-=delta
            for s,pf,pa,m,w in [(h,hs,aw,margin,actual),(a,aw,hs,-margin,1-actual)]:
                s["games"]+=1; s["pf"].append(pf); s["pa"].append(pa); s["margin"].append(m); s["wins"].append(w)
                s["last_date"]=gd if pd.notna(gd) else s["last_date"]
    out=pd.DataFrame(rows)
    out["target_margin"]=out.home_score-out.away_score
    out["target_total"]=out.home_score+out.away_score
    out["pf_matchup"]=(out.home_pf5+out.away_pa5)-(out.away_pf5+out.home_pa5)
    out["form_diff"]=out.home_margin5-out.away_margin5
    out["winform_diff"]=out.home_win5-out.away_win5
    return out

def pipe(alpha):
    return Pipeline([("impute",SimpleImputer(strategy="median")),
                     ("scale",StandardScaler()),("model",Ridge(alpha=alpha))])

class NFLEdgeModel:
    def __init__(self):
        self.margin=pipe(25); self.total=pipe(30)
        self.win_model=Pipeline([("impute",SimpleImputer(strategy="median")),
                                 ("scale",StandardScaler()),
                                 ("model",LogisticRegression(C=.45,max_iter=2000))])
    def fit(self,games):
        f=pregame_features(games); t=f.dropna(subset=["target_margin","target_total"]).copy()
        mature=t[(t.home_games_prior>=2)&(t.away_games_prior>=2)]
        if len(mature)>100: t=mature
        self.margin.fit(t[FEATURES],t.target_margin); self.total.fit(t[FEATURES],t.target_total)
        pm=self.margin.predict(t[FEATURES]); pt=self.total.predict(t[FEATURES])
        self.margin_residuals=np.asarray(t.target_margin-pm,dtype=float)
        self.total_residuals=np.asarray(t.target_total-pt,dtype=float)
        self.win_model.fit(t[FEATURES],(t.target_margin>0).astype(int))
        self.feature_df=f; return self
    def predict(self,row,n=50000,seed=42):
        X=pd.DataFrame([{f:row[f] for f in FEATURES}])
        margin=float(self.margin.predict(X)[0]); total=float(self.total.predict(X)[0])
        rng=np.random.default_rng(seed)
        margins=margin+rng.choice(self.margin_residuals,size=n,replace=True)
        totals=total+rng.choice(self.total_residuals,size=n,replace=True)
        hs=np.maximum(0,(totals+margins)/2); aws=np.maximum(0,(totals-margins)/2)
        sim_home=float(np.mean(hs>aws)); calibrated=float(self.win_model.predict_proba(X)[0,1])
        home_win=.5*sim_home+.5*calibrated
        out={"home_score":max(6,(total+margin)/2),"away_score":max(6,(total-margin)/2),
             "margin":margin,"total":total,"home_win_prob":home_win,
             "sim_home_win_prob":sim_home,"calibrated_home_win_prob":calibrated,
             "home_scores":hs,"away_scores":aws}
        spread=row.get("spread_line",np.nan)
        if pd.notna(spread): out["home_cover_prob"]=float(np.mean(margins-float(spread)>0))
        tl=row.get("total_line",np.nan)
        if pd.notna(tl): out["over_prob"]=float(np.mean(totals>float(tl)))
        if pd.notna(spread) and float(spread)!=0:
            fav_home=float(spread)>0
            upset=(1-home_win) if fav_home else home_win
            fav_model=margin if fav_home else -margin
            out["upset_prob"]=upset
            out["upset_score"]=min(100,100*upset+2.5*max(0,abs(float(spread))-fav_model))
        return out

def walk_forward_backtest(games,first_test_season=2022):
    f=pregame_features(games); done=f.dropna(subset=["target_margin","target_total"]).copy()
    done=done.sort_values(["gameday","gametime"],na_position="last")
    outs=[]
    test=done[(done.season>=first_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min()
        tr=done[done.gameday<cutoff].copy()
        tr=tr[(tr.home_games_prior>=2)&(tr.away_games_prior>=2)]
        if len(tr)<300: continue
        mm,tm=pipe(25),pipe(30)
        wm=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                     ("model",LogisticRegression(C=.45,max_iter=2000))])
        mm.fit(tr[FEATURES],tr.target_margin); tm.fit(tr[FEATURES],tr.target_total)
        wm.fit(tr[FEATURES],(tr.target_margin>0).astype(int))
        pm,pt=mm.predict(te[FEATURES]),tm.predict(te[FEATURES]); wp=wm.predict_proba(te[FEATURES])[:,1]
        for (_,r),margin,total,p in zip(te.iterrows(),pm,pt,wp):
            spread=r.get("spread_line",np.nan); gap=np.nan; ats=np.nan; side="NO LINE"
            if pd.notna(spread):
                gap=float(margin-spread); cover=float(r.target_margin-spread)
                side="HOME" if gap>0 else "AWAY" if gap<0 else "PASS"
                if abs(cover)>1e-9 and side!="PASS":
                    ats=bool(cover>0) if side=="HOME" else bool(cover<0)
            actual_home=float(r.target_margin>0)
            outs.append({"season":int(season),"week":int(week),"gameday":r.gameday,
                         "Game":f"{r.away_team} @ {r.home_team}",
                         "actual_margin":r.target_margin,"pred_margin":float(margin),
                         "actual_total":r.target_total,"pred_total":float(total),
                         "home_win_prob":float(p),"winner_correct":bool((p>=.5)==bool(actual_home)),
                         "brier":float((p-actual_home)**2),
                         "margin_abs_error":abs(margin-r.target_margin),"total_abs_error":abs(total-r.target_total),
                         "market_spread":spread,"model_market_margin_gap":gap,"model_side":side,"ats_result":ats})
    return pd.DataFrame(outs)

def calibration_table(bt):
    if bt.empty:return pd.DataFrame()
    x=bt.copy(); x["actual_home_win"]=(x.actual_margin>0).astype(float)
    x["bucket"]=pd.cut(x.home_win_prob,bins=[0,.4,.5,.6,1],include_lowest=True)
    return x.groupby("bucket",observed=True).agg(Games=("Game","size"),
        Predicted=("home_win_prob","mean"),Actual=("actual_home_win","mean")).reset_index()


import numpy as np
import pandas as pd

PLAYER_URL_TEMPLATE="https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"

def _normalize_player_team_column(d):
    """Normalize nflverse team-field schema changes to recent_team."""
    if "recent_team" in d.columns:
        return d
    for candidate in ["team","team_abbr","club_code"]:
        if candidate in d.columns:
            return d.rename(columns={candidate:"recent_team"})
    d=d.copy(); d["recent_team"]=pd.NA
    return d

def load_player_stats(start_season=2020, end_season=2026):
    frames=[]
    for season in range(start_season,end_season+1):
        try:
            x=pd.read_csv(PLAYER_URL_TEMPLATE.format(season=season), low_memory=False)
            frames.append(_normalize_player_team_column(x))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    d=pd.concat(frames,ignore_index=True)
    d=_normalize_player_team_column(d)
    for c in ["season","week"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    if "recent_team" in d.columns:
        d["recent_team"]=d["recent_team"].map(lambda x: TEAM_ALIASES.get(str(x),str(x)) if pd.notna(x) else x)
    return d

def _weighted(values, weights):
    x=pd.to_numeric(values,errors="coerce")
    mask=x.notna()
    if not mask.any(): return 0.0
    w=np.asarray(weights)[-len(x):][mask.to_numpy()]
    return float(np.average(x[mask],weights=w))

def player_usage_before_week(stats, season, week, team=None, lookback=6):
    """Only completed weeks < selected week are used."""
    if stats is None or stats.empty:return pd.DataFrame()
    d=_normalize_player_team_column(stats)
    d=d[(d.season==season)&(d.week<week)].copy()
    if team: d=d[d.recent_team==team]
    if d.empty: return pd.DataFrame()
    idcol="player_id" if "player_id" in d else "player_display_name"
    rows=[]
    for pid,g in d.groupby(idcol):
        g=g.sort_values("week").tail(lookback)
        weights=np.arange(1,len(g)+1,dtype=float)
        name=g["player_display_name"].iloc[-1] if "player_display_name" in g else str(pid)
        pos=g["position"].iloc[-1] if "position" in g else ""
        tm=g["recent_team"].iloc[-1] if "recent_team" in g else ""
        def col(c): return _weighted(g[c],weights) if c in g else 0.0
        rows.append({
            "player_id":pid,"player":name,"position":pos,"team":tm,
            "games_used":len(g),
            "attempts":col("attempts"),"completions":col("completions"),
            "passing_yards":col("passing_yards"),"passing_tds":col("passing_tds"),
            "carries":col("carries"),"rushing_yards":col("rushing_yards"),
            "rushing_tds":col("rushing_tds"),"targets":col("targets"),
            "receptions":col("receptions"),"receiving_yards":col("receiving_yards"),
            "receiving_tds":col("receiving_tds")
        })
    return pd.DataFrame(rows)

def project_team_players(usage, team_points=22.0, game_total=44.0, team_pass_multiplier=1.0, team_rush_multiplier=1.0):
    """V0.5 opportunity projection. Team environment modifies recent player usage."""
    if usage.empty: return usage
    x=usage.copy()
    scoring=max(.65,min(1.45,team_points/22.0))
    pace=max(.75,min(1.30,game_total/44.0))
    x["proj_attempts"]=x.attempts*pace*team_pass_multiplier
    x["proj_carries"]=x.carries*pace*team_rush_multiplier
    x["proj_targets"]=x.targets*pace*team_pass_multiplier
    ypa=np.where(x.attempts>0,x.passing_yards/x.attempts,0)
    ypc=np.where(x.carries>0,x.rushing_yards/x.carries,0)
    ypt=np.where(x.targets>0,x.receiving_yards/x.targets,0)
    catch=np.where(x.targets>0,x.receptions/x.targets,0)
    x["proj_passing_yards"]=x.proj_attempts*ypa
    x["proj_rushing_yards"]=x.proj_carries*ypc
    x["proj_receiving_yards"]=x.proj_targets*ypt
    x["proj_receptions"]=x.proj_targets*catch
    x["proj_passing_tds"]=x.passing_tds*scoring
    x["proj_rushing_tds"]=x.rushing_tds*scoring
    x["proj_receiving_tds"]=x.receiving_tds*scoring
    return x

def fantasy_points(x, scoring="PPR"):
    reception_value={"PPR":1.0,"Half-PPR":0.5,"Standard":0.0}[scoring]
    return (x.proj_passing_yards/25+x.proj_passing_tds*4+x.proj_rushing_yards/10+x.proj_rushing_tds*6+x.proj_receiving_yards/10+x.proj_receiving_tds*6+x.proj_receptions*reception_value)

def simulate_player(row, scoring="PPR", n=30000, seed=42):
    rng=np.random.default_rng(seed)
    att=np.maximum(0,rng.normal(row.proj_attempts,max(2,0.14*row.proj_attempts),n)); car=np.maximum(0,rng.normal(row.proj_carries,max(1.5,0.22*row.proj_carries),n)); tar=np.maximum(0,rng.normal(row.proj_targets,max(1.2,0.25*row.proj_targets),n))
    py=np.maximum(0,rng.normal(row.proj_passing_yards,max(18,0.20*row.proj_passing_yards),n)); ry=np.maximum(0,rng.normal(row.proj_rushing_yards,max(8,0.30*row.proj_rushing_yards),n)); recy=np.maximum(0,rng.normal(row.proj_receiving_yards,max(10,0.30*row.proj_receiving_yards),n))
    rec=np.minimum(tar,np.maximum(0,rng.normal(row.proj_receptions,max(.8,0.20*row.proj_receptions),n))); ptd=rng.poisson(max(0,row.proj_passing_tds),n); rtd=rng.poisson(max(0,row.proj_rushing_tds),n); rectd=rng.poisson(max(0,row.proj_receiving_tds),n); rv={"PPR":1.0,"Half-PPR":.5,"Standard":0}[scoring]
    fp=py/25+ptd*4+ry/10+rtd*6+recy/10+rectd*6+rec*rv
    return {"floor":float(np.percentile(fp,20)),"median":float(np.percentile(fp,50)),"ceiling":float(np.percentile(fp,80)),"mean":float(np.mean(fp)),"fantasy_samples":fp,"passing_yards":py,"rushing_yards":ry,"receiving_yards":recy,"receptions":rec,"targets":tar,"carries":car,"attempts":att}

STAT_MAP={"Passing Yards":("proj_passing_yards","passing_yards"),"Rushing Yards":("proj_rushing_yards","rushing_yards"),"Receiving Yards":("proj_receiving_yards","receiving_yards"),"Receptions":("proj_receptions","receptions"),"Targets":("proj_targets","targets"),"Carries":("proj_carries","carries")}

def backtest_players(stats, season, scoring="PPR", start_week=4):
    rows=[]; season_data=_normalize_player_team_column(stats[stats.season==season].copy())
    for week in sorted(season_data.week.dropna().astype(int).unique()):
        if week<start_week: continue
        actual=season_data[season_data.week==week]
        for team in actual.recent_team.dropna().unique():
            usage=player_usage_before_week(stats,season,week,team)
            if usage.empty: continue
            proj=project_team_players(usage)
            for _,p in proj.iterrows():
                a=actual[actual.player_id==p.player_id] if "player_id" in actual and "player_id" in p else pd.DataFrame()
                if a.empty: continue
                a=a.iloc[0]; record={"season":season,"week":week,"player":p.player,"position":p.position,"team":team}
                for label,(pc,ac) in STAT_MAP.items():
                    record[f"proj_{label}"]=float(p.get(pc,0)); record[f"actual_{label}"]=float(a.get(ac,0) if pd.notna(a.get(ac,0)) else 0); record[f"ae_{label}"]=abs(record[f"proj_{label}"]-record[f"actual_{label}"])
                record["proj_fantasy"]=fantasy_points(p,scoring); recval={"PPR":1.0,"Half-PPR":.5,"Standard":0}[scoring]
                record["actual_fantasy"]=(float(a.get("passing_yards",0) or 0)/25+float(a.get("passing_tds",0) or 0)*4+float(a.get("rushing_yards",0) or 0)/10+float(a.get("rushing_tds",0) or 0)*6+float(a.get("receiving_yards",0) or 0)/10+float(a.get("receiving_tds",0) or 0)*6+float(a.get("receptions",0) or 0)*recval)
                record["ae_fantasy"]=abs(record["proj_fantasy"]-record["actual_fantasy"]); rows.append(record)
    return pd.DataFrame(rows)

def summarize(bt):
    if bt.empty:return pd.DataFrame()
    rows=[]
    for pos,g in list(bt.groupby("position"))+[("ALL",bt)]: rows.append({"Position":pos,"Player-games":len(g),"Fantasy MAE":g.ae_fantasy.mean(),"Pass Yds MAE":g["ae_Passing Yards"].mean(),"Rush Yds MAE":g["ae_Rushing Yards"].mean(),"Rec Yds MAE":g["ae_Receiving Yards"].mean(),"Receptions MAE":g["ae_Receptions"].mean(),"Targets MAE":g["ae_Targets"].mean(),"Carries MAE":g["ae_Carries"].mean()})
    return pd.DataFrame(rows)

def prop_probability(samples,line,over=True):
    arr=np.asarray(samples); return float(np.mean(arr>line)) if over else float(np.mean(arr<line))
def edge_label(prob,pass_floor=.55,strong=.60):
    if prob<strong and prob<pass_floor:return "PASS"
    if prob>=.67:return "VERY HIGH"
    if prob>=strong:return "HIGH"
    if prob>=pass_floor:return "LEAN"
    return "PASS"

# NOTE: Remaining validated Game Engine / Football Intelligence / Streamlit UI code
# is intentionally retained in the repository history and will be restored in the
# next modularization commit. This compatibility commit focuses on the player feed.
