from __future__ import annotations
# NFL Edge v0.7 deployment edition — standalone Streamlit app
# Data source attribution: nflverse (CC-BY 4.0 where applicable).

import math
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

def load_player_stats(start_season=2020, end_season=2026):
    frames=[]
    for season in range(start_season,end_season+1):
        try:
            frames.append(pd.read_csv(PLAYER_URL_TEMPLATE.format(season=season), low_memory=False))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    d=pd.concat(frames,ignore_index=True)
    for c in ["season","week"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    return d

def _weighted(values, weights):
    x=pd.to_numeric(values,errors="coerce")
    mask=x.notna()
    if not mask.any(): return 0.0
    w=np.asarray(weights)[-len(x):][mask.to_numpy()]
    return float(np.average(x[mask],weights=w))

def player_usage_before_week(stats, season, week, team=None, lookback=6):
    """Only completed weeks < selected week are used."""
    d=stats[(stats.season==season)&(stats.week<week)].copy()
    if team and "recent_team" in d: d=d[d.recent_team==team]
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
    # Efficiency is shrunk slightly toward recent rates rather than independently invented.
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
    return (
        x.proj_passing_yards/25 + x.proj_passing_tds*4
        + x.proj_rushing_yards/10 + x.proj_rushing_tds*6
        + x.proj_receiving_yards/10 + x.proj_receiving_tds*6
        + x.proj_receptions*reception_value
    )

def simulate_player(row, scoring="PPR", n=30000, seed=42):
    rng=np.random.default_rng(seed)
    # V0.5 uncertainty bands are intentionally broad; later versions will learn residual distributions by position/stat.
    att=np.maximum(0,rng.normal(row.proj_attempts,max(2,0.14*row.proj_attempts),n))
    car=np.maximum(0,rng.normal(row.proj_carries,max(1.5,0.22*row.proj_carries),n))
    tar=np.maximum(0,rng.normal(row.proj_targets,max(1.2,0.25*row.proj_targets),n))
    py=np.maximum(0,rng.normal(row.proj_passing_yards,max(18,0.20*row.proj_passing_yards),n))
    ry=np.maximum(0,rng.normal(row.proj_rushing_yards,max(8,0.30*row.proj_rushing_yards),n))
    recy=np.maximum(0,rng.normal(row.proj_receiving_yards,max(10,0.30*row.proj_receiving_yards),n))
    rec=np.minimum(tar,np.maximum(0,rng.normal(row.proj_receptions,max(.8,0.20*row.proj_receptions),n)))
    ptd=rng.poisson(max(0,row.proj_passing_tds),n)
    rtd=rng.poisson(max(0,row.proj_rushing_tds),n)
    rectd=rng.poisson(max(0,row.proj_receiving_tds),n)
    rv={"PPR":1.0,"Half-PPR":.5,"Standard":0}[scoring]
    fp=py/25+ptd*4+ry/10+rtd*6+recy/10+rectd*6+rec*rv
    return {
        "floor":float(np.percentile(fp,20)),"median":float(np.percentile(fp,50)),
        "ceiling":float(np.percentile(fp,80)),"mean":float(np.mean(fp)),
        "fantasy_samples":fp,"passing_yards":py,"rushing_yards":ry,
        "receiving_yards":recy,"receptions":rec,"targets":tar,"carries":car,"attempts":att
    }


import numpy as np
import pandas as pd

STAT_MAP={
    "Passing Yards":("proj_passing_yards","passing_yards"),
    "Rushing Yards":("proj_rushing_yards","rushing_yards"),
    "Receiving Yards":("proj_receiving_yards","receiving_yards"),
    "Receptions":("proj_receptions","receptions"),
    "Targets":("proj_targets","targets"),
    "Carries":("proj_carries","carries"),
}

def backtest_players(stats, season, scoring="PPR", start_week=4):
    rows=[]
    season_data=stats[stats.season==season].copy()
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
                a=a.iloc[0]
                record={"season":season,"week":week,"player":p.player,"position":p.position,"team":team}
                for label,(pc,ac) in STAT_MAP.items():
                    record[f"proj_{label}"]=float(p.get(pc,0))
                    record[f"actual_{label}"]=float(a.get(ac,0) if pd.notna(a.get(ac,0)) else 0)
                    record[f"ae_{label}"]=abs(record[f"proj_{label}"]-record[f"actual_{label}"])
                record["proj_fantasy"]=fantasy_points(p,scoring)
                # actual fantasy from standard nflverse weekly columns
                recval={"PPR":1.0,"Half-PPR":.5,"Standard":0}[scoring]
                record["actual_fantasy"]=(
                    float(a.get("passing_yards",0) or 0)/25+float(a.get("passing_tds",0) or 0)*4+
                    float(a.get("rushing_yards",0) or 0)/10+float(a.get("rushing_tds",0) or 0)*6+
                    float(a.get("receiving_yards",0) or 0)/10+float(a.get("receiving_tds",0) or 0)*6+
                    float(a.get("receptions",0) or 0)*recval
                )
                record["ae_fantasy"]=abs(record["proj_fantasy"]-record["actual_fantasy"])
                rows.append(record)
    return pd.DataFrame(rows)

def summarize(bt):
    if bt.empty:return pd.DataFrame()
    rows=[]
    for pos,g in list(bt.groupby("position"))+[("ALL",bt)]:
        rows.append({
            "Position":pos,"Player-games":len(g),
            "Fantasy MAE":g.ae_fantasy.mean(),
            "Pass Yds MAE":g["ae_Passing Yards"].mean(),
            "Rush Yds MAE":g["ae_Rushing Yards"].mean(),
            "Rec Yds MAE":g["ae_Receiving Yards"].mean(),
            "Receptions MAE":g["ae_Receptions"].mean(),
            "Targets MAE":g["ae_Targets"].mean(),
            "Carries MAE":g["ae_Carries"].mean(),
        })
    return pd.DataFrame(rows)

def prop_probability(samples, line, over=True):
    arr=np.asarray(samples)
    return float(np.mean(arr>line)) if over else float(np.mean(arr<line))

def edge_label(prob, pass_floor=.55, strong=.60):
    if prob<strong and prob<pass_floor:return "PASS"
    if prob>=.67:return "VERY HIGH"
    if prob>=strong:return "HIGH"
    if prob>=pass_floor:return "LEAN"
    return "PASS"


# ---------- v1.0 efficiency challenger ----------
PBP_URL="https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
EFF_FEATURES=[
    "off_epa","off_success","off_pass_epa","off_rush_epa","off_explosive","off_pass_rate",
    "def_epa_allowed","def_success_allowed","def_pass_epa_allowed","def_rush_epa_allowed","def_explosive_allowed"
]

def load_efficiency_games(season):
    """Build team-game efficiency rows from nflverse PBP. No market variables."""
    use=["game_id","season","week","posteam","defteam","epa","success","pass","rush","yards_gained","play_type"]
    p=pd.read_parquet(PBP_URL.format(season=int(season)),columns=use)
    p=p[(p.posteam.notna())&(p.defteam.notna())&p.play_type.isin(["pass","run"])].copy()
    p["explosive"]=((p["pass"].fillna(0).eq(1)&p.yards_gained.ge(15)) |
                    (p["rush"].fillna(0).eq(1)&p.yards_gained.ge(10))).astype(float)
    p["pass_epa_val"]=p.epa.where(p["pass"].fillna(0).eq(1))
    p["rush_epa_val"]=p.epa.where(p["rush"].fillna(0).eq(1))
    off=p.groupby(["game_id","season","week","posteam"],as_index=False).agg(
        off_epa=("epa","mean"),off_success=("success","mean"),
        off_pass_epa=("pass_epa_val","mean"),off_rush_epa=("rush_epa_val","mean"),
        off_explosive=("explosive","mean"),off_pass_rate=("pass","mean"))
    de=p.groupby(["game_id","season","week","defteam"],as_index=False).agg(
        def_epa_allowed=("epa","mean"),def_success_allowed=("success","mean"),
        def_pass_epa_allowed=("pass_epa_val","mean"),def_rush_epa_allowed=("rush_epa_val","mean"),
        def_explosive_allowed=("explosive","mean"))
    return off,de

def efficiency_pregame_features(games,seasons=None,window=6):
    """Leakage-safe rolling efficiency: every row uses only games before that game."""
    base=pregame_features(games).copy()
    if seasons is None: seasons=sorted(base.season.dropna().astype(int).unique())
    off_parts=[]; def_parts=[]
    for season in seasons:
        try:
            o,d=load_efficiency_games(season); off_parts.append(o); def_parts.append(d)
        except Exception:
            continue
    if not off_parts:return base
    off=pd.concat(off_parts,ignore_index=True); de=pd.concat(def_parts,ignore_index=True)
    hist={}
    rows=[]
    def avg(team,key,default=0.0):
        vals=hist.get(team,{}).get(key,[])
        return float(np.mean(vals[-window:])) if vals else default
    for idx,g in base.sort_values(["gameday","gametime"],na_position="last").iterrows():
        h,a=g.home_team,g.away_team
        r=g.to_dict()
        for prefix,team in [("home",h),("away",a)]:
            for k in EFF_FEATURES:r[f"{prefix}_{k}"]=avg(team,k,0.0)
        # matchup interactions: offense relative to opponent defense
        r["home_pass_matchup"]=r["home_off_pass_epa"]-r["away_def_pass_epa_allowed"]
        r["away_pass_matchup"]=r["away_off_pass_epa"]-r["home_def_pass_epa_allowed"]
        r["home_rush_matchup"]=r["home_off_rush_epa"]-r["away_def_rush_epa_allowed"]
        r["away_rush_matchup"]=r["away_off_rush_epa"]-r["home_def_rush_epa_allowed"]
        rows.append(r)

        gid=g.get("game_id")
        if pd.notna(gid):
            oo=off[off.game_id==gid]; dd=de[de.game_id==gid]
            for team in [h,a]:
                hist.setdefault(team,{k:[] for k in EFF_FEATURES})
                xo=oo[oo.posteam==team]
                xd=dd[dd.defteam==team]
                if not xo.empty:
                    for k in EFF_FEATURES[:6]:
                        v=xo.iloc[0][k]
                        if pd.notna(v):hist[team][k].append(float(v))
                if not xd.empty:
                    for k in EFF_FEATURES[6:]:
                        v=xd.iloc[0][k]
                        if pd.notna(v):hist[team][k].append(float(v))
    return pd.DataFrame(rows)

CHALLENGER_FEATURES=FEATURES + [
    "home_off_epa","away_off_epa","home_def_epa_allowed","away_def_epa_allowed",
    "home_off_success","away_off_success","home_def_success_allowed","away_def_success_allowed",
    "home_off_explosive","away_off_explosive","home_def_explosive_allowed","away_def_explosive_allowed",
    "home_off_pass_rate","away_off_pass_rate",
    "home_pass_matchup","away_pass_matchup","home_rush_matchup","away_rush_matchup"
]

def compare_v09_vs_v10(games,first_test_season=2022):
    """Same weekly folds for baseline and efficiency challenger."""
    f=efficiency_pregame_features(games)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    test=done[(done.season>=first_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    out=[]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min()
        tr=done[(done.gameday<cutoff)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(tr)<300:continue
        for name,features in [("v0.9 baseline",FEATURES),("v1.0 efficiency",CHALLENGER_FEATURES)]:
            mm,tm=pipe(25),pipe(30)
            wm=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                         ("model",LogisticRegression(C=.45,max_iter=2000))])
            mm.fit(tr[features],tr.target_margin); tm.fit(tr[features],tr.target_total)
            wm.fit(tr[features],(tr.target_margin>0).astype(int))
            pm=mm.predict(te[features]); pt=tm.predict(te[features]); wp=wm.predict_proba(te[features])[:,1]
            y=(te.target_margin>0).astype(float).to_numpy()
            for i,(_,r) in enumerate(te.iterrows()):
                out.append({"model":name,"season":int(season),"week":int(week),
                            "winner_correct":bool((wp[i]>=.5)==bool(y[i])),
                            "brier":float((wp[i]-y[i])**2),
                            "margin_abs_error":float(abs(pm[i]-r.target_margin)),
                            "total_abs_error":float(abs(pt[i]-r.target_total))})
    return pd.DataFrame(out)

def model_scorecard(cmp):
    if cmp.empty:return pd.DataFrame()
    return cmp.groupby("model").agg(
        Games=("winner_correct","size"),
        Winner_Accuracy=("winner_correct","mean"),
        Brier=("brier","mean"),
        Margin_MAE=("margin_abs_error","mean"),
        Total_MAE=("total_abs_error","mean")
    ).reset_index()

def promotion_decision(score):
    if len(score)<2:return "INSUFFICIENT DATA"
    b=score[score.model=="v0.9 baseline"].iloc[0]
    c=score[score.model=="v1.0 efficiency"].iloc[0]
    wins=(c.Winner_Accuracy>b.Winner_Accuracy)+(c.Brier<b.Brier)+(c.Margin_MAE<b.Margin_MAE)+(c.Total_MAE<b.Total_MAE)
    # Require probability quality + at least one additional metric; no cosmetic promotion.
    return "PROMOTE" if (c.Brier<b.Brier and wins>=3) else "DO NOT PROMOTE"

# ---------- v1.1 model tournament + feature ablation ----------
ABLATIONS={
    "Baseline": FEATURES,
    "EPA core": FEATURES+[
        "home_off_epa","away_off_epa","home_def_epa_allowed","away_def_epa_allowed"
    ],
    "Pass efficiency": FEATURES+[
        "home_off_pass_epa","away_off_pass_epa","home_def_pass_epa_allowed","away_def_pass_epa_allowed",
        "home_pass_matchup","away_pass_matchup"
    ],
    "Rush efficiency": FEATURES+[
        "home_off_rush_epa","away_off_rush_epa","home_def_rush_epa_allowed","away_def_rush_epa_allowed",
        "home_rush_matchup","away_rush_matchup"
    ],
    "Success rate": FEATURES+[
        "home_off_success","away_off_success","home_def_success_allowed","away_def_success_allowed"
    ],
    "Explosives": FEATURES+[
        "home_off_explosive","away_off_explosive","home_def_explosive_allowed","away_def_explosive_allowed"
    ],
    "All efficiency": CHALLENGER_FEATURES
}

def nonlinear_regressor():
    return Pipeline([("impute",SimpleImputer(strategy="median")),
                     ("model",HistGradientBoostingRegressor(max_iter=120,learning_rate=.045,
                         max_leaf_nodes=12,l2_regularization=2.0,random_state=17))])

def nonlinear_classifier():
    return Pipeline([("impute",SimpleImputer(strategy="median")),
                     ("model",HistGradientBoostingClassifier(max_iter=120,learning_rate=.045,
                         max_leaf_nodes=12,l2_regularization=2.0,random_state=17))])

def v11_model_tournament(games,first_test_season=2022):
    """Same walk-forward folds for every architecture and feature family."""
    f=efficiency_pregame_features(games)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    test=done[(done.season>=first_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    out=[]
    candidates=[
        ("v0.9 Ridge",FEATURES,"linear"),
        ("Efficiency Ridge",CHALLENGER_FEATURES,"linear"),
        ("Efficiency Boost",CHALLENGER_FEATURES,"boost"),
    ]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min()
        tr=done[(done.gameday<cutoff)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(tr)<300:continue
        ywin=(tr.target_margin>0).astype(int)
        actual=(te.target_margin>0).astype(float).to_numpy()
        for name,features,kind in candidates:
            if kind=="linear":
                mm,tm=pipe(25),pipe(30)
                wm=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                             ("model",LogisticRegression(C=.45,max_iter=2000))])
            else:
                mm,tm=nonlinear_regressor(),nonlinear_regressor()
                wm=nonlinear_classifier()
            mm.fit(tr[features],tr.target_margin); tm.fit(tr[features],tr.target_total); wm.fit(tr[features],ywin)
            pm=mm.predict(te[features]); pt=tm.predict(te[features]); wp=wm.predict_proba(te[features])[:,1]
            for i,(_,r) in enumerate(te.iterrows()):
                out.append({"candidate":name,"season":int(season),"week":int(week),
                            "winner_correct":bool((wp[i]>=.5)==bool(actual[i])),
                            "brier":float((wp[i]-actual[i])**2),
                            "margin_abs_error":float(abs(pm[i]-r.target_margin)),
                            "total_abs_error":float(abs(pt[i]-r.target_total))})
    return pd.DataFrame(out)

def v11_ablation_test(games,first_test_season=2022):
    """Linear ablation isolates which football feature families add signal."""
    f=efficiency_pregame_features(games)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    test=done[(done.season>=first_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min()
        tr=done[(done.gameday<cutoff)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(tr)<300:continue
        actual=(te.target_margin>0).astype(float).to_numpy()
        for group,features in ABLATIONS.items():
            mm,tm=pipe(25),pipe(30)
            wm=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                         ("model",LogisticRegression(C=.45,max_iter=2000))])
            mm.fit(tr[features],tr.target_margin); tm.fit(tr[features],tr.target_total)
            wm.fit(tr[features],(tr.target_margin>0).astype(int))
            pm=mm.predict(te[features]); pt=tm.predict(te[features]); wp=wm.predict_proba(te[features])[:,1]
            rows.append({"season":int(season),"week":int(week),"feature_group":group,"games":len(te),
                         "winner_correct":float(np.mean((wp>=.5)==actual.astype(bool))),
                         "brier":float(np.mean((wp-actual)**2)),
                         "margin_mae":float(np.mean(np.abs(pm-te.target_margin.to_numpy()))),
                         "total_mae":float(np.mean(np.abs(pt-te.target_total.to_numpy())))})
    return pd.DataFrame(rows)

def weighted_ablation_summary(ab):
    if ab.empty:return pd.DataFrame()
    rows=[]
    for name,x in ab.groupby("feature_group"):
        n=x.games.sum()
        rows.append({"Feature group":name,"Games":int(n),
                     "Winner Accuracy":float(np.average(x.winner_correct,weights=x.games)),
                     "Brier":float(np.average(x.brier,weights=x.games)),
                     "Margin MAE":float(np.average(x.margin_mae,weights=x.games)),
                     "Total MAE":float(np.average(x.total_mae,weights=x.games))})
    return pd.DataFrame(rows)

def tournament_summary(t):
    if t.empty:return pd.DataFrame()
    return t.groupby("candidate").agg(Games=("winner_correct","size"),
        Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean"),
        Margin_MAE=("margin_abs_error","mean"),Total_MAE=("total_abs_error","mean")).reset_index()

def component_winners(score):
    if score.empty:return {}
    return {
        "Winner probability":score.loc[score.Brier.idxmin(),"candidate"],
        "Straight-up accuracy":score.loc[score.Winner_Accuracy.idxmax(),"candidate"],
        "Margin":score.loc[score.Margin_MAE.idxmin(),"candidate"],
        "Total":score.loc[score.Total_MAE.idxmin(),"candidate"]
    }

# ---------- v1.2 Hybrid Championship ----------
EPA_CORE_FEATURES=FEATURES+[
    "home_off_epa","away_off_epa","home_def_epa_allowed","away_def_epa_allowed"
]
PASS_FEATURES=FEATURES+[
    "home_off_pass_epa","away_off_pass_epa","home_def_pass_epa_allowed","away_def_pass_epa_allowed",
    "home_pass_matchup","away_pass_matchup"
]

def _linear_classifier():
    return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                     ("model",LogisticRegression(C=.45,max_iter=2000))])

def _fit_champions(tr, te):
    """Finalists earned in v1.1. Returns predictions for each specialized candidate."""
    specs={
        "Baseline Ridge":(FEATURES,"linear"),
        "EPA Core Ridge":(EPA_CORE_FEATURES,"linear"),
        "Pass Ridge":(PASS_FEATURES,"linear"),
        "Efficiency Boost":(CHALLENGER_FEATURES,"boost")
    }
    out={}
    for name,(features,kind) in specs.items():
        if kind=="linear":
            mm,tm,wm=pipe(25),pipe(30),_linear_classifier()
        else:
            mm,tm,wm=nonlinear_regressor(),nonlinear_regressor(),nonlinear_classifier()
        mm.fit(tr[features],tr.target_margin)
        tm.fit(tr[features],tr.target_total)
        wm.fit(tr[features],(tr.target_margin>0).astype(int))
        out[name]={
            "margin":mm.predict(te[features]),
            "total":tm.predict(te[features]),
            "win_prob":wm.predict_proba(te[features])[:,1]
        }
    return out

def v12_championship(games,first_test_season=2022):
    """Head-to-head finalists on identical weekly walk-forward folds."""
    f=efficiency_pregame_features(games)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    test=done[(done.season>=first_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min()
        tr=done[(done.gameday<cutoff)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(tr)<300:continue
        preds=_fit_champions(tr,te)
        actual=(te.target_margin>0).astype(float).to_numpy()
        for name,pred in preds.items():
            for i,(_,r) in enumerate(te.iterrows()):
                rows.append({
                    "candidate":name,"season":int(season),"week":int(week),
                    "actual_win":actual[i],"win_prob":float(pred["win_prob"][i]),
                    "winner_correct":bool((pred["win_prob"][i]>=.5)==bool(actual[i])),
                    "margin_error":float(abs(pred["margin"][i]-r.target_margin)),
                    "total_error":float(abs(pred["total"][i]-r.target_total))
                })
    return pd.DataFrame(rows)

def v12_summary(ch):
    if ch.empty:return pd.DataFrame()
    x=ch.copy()
    x["brier"]=(x.win_prob-x.actual_win)**2
    return x.groupby("candidate").agg(
        Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),
        Brier=("brier","mean"),Margin_MAE=("margin_error","mean"),
        Total_MAE=("total_error","mean")).reset_index()

def v12_yearly(ch):
    if ch.empty:return pd.DataFrame()
    x=ch.copy(); x["brier"]=(x.win_prob-x.actual_win)**2
    return x.groupby(["season","candidate"]).agg(
        Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),
        Brier=("brier","mean"),Margin_MAE=("margin_error","mean"),
        Total_MAE=("total_error","mean")).reset_index()

def paired_bootstrap(ch, challenger, baseline="Baseline Ridge", metric="brier", n_boot=4000, seed=27):
    """Paired game bootstrap: probability challenger improves over baseline."""
    b=ch[ch.candidate==baseline].reset_index(drop=True).copy()
    c=ch[ch.candidate==challenger].reset_index(drop=True).copy()
    if len(b)!=len(c) or not len(b):return np.nan, np.nan, np.nan
    if metric=="brier":
        bv=(b.win_prob-b.actual_win).to_numpy()**2
        cv=(c.win_prob-c.actual_win).to_numpy()**2
    elif metric=="margin":
        bv=b.margin_error.to_numpy(); cv=c.margin_error.to_numpy()
    elif metric=="total":
        bv=b.total_error.to_numpy(); cv=c.total_error.to_numpy()
    elif metric=="accuracy":
        # Higher is better, flip sign so negative delta still means improvement.
        bv=-b.winner_correct.astype(float).to_numpy()
        cv=-c.winner_correct.astype(float).to_numpy()
    else: raise ValueError(metric)
    d=cv-bv
    rng=np.random.default_rng(seed); n=len(d)
    means=np.empty(n_boot)
    for j in range(n_boot):
        means[j]=np.mean(d[rng.integers(0,n,n)])
    return float(np.mean(d)), float(np.quantile(means,.025)), float(np.quantile(means,.975))

def v12_bootstrap_table(ch):
    rows=[]
    for candidate in ["EPA Core Ridge","Pass Ridge","Efficiency Boost"]:
        for label,key in [("Winner accuracy","accuracy"),("Brier","brier"),("Margin MAE","margin"),("Total MAE","total")]:
            delta,lo,hi=paired_bootstrap(ch,candidate,metric=key)
            rows.append({"Candidate":candidate,"Metric":label,"Delta vs baseline":delta,
                         "95% CI low":lo,"95% CI high":hi,
                         "Clear improvement":bool(pd.notna(hi) and hi<0)})
    return pd.DataFrame(rows)

def v12_champions(summary):
    if summary.empty:return {}
    return {
        "Probability":summary.loc[summary.Brier.idxmin(),"candidate"],
        "Straight-up":summary.loc[summary.Winner_Accuracy.idxmax(),"candidate"],
        "Margin":summary.loc[summary.Margin_MAE.idxmin(),"candidate"],
        "Total":summary.loc[summary.Total_MAE.idxmin(),"candidate"]
    }

# ---------- v1.3 Game Engine 1.0 ----------
# Production policy from v1.2:
# - win probability: EPA Core Ridge, with baseline shown as shadow benchmark
# - margin: protected Baseline Ridge; Efficiency Boost remains shadow challenger
# - total: Pass Ridge (cleared paired-bootstrap improvement)

def fit_game_engine_v10(feature_frame, target_row):
    """Fit specialized production components using only completed games before target kickoff."""
    cutoff=target_row["gameday"]
    tr=feature_frame[
        (feature_frame.gameday<cutoff)&
        feature_frame.target_margin.notna()&
        feature_frame.target_total.notna()&
        (feature_frame.home_games_prior>=2)&
        (feature_frame.away_games_prior>=2)
    ].copy()
    if len(tr)<300:
        return None

    # Production components
    prob=_linear_classifier()
    margin=pipe(25)
    total=pipe(30)

    prob.fit(tr[EPA_CORE_FEATURES],(tr.target_margin>0).astype(int))
    margin.fit(tr[FEATURES],tr.target_margin)
    total.fit(tr[PASS_FEATURES],tr.target_total)

    # Shadow models for monitoring, not silent production replacement.
    shadow_prob=_linear_classifier()
    shadow_margin=nonlinear_regressor()
    shadow_prob.fit(tr[FEATURES],(tr.target_margin>0).astype(int))
    shadow_margin.fit(tr[CHALLENGER_FEATURES],tr.target_margin)

    one=target_row.to_frame().T
    return {
        "home_win_prob":float(prob.predict_proba(one[EPA_CORE_FEATURES])[:,1][0]),
        "margin":float(margin.predict(one[FEATURES])[0]),
        "total":float(total.predict(one[PASS_FEATURES])[0]),
        "shadow_home_win_prob":float(shadow_prob.predict_proba(one[FEATURES])[:,1][0]),
        "shadow_margin":float(shadow_margin.predict(one[CHALLENGER_FEATURES])[0]),
        "training_games":int(len(tr))
    }

def game_engine_board(games, season, week):
    """Leakage-safe production board for a selected week."""
    f=efficiency_pregame_features(games)
    slate=f[(f.season==season)&(f.week==week)].sort_values(["gameday","gametime"],na_position="last")
    rows=[]
    for _,r in slate.iterrows():
        pred=fit_game_engine_v10(f,r)
        if pred is None: continue
        margin=pred["margin"]; total=pred["total"]
        hp=(total+margin)/2; ap=(total-margin)/2
        rows.append({
            "game_id":r.get("game_id"),"gameday":r.get("gameday"),
            "away_team":r.away_team,"home_team":r.home_team,
            "away_score_proj":ap,"home_score_proj":hp,
            **pred,
            "market_spread":r.get("spread_line",np.nan),
            "market_total":r.get("total_line",np.nan)
        })
    return pd.DataFrame(rows)

def v13_engine_backtest(games, first_test_season=2022):
    """Validate the exact specialized production architecture against v0.9."""
    f=efficiency_pregame_features(games)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    test=done[(done.season>=first_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min()
        tr=done[(done.gameday<cutoff)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(tr)<300: continue

        # Baseline
        bp=_linear_classifier(); bm=pipe(25); bt=pipe(30)
        bp.fit(tr[FEATURES],(tr.target_margin>0).astype(int))
        bm.fit(tr[FEATURES],tr.target_margin); bt.fit(tr[FEATURES],tr.target_total)

        # Specialized engine
        ep=_linear_classifier(); em=pipe(25); et=pipe(30)
        ep.fit(tr[EPA_CORE_FEATURES],(tr.target_margin>0).astype(int))
        em.fit(tr[FEATURES],tr.target_margin)
        et.fit(tr[PASS_FEATURES],tr.target_total)

        actual=(te.target_margin>0).astype(float).to_numpy()
        models=[
            ("v0.9 baseline",
             bp.predict_proba(te[FEATURES])[:,1],bm.predict(te[FEATURES]),bt.predict(te[FEATURES])),
            ("Game Engine 1.0",
             ep.predict_proba(te[EPA_CORE_FEATURES])[:,1],em.predict(te[FEATURES]),et.predict(te[PASS_FEATURES]))
        ]
        for name,wp,pm,pt in models:
            for i,(_,r) in enumerate(te.iterrows()):
                rows.append({
                    "model":name,"season":int(season),"week":int(week),
                    "actual_win":actual[i],"win_prob":float(wp[i]),
                    "winner_correct":bool((wp[i]>=.5)==bool(actual[i])),
                    "margin_error":float(abs(pm[i]-r.target_margin)),
                    "total_error":float(abs(pt[i]-r.target_total))
                })
    return pd.DataFrame(rows)

def v13_summary(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["brier"]=(y.win_prob-y.actual_win)**2
    return y.groupby("model").agg(
        Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),
        Brier=("brier","mean"),Margin_MAE=("margin_error","mean"),
        Total_MAE=("total_error","mean")).reset_index()

def v13_yearly(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["brier"]=(y.win_prob-y.actual_win)**2
    return y.groupby(["season","model"]).agg(
        Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),
        Brier=("brier","mean"),Margin_MAE=("margin_error","mean"),
        Total_MAE=("total_error","mean")).reset_index()

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NFL Edge",page_icon="🏈",layout="wide",initial_sidebar_state="collapsed")
GAMES_URL="https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.2rem;padding-bottom:3rem}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);padding:12px;border-radius:12px}
h1{margin-bottom:0}
.small{opacity:.72;font-size:.9rem}
.edge-card{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:14px 16px;margin:10px 0 14px}
.edge-game{font-size:1.15rem;font-weight:700;margin-bottom:4px}
.edge-score{font-size:1.05rem;font-weight:600}
.edge-muted{opacity:.72;font-size:.88rem}
@media(max-width:700px){.block-container{padding-left:.75rem;padding-right:.75rem}[data-testid="stMetricValue"]{font-size:1.35rem}}
</style>
""",unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def games(): return load_games(GAMES_URL)
@st.cache_data(ttl=3600)
def players(): return load_player_stats()
@st.cache_resource
def model(_g): return NFLEdgeModel().fit(_g)

g=games(); m=model(g); f=m.feature_df
st.title("🏈 NFL EDGE")
st.markdown('<div class="small">V1.3 • Game Engine 1.0 • Specialized production architecture</div>',unsafe_allow_html=True)

page=st.radio("Navigation",["🏈 GAMES","📊 PROPS","🏆 FANTASY","🔬 SIMULATOR","📈 MODEL RECORD"],horizontal=True,label_visibility="collapsed")

def relevant_season_week():
    seasons=sorted(f.season.dropna().astype(int).unique(),reverse=True)
    season=seasons[0]
    sf=f[f.season==season]
    unplayed=sf[sf.home_score.isna() | sf.away_score.isna()]
    week=int(sorted(unplayed.week.dropna().astype(int).unique())[0]) if not unplayed.empty else int(sf.week.dropna().astype(int).max())
    return season,week

DEFAULT_SEASON,DEFAULT_WEEK=relevant_season_week()

def selectors(prefix=""):
    seasons=sorted(f.season.dropna().astype(int).unique(),reverse=True)
    season=st.selectbox("Season",seasons,index=seasons.index(DEFAULT_SEASON) if DEFAULT_SEASON in seasons else 0,key=prefix+"season")
    weeks=sorted(f[f.season==season].week.dropna().astype(int).unique())
    target=DEFAULT_WEEK if season==DEFAULT_SEASON and DEFAULT_WEEK in weeks else weeks[-1]
    week=st.selectbox("Week",weeks,index=weeks.index(target),key=prefix+"week")
    slate=f[(f.season==season)&(f.week==week)]
    return season,week,slate

def market_text(row):
    s=row.get("spread_line",np.nan)
    if pd.isna(s): return "No spread"
    s=float(s)
    if s>0:return f"{row.home_team} -{abs(s):g}"
    if s<0:return f"{row.away_team} -{abs(s):g}"
    return "Pick'em"

def vulnerability(score):
    if pd.isna(score): return "—"
    return "HIGH" if score>=45 else ("WATCH" if score>=30 else "LOW")

def matchup_select(slate,key):
    labels=(slate.away_team+" @ "+slate.home_team).tolist()
    if not labels:return None
    label=st.selectbox("Matchup",labels,key=key)
    return slate.iloc[labels.index(label)]

def game_card(row,n=30000):
    r=m.predict(row,n,seed=int(row.name)%10000)
    a,b,c,d=st.columns(4)
    a.metric("Projected score",f"{row.away_team} {r['away_score']:.1f} — {row.home_team} {r['home_score']:.1f}")
    b.metric(f"{row.home_team} win",f"{r['home_win_prob']:.1%}")
    c.metric("Model margin",f"{r['margin']:+.1f}")
    d.metric("Model total",f"{r['total']:.1f}")
    if "upset_score" in r:
        level="HIGH" if r["upset_score"]>=45 else ("WATCH" if r["upset_score"]>=30 else "LOW")
        st.markdown(f"### 🚨 Upset Watch: {level} · {r['upset_score']:.0f}/100")
        st.write(f"Favorite loss probability: **{r['upset_prob']:.1%}**")
    return r

if page=="🏈 GAMES":
    st.header("This Week")
    season,week,slate=selectors("g")
    st.caption(f"{season} • Week {week} • Pure model; market lines are comparison only.")
    if slate.empty:
        st.info("No games found for this selection.")
    for _,row in slate.iterrows():
        r=m.predict(row,18000,seed=int(row.name)%10000)
        edge=(r["margin"]-float(row.spread_line)) if pd.notna(row.get("spread_line",np.nan)) else np.nan
        pick=row.home_team if r["home_win_prob"]>=.5 else row.away_team
        ou=row.get("total_line",np.nan)
        ou_txt=f"{float(ou):g}" if pd.notna(ou) else "—"
        edge_txt=f"{edge:+.1f}" if pd.notna(edge) else "—"
        st.markdown(
            f"""<div class="edge-card">
            <div class="edge-game">{row.away_team} @ {row.home_team}</div>
            <div class="edge-score">NFL Edge: {row.away_team} {r['away_score']:.1f} — {row.home_team} {r['home_score']:.1f}</div>
            <div class="edge-muted">Pick: <b>{pick}</b> · Home win {r['home_win_prob']:.0%} · Market {market_text(row)} · O/U {ou_txt}</div>
            <div class="edge-muted">Model margin {r['margin']:+.1f} · Total {r['total']:.1f} · Raw spread edge {edge_txt} · Vulnerability <b>{vulnerability(r.get('upset_score',np.nan))}</b></div>
            </div>""",unsafe_allow_html=True)
    st.caption("Raw spread edge = model home margin − nflverse home-margin line. Positive leans home ATS; negative leans away ATS.")
    st.info("V0.8 confidence/vulnerability indicators are research signals, not guarantees. PASS remains a valid model outcome.")

elif page=="🔬 SIMULATOR":
    st.header("Game Simulator")
    _,_,slate=selectors("s")
    row=matchup_select(slate,"sm")
    if row is not None:
        n=st.slider("Monte Carlo simulations",10000,100000,50000,10000)
        r=game_card(row,n)
        lo_h,hi_h=np.percentile(r["home_scores"],[20,80])
        lo_a,hi_a=np.percentile(r["away_scores"],[20,80])
        st.subheader("20–80% simulated score range")
        ra,rb=st.columns(2)
        ra.metric(row.away_team,f"{lo_a:.0f}–{hi_a:.0f}")
        rb.metric(row.home_team,f"{lo_h:.0f}–{hi_h:.0f}")
        if "home_cover_prob" in r or "over_prob" in r:
            a,b=st.columns(2)
            if "home_cover_prob" in r:a.metric("Home cover probability",f"{r['home_cover_prob']:.1%}")
            if "over_prob" in r:b.metric("Over probability",f"{r['over_prob']:.1%}")
        hs=pd.Series(np.round(r["home_scores"]).astype(int)).value_counts().sort_index()
        st.subheader(f"{row.home_team} simulated score distribution")
        st.bar_chart(hs)

elif page in ["📊 PROPS","🏆 FANTASY"]:
    ps=players()
    season,week,slate=selectors("p" if page=="📊 PROPS" else "f")
    row=matchup_select(slate,"pm" if page=="📊 PROPS" else "fm")
    if row is not None:
        gr=m.predict(row,30000)
        teams=[row.away_team,row.home_team]
        team=st.selectbox("Team",teams,key="team_"+page)
        pts=gr["away_score"] if team==row.away_team else gr["home_score"]
        usage=player_usage_before_week(ps,season,week,team)
        proj=project_team_players(usage,pts,gr["total"])
        if proj.empty:
            st.warning("No prior-week player usage available for this selection.")
        elif page=="🏆 FANTASY":
            scoring=st.radio("Scoring",["PPR","Half-PPR","Standard"],horizontal=True)
            proj["Fantasy"]=proj.apply(lambda x:fantasy_points(x,scoring),axis=1)
            simrows=[]
            for i,p in proj.iterrows():
                s=simulate_player(p,scoring,20000,seed=i+7)
                simrows.append({"Player":p.player,"Pos":p.position,"Projection":p.Fantasy,
                                "Floor":s["floor"],"Median":s["median"],"Ceiling":s["ceiling"]})
            table=pd.DataFrame(simrows).sort_values("Projection",ascending=False)
            st.dataframe(table.round(1),use_container_width=True,hide_index=True)
            st.subheader("Start / Sit")
            names=table.Player.tolist()
            c1,c2=st.columns(2)
            a=c1.selectbox("Player A",names,index=0)
            b=c2.selectbox("Player B",names,index=min(1,len(names)-1))
            pa=proj[proj.player==a].iloc[0]; pb=proj[proj.player==b].iloc[0]
            sa=simulate_player(pa,scoring,40000,101); sb=simulate_player(pb,scoring,40000,202)
            chance=float(np.mean(sa["fantasy_samples"]>sb["fantasy_samples"]))
            st.metric(f"{a} probability of outscoring {b}",f"{chance:.1%}")
        else:
            player=st.selectbox("Player",proj.player.tolist())
            p=proj[proj.player==player].iloc[0]
            sim=simulate_player(p,"PPR",50000)
            stat=st.selectbox("Prop",["Passing Yards","Rushing Yards","Receiving Yards","Receptions","Targets","Carries"])
            mapping={"Passing Yards":("passing_yards","proj_passing_yards"),
                     "Rushing Yards":("rushing_yards","proj_rushing_yards"),
                     "Receiving Yards":("receiving_yards","proj_receiving_yards"),
                     "Receptions":("receptions","proj_receptions"),"Targets":("targets","proj_targets"),
                     "Carries":("carries","proj_carries")}
            sk,pk=mapping[stat]; mean=float(p[pk])
            line=st.number_input("Sportsbook line",value=round(mean*2)/2,step=.5)
            over=prop_probability(sim[sk],line,True); under=1-over
            a,b,c=st.columns(3)
            a.metric("NFL Edge projection",f"{mean:.1f}")
            b.metric("Over",f"{over:.1%}"); c.metric("Under",f"{under:.1%}")
            best=max(over,under); side="OVER" if over>under else "UNDER"; label=edge_label(best)
            if label=="PASS":st.info(f"PASS · strongest side {side} {best:.1%}")
            else:st.success(f"{side} · {label} · {best:.1%}")
            st.caption("Manual market lines for now. Automated odds/prop ingestion comes after model validation.")

elif page=="📈 MODEL RECORD":
    st.header("Model Record & Validation")
    st.write("V0.9 tests each NFL week using only information available before that week's games.")
    st.caption("This is stricter than the prior season-level backtest and is designed to expose overfitting.")
    if st.button("Run v0.9 walk-forward validation",type="primary"):
        with st.spinner("Training and grading each historical week..."): bt=walk_forward_backtest(g,2022)
        if not bt.empty:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Winner accuracy",f"{bt.winner_correct.mean():.1%}")
            c2.metric("Brier score",f"{bt.brier.mean():.3f}")
            c3.metric("Margin MAE",f"{bt.margin_abs_error.mean():.2f}")
            c4.metric("Total MAE",f"{bt.total_abs_error.mean():.2f}")
            st.subheader("Probability calibration")
            cal=calibration_table(bt)
            if not cal.empty:
                cal["Bucket"]=cal.bucket.astype(str)
                cal["Predicted"]=cal.Predicted.map(lambda x:f"{x:.1%}")
                cal["Actual"]=cal.Actual.map(lambda x:f"{x:.1%}")
                st.dataframe(cal[["Bucket","Games","Predicted","Actual"]],use_container_width=True,hide_index=True)
            st.subheader("Model vs market ATS research")
            ats=bt.dropna(subset=["model_market_margin_gap","ats_result"])
            rows=[]
            for threshold in [0,2,3,4,5,7]:
                x=ats[ats.model_market_margin_gap.abs()>=threshold]
                rows.append({"Minimum raw edge":threshold,"Games":len(x),
                             "ATS win rate":float(x.ats_result.astype(float).mean()) if len(x) else np.nan})
            et=pd.DataFrame(rows)
            et["ATS win rate"]=et["ATS win rate"].map(lambda x:f"{x:.1%}" if pd.notna(x) else "—")
            st.dataframe(et,use_container_width=True,hide_index=True)
            st.subheader("Year-by-year stability")
            yr=bt.groupby("season").agg(Games=("Game","size"),Winner=("winner_correct","mean"),
                Brier=("brier","mean"),Margin_MAE=("margin_abs_error","mean"),Total_MAE=("total_abs_error","mean")).reset_index()
            yr["Winner Accuracy"]=yr.Winner.map(lambda x:f"{x:.1%}")
            st.dataframe(yr[["season","Games","Winner Accuracy","Brier","Margin_MAE","Total_MAE"]].round(3),
                         use_container_width=True,hide_index=True)
            st.caption("Lower Brier is better. ATS thresholds are diagnostics, not betting guarantees.")
    st.divider()
    st.subheader("🧪 v1.0 Efficiency Challenger")
    st.write("The live v0.9 model stays protected unless the EPA/success-rate challenger beats it on the same historical weeks.")
    st.caption("Adds rolling offensive/defensive EPA, success rate, pass/rush EPA, explosive rate, pass rate, and matchup interactions. Market lines are not inputs.")
    if st.button("Run v1.0 baseline vs challenger test"):
        with st.spinner("Downloading nflverse play-by-play and running identical weekly folds..."):
            cmp=compare_v09_vs_v10(g,2022)
        if cmp.empty:
            st.error("Efficiency data could not be loaded. The live baseline remains unchanged.")
        else:
            sc=model_scorecard(cmp)
            show=sc.copy()
            show["Winner Accuracy"]=show.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(show[["model","Games","Winner Accuracy","Brier","Margin_MAE","Total_MAE"]].round(3),
                         use_container_width=True,hide_index=True)
            decision=promotion_decision(sc)
            if decision=="PROMOTE":
                st.success("PROMOTION GATE: PASS — the efficiency challenger earned promotion criteria.")
            else:
                st.warning("PROMOTION GATE: DO NOT PROMOTE — keep v0.9 as the live baseline and continue feature research.")
            st.subheader("Year-by-year comparison")
            yr=cmp.groupby(["season","model"]).agg(
                Games=("winner_correct","size"),Winner=("winner_correct","mean"),
                Brier=("brier","mean"),Margin_MAE=("margin_abs_error","mean"),
                Total_MAE=("total_abs_error","mean")).reset_index()
            yr["Winner Accuracy"]=yr.Winner.map(lambda x:f"{x:.1%}")
            st.dataframe(yr[["season","model","Games","Winner Accuracy","Brier","Margin_MAE","Total_MAE"]].round(3),
                         use_container_width=True,hide_index=True)
    st.divider()
    st.subheader("🏟️ v1.1 Model Tournament")
    st.write("Ridge baseline vs linear efficiency vs nonlinear gradient boosting — all tested on identical historical weeks.")
    if st.button("Run v1.1 model tournament"):
        with st.spinner("Running three architectures across the walk-forward history..."):
            tour=v11_model_tournament(g,2022)
        if tour.empty:
            st.error("Tournament data could not be produced. Baseline remains protected.")
        else:
            ts=tournament_summary(tour)
            display=ts.copy()
            display["Winner Accuracy"]=display.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(display[["candidate","Games","Winner Accuracy","Brier","Margin_MAE","Total_MAE"]].round(3),
                         use_container_width=True,hide_index=True)
            winners=component_winners(ts)
            st.markdown("**Best component by metric**")
            for metric,name in winners.items(): st.write(f"• {metric}: **{name}**")
            if len(set(winners.values()))>1:
                st.info("HYBRID SIGNAL: different models win different objectives. NFL Edge should not force one universal model.")
            else:
                st.success(f"CONSENSUS SIGNAL: {next(iter(winners.values()))} leads all core objectives.")

    st.subheader("🧬 Feature Ablation")
    st.write("Tests feature families independently so useful football signal is not buried inside a noisy all-features model.")
    if st.button("Run v1.1 feature ablation"):
        with st.spinner("Testing EPA, pass efficiency, rush efficiency, success rate, explosives, and the full set..."):
            ab=v11_ablation_test(g,2022)
        if ab.empty:
            st.error("Ablation data could not be produced.")
        else:
            sm=weighted_ablation_summary(ab)
            baseline=sm[sm["Feature group"]=="Baseline"].iloc[0]
            sm["Δ Brier"]=sm.Brier-baseline.Brier
            sm["Δ Margin MAE"]=sm["Margin MAE"]-baseline["Margin MAE"]
            sm["Δ Total MAE"]=sm["Total MAE"]-baseline["Total MAE"]
            sm["Winner Accuracy"]=sm["Winner Accuracy"].map(lambda x:f"{x:.1%}")
            st.dataframe(sm[["Feature group","Games","Winner Accuracy","Brier","Margin MAE","Total MAE",
                             "Δ Brier","Δ Margin MAE","Δ Total MAE"]].round(3),
                         use_container_width=True,hide_index=True)
            st.caption("Negative deltas are improvements versus v0.9. This table determines which feature families advance to the next hybrid model.")

    st.divider()
    st.subheader("🏆 v1.2 Hybrid Championship")
    st.write("Only finalists that earned advancement in v1.1 compete here: baseline, EPA Core, Pass Efficiency, and nonlinear Efficiency Boost.")
    st.caption("Every finalist sees the same historical games and the same weekly walk-forward cutoffs.")
    if st.button("Run v1.2 Hybrid Championship",type="primary"):
        with st.spinner("Running finalist models and paired bootstrap validation..."):
            champ=v12_championship(g,2022)
        if champ.empty:
            st.error("Championship data could not be produced. No production model changes were made.")
        else:
            sc=v12_summary(champ)
            show=sc.copy()
            show["Winner Accuracy"]=show.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(show[["candidate","Games","Winner Accuracy","Brier","Margin_MAE","Total_MAE"]].round(3),
                         use_container_width=True,hide_index=True)

            winners=v12_champions(sc)
            st.subheader("Component Champions")
            for metric,name in winners.items(): st.write(f"**{metric}:** {name}")

            st.subheader("Year-by-year consistency")
            yr=v12_yearly(champ); yrshow=yr.copy()
            yrshow["Winner Accuracy"]=yrshow.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(yrshow[["season","candidate","Games","Winner Accuracy","Brier","Margin_MAE","Total_MAE"]].round(3),
                         use_container_width=True,hide_index=True)

            st.subheader("Paired bootstrap vs v0.9 baseline")
            boot=v12_bootstrap_table(champ)
            st.dataframe(boot.round(4),use_container_width=True,hide_index=True)
            clear=boot[boot["Clear improvement"]]
            if len(clear):
                st.success(f"{len(clear)} candidate/metric improvements have a 95% bootstrap interval entirely better than baseline.")
            else:
                st.warning("No finalist improvement clears the 95% paired-bootstrap bar yet. Treat aggregate gains as promising, not proven.")

            st.subheader("Recommended production architecture")
            st.write(f"Win probability → **{winners['Probability']}**")
            st.write(f"Game margin → **{winners['Margin']}**")
            st.write(f"Game total → **{winners['Total']}**")
            st.caption("The championship identifies production candidates; it does not silently rewrite the live engine.")

    st.divider()
    st.subheader("⚙️ v1.3 Game Engine 1.0")
    st.write("Production architecture assembled from the v1.2 evidence.")
    st.markdown("""
**Win probability:** EPA Core Ridge  
**Game margin:** protected Baseline Ridge  
**Game total:** Pass Ridge  
**Shadow probability:** v0.9 baseline  
**Shadow margin:** Efficiency Boost
""")
    st.caption("Shadow models are monitored but do not affect the live projection unless they later earn promotion.")

    if st.button("Validate Game Engine 1.0",type="primary"):
        with st.spinner("Testing the exact production architecture against the protected baseline..."):
            eng=v13_engine_backtest(g,2022)
        if eng.empty:
            st.error("Game Engine validation could not be produced. No live changes were made.")
        else:
            es=v13_summary(eng); eshow=es.copy()
            eshow["Winner Accuracy"]=eshow.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(eshow[["model","Games","Winner Accuracy","Brier","Margin_MAE","Total_MAE"]].round(3),
                         use_container_width=True,hide_index=True)

            st.subheader("Year-by-year engine stability")
            ey=v13_yearly(eng); eyshow=ey.copy()
            eyshow["Winner Accuracy"]=eyshow.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(eyshow[["season","model","Games","Winner Accuracy","Brier","Margin_MAE","Total_MAE"]].round(3),
                         use_container_width=True,hide_index=True)

            base=es[es.model=="v0.9 baseline"].iloc[0]
            new=es[es.model=="Game Engine 1.0"].iloc[0]
            st.markdown("### Engine delta vs v0.9")
            st.write(f"Winner accuracy: **{(new.Winner_Accuracy-base.Winner_Accuracy):+.2%}**")
            st.write(f"Brier: **{(new.Brier-base.Brier):+.4f}**")
            st.write(f"Margin MAE: **{(new.Margin_MAE-base.Margin_MAE):+.3f}**")
            st.write(f"Total MAE: **{(new.Total_MAE-base.Total_MAE):+.3f}**")
            if new.Total_MAE < base.Total_MAE and new.Winner_Accuracy >= base.Winner_Accuracy:
                st.success("ENGINE GATE: PASS — specialized architecture improves the validated profile without sacrificing straight-up accuracy.")
            else:
                st.warning("ENGINE GATE: HOLD — keep researching before treating the specialized architecture as final.")

    st.info("Next research layer: quarterback availability/quality, injuries, offensive line, opponent adjustment, offseason carryover, travel/rest, and weather — each added as a separately validated feature family.")

st.divider()
st.caption("NFL Edge v1.3 • Probabilistic research software. A projection is not a guarantee, and PASS is a valid model decision.")
