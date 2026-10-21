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



# ---------- v2.1 Player Projection Championship ----------
PLAYER_CHAMP_STATS = {
    "Passing Yards": ("proj_Passing Yards", "actual_Passing Yards", {"QB"}),
    "Rushing Yards": ("proj_Rushing Yards", "actual_Rushing Yards", {"QB","RB","FB"}),
    "Receiving Yards": ("proj_Receiving Yards", "actual_Receiving Yards", {"WR","TE","RB","FB"}),
    "Receptions": ("proj_Receptions", "actual_Receptions", {"WR","TE","RB","FB"}),
    "Targets": ("proj_Targets", "actual_Targets", {"WR","TE","RB","FB"}),
    "Carries": ("proj_Carries", "actual_Carries", {"QB","RB","FB"}),
}

def v21_player_championship(stats, start_season=2022, end_season=2026, start_week=4):
    """Chronological, prior-week-only audit of the frozen player opportunity engine.
    No current-week production is used in its own projection.
    """
    frames=[]
    for season in range(start_season,end_season+1):
        try:
            bt=backtest_players(stats,season,"PPR",start_week=start_week)
        except Exception:
            bt=pd.DataFrame()
        if not bt.empty: frames.append(bt)
    if not frames: return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    x=pd.concat(frames,ignore_index=True)
    rows=[]
    for stat,(pc,ac,positions) in PLAYER_CHAMP_STATS.items():
        g=x[x.position.astype(str).isin(positions)].copy()
        if g.empty: continue
        pred=pd.to_numeric(g[pc],errors="coerce"); actual=pd.to_numeric(g[ac],errors="coerce")
        ok=pred.notna()&actual.notna(); g=g[ok].copy(); pred=pred[ok]; actual=actual[ok]
        if len(g)<25: continue
        err=(pred-actual).abs(); rmse=float(np.sqrt(np.mean((pred-actual)**2)))
        bias=float(np.mean(pred-actual)); mae=float(err.mean())
        denom=float(np.mean(np.abs(actual)))
        nmae=mae/denom if denom>0 else np.nan
        corr=float(np.corrcoef(pred,actual)[0,1]) if pred.std()>0 and actual.std()>0 else np.nan
        # Reliability tiers are deliberately conservative and descriptive, not betting-edge labels.
        if len(g)>=250 and corr>=.55 and nmae<=.55: tier="STRONG"
        elif len(g)>=150 and corr>=.40 and nmae<=.75: tier="USABLE"
        else: tier="RESEARCH"
        rows.append({"Market":stat,"Player-games":len(g),"MAE":mae,"RMSE":rmse,"Bias":bias,"Correlation":corr,"Normalized MAE":nmae,"Research tier":tier})
    market=pd.DataFrame(rows).sort_values(["Research tier","Normalized MAE"],ascending=[True,True]) if rows else pd.DataFrame()
    pos=x.groupby("position").agg(Player_games=("player","size"),Fantasy_MAE=("ae_fantasy","mean")).reset_index().sort_values("Player_games",ascending=False)
    yr=x.groupby("season").agg(Player_games=("player","size"),Fantasy_MAE=("ae_fantasy","mean")).reset_index()
    return market,pos,yr

def v21_td_audit(stats,start_season=2022,end_season=2026,start_week=4):
    """TD projections are audited separately because rare-event MAE can look deceptively good."""
    rows=[]
    for season in range(start_season,end_season+1):
        d=stats[stats.season==season].copy()
        for week in sorted(d.week.dropna().astype(int).unique()):
            if week<start_week: continue
            actual=d[d.week==week]
            for team in actual.recent_team.dropna().unique():
                u=player_usage_before_week(stats,season,week,team)
                if u.empty: continue
                pr=project_team_players(u)
                for _,r in pr.iterrows():
                    a=actual[actual.player_id==r.player_id] if "player_id" in actual and "player_id" in r else pd.DataFrame()
                    if a.empty: continue
                    a=a.iloc[0]
                    for label,pc,ac,poss in [("Passing TDs","proj_passing_tds","passing_tds",{"QB"}),("Rushing TDs","proj_rushing_tds","rushing_tds",{"QB","RB","FB"}),("Receiving TDs","proj_receiving_tds","receiving_tds",{"WR","TE","RB","FB"})]:
                        if str(r.position) not in poss: continue
                        pv=float(r.get(pc,0) or 0); av=float(a.get(ac,0) or 0)
                        rows.append({"Market":label,"pred":pv,"actual":av})
    if not rows:return pd.DataFrame()
    z=pd.DataFrame(rows)
    return z.groupby("Market").apply(lambda q:pd.Series({"Player-games":len(q),"MAE":float(np.mean(np.abs(q.pred-q.actual))),"Bias":float(np.mean(q.pred-q.actual)),"Actual TD rate":float(q.actual.mean())}),include_groups=False).reset_index()


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


# ---------- v1.4 Football Intelligence: QB Quality ----------
# This first football-intelligence challenger is deliberately PRE-GAME and leakage-safe.
# It measures the recent quality of each team's QB room from completed prior weeks only.
# It does NOT claim to know a same-week starter injury/benching before we add a true injury/depth-chart feed.
QB_QUALITY_FEATURES = [
    "home_qb_epa","away_qb_epa","qb_epa_diff",
    "home_qb_ypa","away_qb_ypa","qb_ypa_diff",
    "home_qb_comp","away_qb_comp","qb_comp_diff",
    "home_qb_td_rate","away_qb_td_rate","qb_td_rate_diff",
    "home_qb_int_rate","away_qb_int_rate","qb_int_rate_diff"
]
QB_PROB_FEATURES = EPA_CORE_FEATURES + QB_QUALITY_FEATURES

def qb_team_week_table(stats):
    """Build team/week QB performance, then shift rolling form so week W uses only weeks < W."""
    if stats is None or stats.empty:return pd.DataFrame()
    d=stats.copy()
    if "position" in d.columns:d=d[d.position.astype(str).eq("QB")]
    need=["season","week","recent_team","attempts"]
    if any(c not in d.columns for c in need):return pd.DataFrame()
    for c in ["season","week","attempts","completions","passing_yards","passing_tds","interceptions","passing_epa"]:
        if c in d.columns:d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0)
    d=d[d.attempts>0].copy()
    if d.empty:return pd.DataFrame()
    # Sum QB room production. This handles games with multiple QBs without peeking at a future starter identity.
    agg={"attempts":"sum"}
    for c in ["completions","passing_yards","passing_tds","interceptions","passing_epa"]:
        if c in d.columns:agg[c]="sum"
    w=d.groupby(["season","week","recent_team"],as_index=False).agg(agg).rename(columns={"recent_team":"team"})
    for c in ["completions","passing_yards","passing_tds","interceptions","passing_epa"]:
        if c not in w:w[c]=0.0
    den=w.attempts.clip(lower=1)
    w["qb_epa"]=w.passing_epa/den
    w["qb_ypa"]=w.passing_yards/den
    w["qb_comp"]=w.completions/den
    w["qb_td_rate"]=w.passing_tds/den
    w["qb_int_rate"]=w.interceptions/den
    metrics=["qb_epa","qb_ypa","qb_comp","qb_td_rate","qb_int_rate"]
    w=w.sort_values(["team","season","week"])
    for m in metrics:
        # Current week excluded. Up to six prior games, with minimum one observation.
        w[m+"_pregame"]=(w.groupby(["team","season"])[m]
                           .transform(lambda s:s.shift(1).rolling(6,min_periods=1).mean()))
    return w[["season","week","team"]+[m+"_pregame" for m in metrics]]

def add_qb_quality_features(feature_frame, stats):
    """Merge leakage-safe team QB form onto the existing game feature frame."""
    f=feature_frame.copy()
    q=qb_team_week_table(stats)
    if q.empty:
        for c in QB_QUALITY_FEATURES:f[c]=np.nan
        return f
    home=q.rename(columns={"team":"home_team",**{m+"_pregame":"home_"+m for m in ["qb_epa","qb_ypa","qb_comp","qb_td_rate","qb_int_rate"]}})
    away=q.rename(columns={"team":"away_team",**{m+"_pregame":"away_"+m for m in ["qb_epa","qb_ypa","qb_comp","qb_td_rate","qb_int_rate"]}})
    f=f.merge(home,on=["season","week","home_team"],how="left")
    f=f.merge(away,on=["season","week","away_team"],how="left")
    for m in ["qb_epa","qb_ypa","qb_comp","qb_td_rate","qb_int_rate"]:
        f[m+"_diff"]=f["home_"+m]-f["away_"+m]
    return f

def v14_qb_quality_backtest(games, stats, first_test_season=2022):
    """Game Engine 1.0 vs identical engine with QB quality added ONLY to win probability."""
    f=add_qb_quality_features(efficiency_pregame_features(games),stats)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    test=done[(done.season>=first_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min()
        tr=done[(done.gameday<cutoff)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(tr)<300:continue
        actual=(te.target_margin>0).astype(float).to_numpy()
        specs=[("Game Engine 1.0",EPA_CORE_FEATURES),("QB Quality Challenger",QB_PROB_FEATURES)]
        for name,features in specs:
            wm=_linear_classifier(); wm.fit(tr[features],(tr.target_margin>0).astype(int))
            wp=wm.predict_proba(te[features])[:,1]
            for i,(_,r) in enumerate(te.iterrows()):
                rows.append({"model":name,"season":int(season),"week":int(week),
                             "winner_correct":bool((wp[i]>=.5)==bool(actual[i])),
                             "brier":float((wp[i]-actual[i])**2)})
    return pd.DataFrame(rows)

def v14_qb_summary(x):
    if x.empty:return pd.DataFrame()
    return x.groupby("model").agg(Games=("winner_correct","size"),
        Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean")).reset_index()

def v14_qb_yearly(x):
    if x.empty:return pd.DataFrame()
    return x.groupby(["season","model"]).agg(Games=("winner_correct","size"),
        Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean")).reset_index()

def v14_qb_bootstrap(x,n_boot=4000,seed=41):
    """Paired game-level bootstrap of challenger minus Game Engine 1.0. Negative Brier is better."""
    if x.empty:return pd.DataFrame()
    keys=["season","week"]
    # Pair by stable within-week row order because both candidates are emitted over identical te rows.
    b=x[x.model=="Game Engine 1.0"].reset_index(drop=True)
    c=x[x.model=="QB Quality Challenger"].reset_index(drop=True)
    n=min(len(b),len(c))
    if n==0:return pd.DataFrame()
    b=b.iloc[:n]; c=c.iloc[:n]
    d_acc=c.winner_correct.astype(float).to_numpy()-b.winner_correct.astype(float).to_numpy()
    d_brier=c.brier.to_numpy()-b.brier.to_numpy()
    rng=np.random.default_rng(seed); idx=rng.integers(0,n,size=(n_boot,n))
    acc=np.mean(d_acc[idx],axis=1); br=np.mean(d_brier[idx],axis=1)
    return pd.DataFrame([
        {"Metric":"Winner accuracy","Delta":float(d_acc.mean()),"95% CI low":float(np.quantile(acc,.025)),"95% CI high":float(np.quantile(acc,.975)),"Clear improvement":bool(np.quantile(acc,.025)>0)},
        {"Metric":"Brier","Delta":float(d_brier.mean()),"95% CI low":float(np.quantile(br,.025)),"95% CI high":float(np.quantile(br,.975)),"Clear improvement":bool(np.quantile(br,.975)<0)}
    ])

# ---------- v1.5 Football Intelligence: QB Availability / Starter Adjustment ----------
# Historical validation uses official weekly depth charts + injury reports only.
# We intentionally restrict the backtest to seasons where the upstream data carries a week field
# and can be aligned pregame without inferring the actual game starter from postgame stats.
DEPTH_URL_TEMPLATE="https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.csv"
INJURY_URL_TEMPLATE="https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
QB_AVAIL_FEATURES=[
    "home_qb_out","away_qb_out","qb_out_diff",
    "home_qb_questionable","away_qb_questionable","qb_questionable_diff",
    "home_qb_practice_dnp","away_qb_practice_dnp","qb_practice_dnp_diff",
    "home_qb_practice_limited","away_qb_practice_limited","qb_practice_limited_diff"
]
QB_AVAIL_PROB_FEATURES=EPA_CORE_FEATURES+QB_AVAIL_FEATURES

@lru_cache(maxsize=1)
def load_qb_availability_history(start_season=2022,end_season=2024):
    """Official weekly QB1 depth-chart identity joined to that week's injury/practice report."""
    rows=[]
    for season in range(start_season,end_season+1):
        try:
            dc=pd.read_csv(DEPTH_URL_TEMPLATE.format(season=season))
            inj=pd.read_csv(INJURY_URL_TEMPLATE.format(season=season))
        except Exception:
            continue
        if dc.empty or not {"season","week"}.issubset(dc.columns): continue
        team_col="club_code" if "club_code" in dc.columns else "team" if "team" in dc.columns else None
        if team_col is None: continue
        # Legacy nflverse depth charts: QB rows + depth_team 1 represent the listed QB1.
        q=dc.copy()
        if "position" in q.columns: q=q[q.position.astype(str).str.upper().eq("QB")]
        elif "depth_position" in q.columns: q=q[q.depth_position.astype(str).str.upper().str.contains("QB",na=False)]
        if "depth_team" in q.columns:
            dep=pd.to_numeric(q.depth_team,errors="coerce")
            if dep.notna().any(): q=q[dep.eq(dep.groupby([q.season,q.week,q[team_col]]).transform("min"))]
        id_col="gsis_id" if "gsis_id" in q.columns and "gsis_id" in inj.columns else None
        name_col="full_name" if "full_name" in q.columns and "full_name" in inj.columns else None
        keep=["season","week",team_col]+([id_col] if id_col else [])+([name_col] if name_col else [])
        q=q[keep].drop_duplicates(["season","week",team_col],keep="first").rename(columns={team_col:"team"})
        ii=inj.copy()
        if "position" in ii.columns: ii=ii[ii.position.astype(str).str.upper().eq("QB")]
        join=["season","week"]
        if id_col: join.append(id_col)
        elif name_col: join.append(name_col)
        else: continue
        cols=join+[c for c in ["team","report_status","practice_status","date_modified"] if c in ii.columns]
        ii=ii[cols].copy()
        if "date_modified" in ii.columns:
            ii["date_modified"]=pd.to_datetime(ii.date_modified,errors="coerce",utc=True)
            ii=ii.sort_values("date_modified").drop_duplicates(join,keep="last")
        else: ii=ii.drop_duplicates(join,keep="last")
        z=q.merge(ii,on=join,how="left",suffixes=("","_inj"))
        rs=z.get("report_status",pd.Series("",index=z.index)).fillna("").astype(str).str.lower()
        ps=z.get("practice_status",pd.Series("",index=z.index)).fillna("").astype(str).str.lower()
        z["qb_out"]=rs.str.contains(r"out|doubtful",regex=True).astype(float)
        z["qb_questionable"]=rs.str.contains("question",regex=False).astype(float)
        z["qb_practice_dnp"]=ps.str.contains(r"did not|dnp|not particip",regex=True).astype(float)
        z["qb_practice_limited"]=ps.str.contains("limit",regex=False).astype(float)
        rows.append(z[["season","week","team","qb_out","qb_questionable","qb_practice_dnp","qb_practice_limited"]])
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

def add_qb_availability_features(feature_frame, availability):
    f=feature_frame.copy()
    if availability is None or availability.empty:
        for c in QB_AVAIL_FEATURES:f[c]=np.nan
        return f
    metrics=["qb_out","qb_questionable","qb_practice_dnp","qb_practice_limited"]
    h=availability.rename(columns={"team":"home_team",**{m:"home_"+m for m in metrics}})
    a=availability.rename(columns={"team":"away_team",**{m:"away_"+m for m in metrics}})
    f=f.merge(h,on=["season","week","home_team"],how="left").merge(a,on=["season","week","away_team"],how="left")
    for m in metrics:
        # No report on the listed QB1 is treated as healthy/no flag, not missing.
        f["home_"+m]=f["home_"+m].fillna(0.0); f["away_"+m]=f["away_"+m].fillna(0.0)
        f[m+"_diff"]=f["home_"+m]-f["away_"+m]
    return f

def v15_qb_availability_backtest(games, availability, first_test_season=2022, last_test_season=2024):
    """Game Engine 1.0 vs QB availability challenger on identical 2022-24 historical games."""
    f=add_qb_availability_features(efficiency_pregame_features(games),availability)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    test=done[(done.season>=first_test_season)&(done.season<=last_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min()
        tr=done[(done.gameday<cutoff)&(done.season<=last_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        # Availability model needs historical rows from the validated availability era.
        tr=tr[tr.season>=first_test_season]
        if len(tr)<250: continue
        actual=(te.target_margin>0).astype(float).to_numpy()
        for name,features in [("Game Engine 1.0",EPA_CORE_FEATURES),("QB Availability Challenger",QB_AVAIL_PROB_FEATURES)]:
            wm=_linear_classifier(); wm.fit(tr[features],(tr.target_margin>0).astype(int)); wp=wm.predict_proba(te[features])[:,1]
            for i,(_,r) in enumerate(te.iterrows()):
                rows.append({"model":name,"season":int(season),"week":int(week),"actual_win":actual[i],"win_prob":float(wp[i]),
                             "winner_correct":bool((wp[i]>=.5)==bool(actual[i]))})
    return pd.DataFrame(rows)

def v15_qb_availability_summary(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["brier"]=(y.win_prob-y.actual_win)**2
    return y.groupby("model").agg(Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean")).reset_index()

def v15_qb_availability_yearly(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["brier"]=(y.win_prob-y.actual_win)**2
    return y.groupby(["season","model"]).agg(Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean")).reset_index()

def v15_qb_availability_bootstrap(x,n_boot=4000,seed=51):
    if x.empty:return pd.DataFrame()
    b=x[x.model=="Game Engine 1.0"].reset_index(drop=True); c=x[x.model=="QB Availability Challenger"].reset_index(drop=True)
    n=min(len(b),len(c));
    if n==0:return pd.DataFrame()
    b=b.iloc[:n]; c=c.iloc[:n]
    da=c.winner_correct.astype(float).to_numpy()-b.winner_correct.astype(float).to_numpy()
    db=(c.win_prob-c.actual_win).pow(2).to_numpy()-(b.win_prob-b.actual_win).pow(2).to_numpy()
    rng=np.random.default_rng(seed); idx=rng.integers(0,n,size=(n_boot,n)); aa=da[idx].mean(1); bb=db[idx].mean(1)
    return pd.DataFrame([
        {"Metric":"Winner accuracy","Delta":float(da.mean()),"95% CI low":float(np.quantile(aa,.025)),"95% CI high":float(np.quantile(aa,.975)),"Clear improvement":bool(np.quantile(aa,.025)>0)},
        {"Metric":"Brier","Delta":float(db.mean()),"95% CI low":float(np.quantile(bb,.025)),"95% CI high":float(np.quantile(bb,.975)),"Clear improvement":bool(np.quantile(bb,.975)<0)}])


# ---------- v1.6 Football Intelligence: Offensive Line Availability ----------
# Uses only official weekly injury/practice reports. It measures OL availability burden,
# not postgame participation, and is validated as a separate challenger family.
OL_POSITIONS={"C","G","T","OT","OG","OL","LT","RT","LG","RG"}
OL_AVAIL_FEATURES=[
    "home_ol_out","away_ol_out","ol_out_diff",
    "home_ol_questionable","away_ol_questionable","ol_questionable_diff",
    "home_ol_dnp","away_ol_dnp","ol_dnp_diff",
    "home_ol_limited","away_ol_limited","ol_limited_diff",
    "home_ol_burden","away_ol_burden","ol_burden_diff"
]
OL_AVAIL_PROB_FEATURES=EPA_CORE_FEATURES+OL_AVAIL_FEATURES

@lru_cache(maxsize=1)
def load_ol_availability_history(start_season=2022,end_season=2024):
    rows=[]
    for season in range(start_season,end_season+1):
        try: inj=pd.read_csv(INJURY_URL_TEMPLATE.format(season=season))
        except Exception: continue
        if inj.empty or not {"season","week"}.issubset(inj.columns): continue
        team_col="team" if "team" in inj.columns else "club_code" if "club_code" in inj.columns else None
        if team_col is None or "position" not in inj.columns: continue
        x=inj.copy(); x["position"]=x.position.fillna("").astype(str).str.upper().str.strip()
        x=x[x.position.isin(OL_POSITIONS)]
        if x.empty: continue
        idcol="gsis_id" if "gsis_id" in x.columns else "full_name" if "full_name" in x.columns else None
        if idcol is None: continue
        # Keep the latest official weekly report row for each player, matching v1.5's pregame feed rule.
        keys=["season","week",team_col,idcol]
        if "date_modified" in x.columns:
            x["date_modified"]=pd.to_datetime(x.date_modified,errors="coerce",utc=True)
            x=x.sort_values("date_modified").drop_duplicates(keys,keep="last")
        else: x=x.drop_duplicates(keys,keep="last")
        rs=x.get("report_status",pd.Series("",index=x.index)).fillna("").astype(str).str.lower()
        ps=x.get("practice_status",pd.Series("",index=x.index)).fillna("").astype(str).str.lower()
        x["ol_out_flag"]=rs.str.contains(r"out|doubtful",regex=True).astype(float)
        x["ol_questionable_flag"]=rs.str.contains("question",regex=False).astype(float)
        x["ol_dnp_flag"]=ps.str.contains(r"did not|dnp|not particip",regex=True).astype(float)
        x["ol_limited_flag"]=ps.str.contains("limit",regex=False).astype(float)
        g=x.groupby(["season","week",team_col],as_index=False).agg(
            ol_out=("ol_out_flag","sum"),ol_questionable=("ol_questionable_flag","sum"),
            ol_dnp=("ol_dnp_flag","sum"),ol_limited=("ol_limited_flag","sum"))
        g=g.rename(columns={team_col:"team"})
        # A compact severity index is included alongside raw counts; the model decides whether it helps.
        g["ol_burden"]=g.ol_out+0.35*g.ol_questionable+0.20*g.ol_dnp+0.10*g.ol_limited
        rows.append(g)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

def add_ol_availability_features(feature_frame,ol):
    f=feature_frame.copy(); metrics=["ol_out","ol_questionable","ol_dnp","ol_limited","ol_burden"]
    if ol is None or ol.empty:
        for c in OL_AVAIL_FEATURES: f[c]=np.nan
        return f
    h=ol.rename(columns={"team":"home_team",**{m:"home_"+m for m in metrics}})
    a=ol.rename(columns={"team":"away_team",**{m:"away_"+m for m in metrics}})
    f=f.merge(h,on=["season","week","home_team"],how="left").merge(a,on=["season","week","away_team"],how="left")
    for m in metrics:
        f["home_"+m]=f["home_"+m].fillna(0.0); f["away_"+m]=f["away_"+m].fillna(0.0)
        f[m+"_diff"]=f["home_"+m]-f["away_"+m]
    return f

def v16_ol_backtest(games,ol,first_test_season=2022,last_test_season=2024):
    f=add_ol_availability_features(efficiency_pregame_features(games),ol)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    test=done[(done.season>=first_test_season)&(done.season<=last_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min(); tr=done[(done.gameday<cutoff)&(done.season>=first_test_season)&(done.season<=last_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(tr)<250: continue
        actual=(te.target_margin>0).astype(float).to_numpy()
        for name,features in [("Game Engine 1.0",EPA_CORE_FEATURES),("OL Availability Challenger",OL_AVAIL_PROB_FEATURES)]:
            wm=_linear_classifier(); wm.fit(tr[features],(tr.target_margin>0).astype(int)); wp=wm.predict_proba(te[features])[:,1]
            for i,(_,r) in enumerate(te.iterrows()):
                rows.append({"model":name,"season":int(season),"week":int(week),"actual_win":actual[i],"win_prob":float(wp[i]),"winner_correct":bool((wp[i]>=.5)==bool(actual[i]))})
    return pd.DataFrame(rows)

def v16_ol_summary(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["brier"]=(y.win_prob-y.actual_win)**2
    return y.groupby("model").agg(Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean")).reset_index()

def v16_ol_yearly(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["brier"]=(y.win_prob-y.actual_win)**2
    return y.groupby(["season","model"]).agg(Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean")).reset_index()

def v16_ol_bootstrap(x,n_boot=4000,seed=61):
    if x.empty:return pd.DataFrame()
    b=x[x.model=="Game Engine 1.0"].reset_index(drop=True); c=x[x.model=="OL Availability Challenger"].reset_index(drop=True); n=min(len(b),len(c))
    if n==0:return pd.DataFrame()
    b=b.iloc[:n]; c=c.iloc[:n]
    da=c.winner_correct.astype(float).to_numpy()-b.winner_correct.astype(float).to_numpy()
    db=(c.win_prob-c.actual_win).pow(2).to_numpy()-(b.win_prob-b.actual_win).pow(2).to_numpy()
    rng=np.random.default_rng(seed); idx=rng.integers(0,n,size=(n_boot,n)); aa=da[idx].mean(1); bb=db[idx].mean(1)
    return pd.DataFrame([
        {"Metric":"Winner accuracy","Delta":float(da.mean()),"95% CI low":float(np.quantile(aa,.025)),"95% CI high":float(np.quantile(aa,.975)),"Clear improvement":bool(np.quantile(aa,.025)>0)},
        {"Metric":"Brier","Delta":float(db.mean()),"95% CI low":float(np.quantile(bb,.025)),"95% CI high":float(np.quantile(bb,.975)),"Clear improvement":bool(np.quantile(bb,.975)<0)}])


# ---------- v1.7 Football Intelligence: Skill-Position Availability ----------
# Separately tests pregame WR/RB/TE injury/practice burden. No postgame snaps or
# same-game production are used. Game Engine 1.0 remains the protected baseline.
SKILL_POSITIONS={"WR","RB","FB","TE"}
SKILL_AVAIL_FEATURES=[
    "home_skill_out","away_skill_out","skill_out_diff",
    "home_skill_questionable","away_skill_questionable","skill_questionable_diff",
    "home_skill_dnp","away_skill_dnp","skill_dnp_diff",
    "home_skill_limited","away_skill_limited","skill_limited_diff",
    "home_skill_burden","away_skill_burden","skill_burden_diff"
]
SKILL_AVAIL_PROB_FEATURES=EPA_CORE_FEATURES+SKILL_AVAIL_FEATURES

@lru_cache(maxsize=1)
def load_skill_availability_history(start_season=2022,end_season=2024):
    rows=[]
    for season in range(start_season,end_season+1):
        try: inj=pd.read_csv(INJURY_URL_TEMPLATE.format(season=season))
        except Exception: continue
        if inj.empty or not {"season","week"}.issubset(inj.columns): continue
        team_col="team" if "team" in inj.columns else "club_code" if "club_code" in inj.columns else None
        if team_col is None or "position" not in inj.columns: continue
        x=inj.copy(); x["position"]=x.position.fillna("").astype(str).str.upper().str.strip()
        x=x[x.position.isin(SKILL_POSITIONS)]
        if x.empty: continue
        idcol="gsis_id" if "gsis_id" in x.columns else "full_name" if "full_name" in x.columns else None
        if idcol is None: continue
        keys=["season","week",team_col,idcol]
        if "date_modified" in x.columns:
            x["date_modified"]=pd.to_datetime(x.date_modified,errors="coerce",utc=True)
            x=x.sort_values("date_modified").drop_duplicates(keys,keep="last")
        else: x=x.drop_duplicates(keys,keep="last")
        rs=x.get("report_status",pd.Series("",index=x.index)).fillna("").astype(str).str.lower()
        ps=x.get("practice_status",pd.Series("",index=x.index)).fillna("").astype(str).str.lower()
        x["skill_out_flag"]=rs.str.contains(r"out|doubtful",regex=True).astype(float)
        x["skill_questionable_flag"]=rs.str.contains("question",regex=False).astype(float)
        x["skill_dnp_flag"]=ps.str.contains(r"did not|dnp|not particip",regex=True).astype(float)
        x["skill_limited_flag"]=ps.str.contains("limit",regex=False).astype(float)
        g=x.groupby(["season","week",team_col],as_index=False).agg(
            skill_out=("skill_out_flag","sum"),skill_questionable=("skill_questionable_flag","sum"),
            skill_dnp=("skill_dnp_flag","sum"),skill_limited=("skill_limited_flag","sum"))
        g=g.rename(columns={team_col:"team"})
        g["skill_burden"]=g.skill_out+0.35*g.skill_questionable+0.20*g.skill_dnp+0.10*g.skill_limited
        rows.append(g)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

def add_skill_availability_features(feature_frame,skill):
    f=feature_frame.copy(); metrics=["skill_out","skill_questionable","skill_dnp","skill_limited","skill_burden"]
    if skill is None or skill.empty:
        for c in SKILL_AVAIL_FEATURES: f[c]=np.nan
        return f
    h=skill.rename(columns={"team":"home_team",**{m:"home_"+m for m in metrics}})
    a=skill.rename(columns={"team":"away_team",**{m:"away_"+m for m in metrics}})
    f=f.merge(h,on=["season","week","home_team"],how="left").merge(a,on=["season","week","away_team"],how="left")
    for m in metrics:
        f["home_"+m]=f["home_"+m].fillna(0.0); f["away_"+m]=f["away_"+m].fillna(0.0)
        f[m+"_diff"]=f["home_"+m]-f["away_"+m]
    return f

def v17_skill_backtest(games,skill,first_test_season=2022,last_test_season=2024):
    f=add_skill_availability_features(efficiency_pregame_features(games),skill)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    test=done[(done.season>=first_test_season)&(done.season<=last_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min(); tr=done[(done.gameday<cutoff)&(done.season>=first_test_season)&(done.season<=last_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(tr)<250: continue
        actual=(te.target_margin>0).astype(float).to_numpy()
        for name,features in [("Game Engine 1.0",EPA_CORE_FEATURES),("Skill Availability Challenger",SKILL_AVAIL_PROB_FEATURES)]:
            wm=_linear_classifier(); wm.fit(tr[features],(tr.target_margin>0).astype(int)); wp=wm.predict_proba(te[features])[:,1]
            for i,(_,r) in enumerate(te.iterrows()):
                rows.append({"model":name,"season":int(season),"week":int(week),"actual_win":actual[i],"win_prob":float(wp[i]),"winner_correct":bool((wp[i]>=.5)==bool(actual[i]))})
    return pd.DataFrame(rows)

def v17_skill_summary(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["brier"]=(y.win_prob-y.actual_win)**2
    return y.groupby("model").agg(Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean")).reset_index()

def v17_skill_yearly(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["brier"]=(y.win_prob-y.actual_win)**2
    return y.groupby(["season","model"]).agg(Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean")).reset_index()

def v17_skill_bootstrap(x,n_boot=4000,seed=71):
    if x.empty:return pd.DataFrame()
    b=x[x.model=="Game Engine 1.0"].reset_index(drop=True); c=x[x.model=="Skill Availability Challenger"].reset_index(drop=True); n=min(len(b),len(c))
    if n==0:return pd.DataFrame()
    b=b.iloc[:n]; c=c.iloc[:n]
    da=c.winner_correct.astype(float).to_numpy()-b.winner_correct.astype(float).to_numpy()
    db=(c.win_prob-c.actual_win).pow(2).to_numpy()-(b.win_prob-b.actual_win).pow(2).to_numpy()
    rng=np.random.default_rng(seed); idx=rng.integers(0,n,size=(n_boot,n)); aa=da[idx].mean(1); bb=db[idx].mean(1)
    return pd.DataFrame([
        {"Metric":"Winner accuracy","Delta":float(da.mean()),"95% CI low":float(np.quantile(aa,.025)),"95% CI high":float(np.quantile(aa,.975)),"Clear improvement":bool(np.quantile(aa,.025)>0)},
        {"Metric":"Brier","Delta":float(db.mean()),"95% CI low":float(np.quantile(bb,.025)),"95% CI high":float(np.quantile(bb,.975)),"Clear improvement":bool(np.quantile(bb,.975)<0)}])

import streamlit as st
import pandas as pd
import numpy as np



# ---------- v1.9 2026 Live Holdout Audit ----------
def v19_2026_holdout(games, start_week=1, end_week=6):
    """Replay 2026 weeks 1-end_week using only games completed before each kickoff.

    Unlike the long-run research backtests, early-season games are intentionally retained.
    Their rolling features contain only what was actually knowable at the time (with the
    model's preseason/default priors where no current-season games had yet been played).
    Training remains historical and leakage-safe.
    """
    f=efficiency_pregame_features(games)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    te_all=done[(done.season==2026)&(done.week>=start_week)&(done.week<=end_week)].copy()
    rows=[]
    for (season,week),te in te_all.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min()
        tr=done[(done.gameday<cutoff)&(done.home_games_prior>=2)&(done.away_games_prior>=2)].copy()
        if len(tr)<300: continue
        ep=_linear_classifier(); em=pipe(25); et=pipe(30)
        ep.fit(tr[EPA_CORE_FEATURES],(tr.target_margin>0).astype(int))
        em.fit(tr[FEATURES],tr.target_margin); et.fit(tr[PASS_FEATURES],tr.target_total)
        wp=ep.predict_proba(te[EPA_CORE_FEATURES])[:,1]
        pm=em.predict(te[FEATURES]); pt=et.predict(te[PASS_FEATURES])
        for i,(_,r) in enumerate(te.iterrows()):
            actual_home=float(r.target_margin>0)
            spread=r.get("spread_line",np.nan); total_line=r.get("total_line",np.nan)
            model_side="PASS"; ats_correct=np.nan; market_fav_correct=np.nan
            if pd.notna(spread):
                gap=float(pm[i]-float(spread))
                if abs(gap)>1e-9: model_side="HOME" if gap>0 else "AWAY"
                cover=float(r.target_margin-float(spread))
                if model_side!="PASS" and abs(cover)>1e-9:
                    ats_correct=bool(cover>0) if model_side=="HOME" else bool(cover<0)
                if abs(float(spread))>1e-9:
                    market_home_fav=float(spread)>0
                    market_fav_correct=bool(actual_home) if market_home_fav else bool(1-actual_home)
            rows.append({
                "season":2026,"week":int(week),"gameday":r.gameday,
                "Game":f"{r.away_team} @ {r.home_team}",
                "home_win_prob":float(wp[i]),"pred_margin":float(pm[i]),"pred_total":float(pt[i]),
                "actual_margin":float(r.target_margin),"actual_total":float(r.target_total),
                "winner_correct":bool((wp[i]>=.5)==bool(actual_home)),
                "brier":float((wp[i]-actual_home)**2),
                "margin_abs_error":float(abs(pm[i]-r.target_margin)),
                "total_abs_error":float(abs(pt[i]-r.target_total)),
                "confidence":float(max(wp[i],1-wp[i])),
                "market_spread":spread,"market_total":total_line,
                "model_side":model_side,"ats_correct":ats_correct,"market_favorite_correct":market_fav_correct,
                "training_games":int(len(tr))})
    return pd.DataFrame(rows)

def v19_holdout_summary(x):
    if x.empty:return pd.DataFrame()
    return pd.DataFrame([{
        "Games":len(x),"Winner Accuracy":x.winner_correct.mean(),"Brier":x.brier.mean(),
        "Margin MAE":x.margin_abs_error.mean(),"Total MAE":x.total_abs_error.mean(),
        "ATS Accuracy":x.ats_correct.dropna().mean() if x.ats_correct.notna().any() else np.nan,
        "Market Favorite Accuracy":x.market_favorite_correct.dropna().mean() if x.market_favorite_correct.notna().any() else np.nan}])

def v19_weekly(x):
    if x.empty:return pd.DataFrame()
    return x.groupby("week").agg(Games=("Game","size"),Winner_Accuracy=("winner_correct","mean"),
        Brier=("brier","mean"),Margin_MAE=("margin_abs_error","mean"),Total_MAE=("total_abs_error","mean")).reset_index()

def v19_confidence(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["Confidence band"]=pd.cut(y.confidence,bins=[.5,.55,.60,.65,.70,1.00001],
        labels=["50–55%","55–60%","60–65%","65–70%","70%+"],include_lowest=True,right=False)
    return y.groupby("Confidence band",observed=True).agg(Games=("Game","size"),Average_Confidence=("confidence","mean"),Winner_Accuracy=("winner_correct","mean")).reset_index()


# ---------- v2.0 Holdout Diagnostic ----------
def v20_holdout_diagnostics(x):
    """Diagnose where the frozen 2026 holdout engine is strong/weak without refitting it."""
    if x.empty:
        return {}
    y=x.copy()
    y["phase"]=np.where(y.week<=2,"Cold start (Weeks 1–2)","Informed (Weeks 3–6)")
    y["pick"]=np.where(y.home_win_prob>=.5,"HOME","AWAY")
    y["picked_home"]=(y.home_win_prob>=.5)
    y["projected_margin_strength"]=abs(y.pred_margin)
    y["market_gap"]=np.where(y.market_spread.notna(),abs(y.pred_margin-y.market_spread),np.nan)
    y["market_agreement"]=np.where(y.market_spread.notna(),
        np.where(((y.pred_margin>=0)&(y.market_spread>=0))|((y.pred_margin<0)&(y.market_spread<0)),"Agree","Disagree"),"No line")
    y["favorite_pick"]=np.where(y.market_spread.notna(),
        np.where(((y.pick=="HOME")&(y.market_spread>0))|((y.pick=="AWAY")&(y.market_spread<0)),"Market favorite","Market underdog"),"No line")
    y["margin_error_signed"]=y.pred_margin-y.actual_margin
    y["total_error_signed"]=y.pred_total-y.actual_total

    def group(col):
        return y.groupby(col,observed=True).agg(
            Games=("Game","size"), Winner_Accuracy=("winner_correct","mean"),
            Brier=("brier","mean"), Margin_MAE=("margin_abs_error","mean"),
            Total_MAE=("total_abs_error","mean")).reset_index()

    phase=group("phase")
    pick=group("pick")
    fav=group("favorite_pick")
    agree=group("market_agreement")

    y["Projected margin band"]=pd.cut(y.projected_margin_strength,[-.001,3,7,14,np.inf],
        labels=["0–3","3–7","7–14","14+"],include_lowest=True,right=False)
    margin_band=group("Projected margin band")
    y["Market gap band"]=pd.cut(y.market_gap,[-.001,2,4,7,np.inf],
        labels=["0–2","2–4","4–7","7+"],include_lowest=True,right=False)
    market_gap=group("Market gap band") if y.market_gap.notna().any() else pd.DataFrame()

    misses=y.sort_values(["brier","margin_abs_error"],ascending=False).head(10).copy()
    hits=y.sort_values(["winner_correct","brier","margin_abs_error"],ascending=[False,True,True]).head(10).copy()
    return {"games":y,"phase":phase,"pick":pick,"favorite":fav,"agreement":agree,
            "margin_band":margin_band,"market_gap":market_gap,"misses":misses,"hits":hits}

def v20_fmt_table(df):
    z=df.copy()
    if "Winner_Accuracy" in z: z["Winner Accuracy"]=z.pop("Winner_Accuracy").map(lambda q:f"{q:.1%}")
    for c in ["Brier","Margin_MAE","Total_MAE"]:
        if c in z: z[c]=z[c].map(lambda q:round(float(q),3))
    return z

# ---------- v1.8 Football Intelligence: Opponent Adjustment ----------
# Tests whether EPA becomes more informative after adjusting each historical
# performance for the quality of the opponent known BEFORE that game.
# No current/future opponent results are used to construct these ratings.
OPP_ADJ_FEATURES=[
    "home_opp_adj_off_epa","away_opp_adj_off_epa",
    "home_opp_adj_def_epa","away_opp_adj_def_epa",
    "opp_adj_off_diff","opp_adj_def_diff"
]
OPP_ADJ_PROB_FEATURES=EPA_CORE_FEATURES+OPP_ADJ_FEATURES

def opponent_adjusted_pregame_features(games,window=6):
    base=efficiency_pregame_features(games).copy()
    # Reuse game-level PBP efficiency to update ratings only AFTER each game row
    off_parts=[]; def_parts=[]
    for season in sorted(base.season.dropna().astype(int).unique()):
        try:
            o,d=load_efficiency_games(season); off_parts.append(o); def_parts.append(d)
        except Exception:
            continue
    if not off_parts:
        for c in OPP_ADJ_FEATURES: base[c]=0.0
        return base
    off=pd.concat(off_parts,ignore_index=True); de=pd.concat(def_parts,ignore_index=True)
    raw_off={}; raw_def={}; adj_off={}; adj_def={}; rows=[]
    def rav(store,team):
        v=store.get(team,[]); return float(np.mean(v[-window:])) if v else 0.0
    for _,g in base.sort_values(["gameday","gametime"],na_position="last").iterrows():
        h,a=g.home_team,g.away_team; r=g.to_dict()
        r["home_opp_adj_off_epa"]=rav(adj_off,h); r["away_opp_adj_off_epa"]=rav(adj_off,a)
        r["home_opp_adj_def_epa"]=rav(adj_def,h); r["away_opp_adj_def_epa"]=rav(adj_def,a)
        r["opp_adj_off_diff"]=r["home_opp_adj_off_epa"]-r["away_opp_adj_off_epa"]
        # Lower defensive EPA allowed is better, so away-home gives positive home defensive edge.
        r["opp_adj_def_diff"]=r["away_opp_adj_def_epa"]-r["home_opp_adj_def_epa"]
        rows.append(r)
        gid=g.get("game_id")
        if pd.isna(gid): continue
        oo=off[off.game_id==gid]; dd=de[de.game_id==gid]
        # Opponent baselines below are strictly pregame ratings.
        pre_off_h,pre_off_a=rav(raw_off,h),rav(raw_off,a)
        pre_def_h,pre_def_a=rav(raw_def,h),rav(raw_def,a)
        vals={}
        for team,opp,opp_def,opp_off in [(h,a,pre_def_a,pre_off_a),(a,h,pre_def_h,pre_off_h)]:
            xo=oo[oo.posteam==team]; xd=dd[dd.defteam==team]
            if xo.empty or xd.empty: continue
            oe=float(xo.iloc[0].off_epa); da=float(xd.iloc[0].def_epa_allowed)
            if not np.isfinite(oe) or not np.isfinite(da): continue
            # Offense gets credit for beating a strong (low EPA allowed) defense.
            # Defense gets credit for holding a strong (high EPA) offense below expectation.
            adj_o=oe-opp_def
            adj_d=da-opp_off
            vals[team]=(oe,da,adj_o,adj_d)
        for team,(oe,da,ao,ad) in vals.items():
            raw_off.setdefault(team,[]).append(oe); raw_def.setdefault(team,[]).append(da)
            adj_off.setdefault(team,[]).append(ao); adj_def.setdefault(team,[]).append(ad)
    return pd.DataFrame(rows)

def v18_opponent_backtest(games,first_test_season=2022,last_test_season=2024):
    f=opponent_adjusted_pregame_features(games)
    done=f.dropna(subset=["target_margin","target_total"]).sort_values(["gameday","gametime"],na_position="last")
    test=done[(done.season>=first_test_season)&(done.season<=last_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    for (season,week),te in test.groupby(["season","week"],sort=True):
        cutoff=te.gameday.min()
        tr=done[(done.gameday<cutoff)&(done.season>=first_test_season)&(done.season<=last_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(tr)<250: continue
        actual=(te.target_margin>0).astype(float).to_numpy()
        for name,features in [("Game Engine 1.0",EPA_CORE_FEATURES),("Opponent-Adjusted Challenger",OPP_ADJ_PROB_FEATURES)]:
            wm=_linear_classifier(); wm.fit(tr[features],(tr.target_margin>0).astype(int)); wp=wm.predict_proba(te[features])[:,1]
            for i,(_,r) in enumerate(te.iterrows()):
                rows.append({"model":name,"season":int(season),"week":int(week),"actual_win":actual[i],"win_prob":float(wp[i]),"winner_correct":bool((wp[i]>=.5)==bool(actual[i]))})
    return pd.DataFrame(rows)

def v18_opponent_summary(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["brier"]=(y.win_prob-y.actual_win)**2
    return y.groupby("model").agg(Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean")).reset_index()

def v18_opponent_yearly(x):
    if x.empty:return pd.DataFrame()
    y=x.copy(); y["brier"]=(y.win_prob-y.actual_win)**2
    return y.groupby(["season","model"]).agg(Games=("winner_correct","size"),Winner_Accuracy=("winner_correct","mean"),Brier=("brier","mean")).reset_index()

def v18_opponent_bootstrap(x,n_boot=4000,seed=81):
    if x.empty:return pd.DataFrame()
    b=x[x.model=="Game Engine 1.0"].reset_index(drop=True); c=x[x.model=="Opponent-Adjusted Challenger"].reset_index(drop=True); n=min(len(b),len(c))
    if n==0:return pd.DataFrame()
    b=b.iloc[:n]; c=c.iloc[:n]
    da=c.winner_correct.astype(float).to_numpy()-b.winner_correct.astype(float).to_numpy()
    db=(c.win_prob-c.actual_win).pow(2).to_numpy()-(b.win_prob-b.actual_win).pow(2).to_numpy()
    rng=np.random.default_rng(seed); idx=rng.integers(0,n,size=(n_boot,n)); aa=da[idx].mean(1); bb=db[idx].mean(1)
    return pd.DataFrame([
        {"Metric":"Winner accuracy","Delta":float(da.mean()),"95% CI low":float(np.quantile(aa,.025)),"95% CI high":float(np.quantile(aa,.975)),"Clear improvement":bool(np.quantile(aa,.025)>0)},
        {"Metric":"Brier","Delta":float(db.mean()),"95% CI low":float(np.quantile(bb,.025)),"95% CI high":float(np.quantile(bb,.975)),"Clear improvement":bool(np.quantile(bb,.975)<0)}])

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
st.markdown('<div class="small">v2.1 • Player Projection Championship • Game Engine 1.0 frozen</div>',unsafe_allow_html=True)

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

    st.divider()
    st.subheader("🧠 v1.4 Football Intelligence — QB Quality")
    st.write("First challenger layer: recent quarterback-room quality from completed prior weeks only.")
    st.caption("Leakage guard: week W never uses week W QB production. This stage tests QB quality only; same-week starter availability/injury status is intentionally NOT inferred until a true pregame injury/depth-chart source is added.")
    if st.button("Run v1.4 QB Quality Challenge",type="primary"):
        with st.spinner("Loading historical QB production and running paired walk-forward validation..."):
            ps=players()
            qb=v14_qb_quality_backtest(g,ps,2022)
        if qb.empty:
            st.error("QB Quality validation could not be produced. Game Engine 1.0 remains unchanged.")
        else:
            qs=v14_qb_summary(qb); qshow=qs.copy()
            qshow["Winner Accuracy"]=qshow.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(qshow[["model","Games","Winner Accuracy","Brier"]].round(4),use_container_width=True,hide_index=True)
            st.subheader("Year-by-year QB challenger stability")
            qy=v14_qb_yearly(qb); qyshow=qy.copy(); qyshow["Winner Accuracy"]=qyshow.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(qyshow[["season","model","Games","Winner Accuracy","Brier"]].round(4),use_container_width=True,hide_index=True)
            st.subheader("Paired bootstrap vs Game Engine 1.0")
            qboot=v14_qb_bootstrap(qb)
            st.dataframe(qboot.round(4),use_container_width=True,hide_index=True)
            base=qs[qs.model=="Game Engine 1.0"].iloc[0]; chal=qs[qs.model=="QB Quality Challenger"].iloc[0]
            st.markdown("### QB layer delta")
            st.write(f"Winner accuracy: **{(chal.Winner_Accuracy-base.Winner_Accuracy):+.2%}**")
            st.write(f"Brier: **{(chal.Brier-base.Brier):+.4f}**")
            clear=qboot[qboot["Clear improvement"]]
            if chal.Brier < base.Brier and chal.Winner_Accuracy >= base.Winner_Accuracy:
                if len(clear): st.success("QB QUALITY GATE: STRONG PASS — aggregate profile improves and at least one paired metric clears the 95% bootstrap bar.")
                else: st.info("QB QUALITY GATE: PROMISING — aggregate profile improves, but statistical certainty is not yet strong enough for automatic promotion.")
            else:
                st.warning("QB QUALITY GATE: HOLD — Game Engine 1.0 stays protected. We learn from the result without forcing the feature into production.")
    st.info("Research queue after QB quality: true QB availability/injuries → offensive line → opponent adjustment → offseason carryover → travel/rest → weather. Each must earn promotion independently.")

    st.divider()
    st.subheader("🚑 v1.5 Football Intelligence — QB Availability")
    st.write("Tests whether official pregame QB1 depth-chart + injury/practice status adds signal beyond Game Engine 1.0.")
    st.caption("Validation is intentionally restricted to 2022–2024, where nflverse supplies week-aligned historical depth charts and official injury reports. No actual-game starter or postgame QB production is used to infer availability.")
    if st.button("Run v1.5 QB Availability Challenge",type="primary"):
        with st.spinner("Loading official historical depth charts/injury reports and running paired walk-forward validation..."):
            av=load_qb_availability_history(2022,2024)
            qa=v15_qb_availability_backtest(g,av,2022,2024)
        if av.empty:
            st.error("Historical availability feeds could not be loaded. Game Engine 1.0 remains unchanged.")
        elif qa.empty:
            st.error("QB availability validation could not be produced. Game Engine 1.0 remains unchanged.")
        else:
            st.caption(f"Official QB1 availability rows loaded: {len(av):,}")
            sm=v15_qb_availability_summary(qa); show=sm.copy(); show["Winner Accuracy"]=show.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(show[["model","Games","Winner Accuracy","Brier"]].round(4),use_container_width=True,hide_index=True)
            st.subheader("Year-by-year availability stability")
            yy=v15_qb_availability_yearly(qa); ys=yy.copy(); ys["Winner Accuracy"]=ys.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(ys[["season","model","Games","Winner Accuracy","Brier"]].round(4),use_container_width=True,hide_index=True)
            st.subheader("Paired bootstrap vs Game Engine 1.0")
            bt=v15_qb_availability_bootstrap(qa); st.dataframe(bt.round(4),use_container_width=True,hide_index=True)
            base=sm[sm.model=="Game Engine 1.0"].iloc[0]; chal=sm[sm.model=="QB Availability Challenger"].iloc[0]
            st.markdown("### Availability layer delta")
            st.write(f"Winner accuracy: **{(chal.Winner_Accuracy-base.Winner_Accuracy):+.2%}**")
            st.write(f"Brier: **{(chal.Brier-base.Brier):+.4f}**")
            clear=bt[bt["Clear improvement"]]
            if chal.Brier < base.Brier and chal.Winner_Accuracy >= base.Winner_Accuracy:
                if len(clear): st.success("QB AVAILABILITY GATE: STRONG PASS — official pregame availability improves the profile and clears a paired 95% bootstrap bar.")
                else: st.info("QB AVAILABILITY GATE: PROMISING — aggregate profile improves, but certainty is not strong enough for automatic production promotion.")
            else:
                st.warning("QB AVAILABILITY GATE: HOLD — Game Engine 1.0 remains protected.")
    st.info("v1.5 research rule: availability must be knowable before kickoff. Historical actual starters are never reverse-engineered from postgame stats.")

    st.divider()
    st.subheader("🧱 v1.6 Football Intelligence — Offensive Line Availability")
    st.write("Tests whether official pregame offensive-line injury/practice burden adds independent signal beyond Game Engine 1.0.")
    st.caption("This challenger uses weekly official injury/practice reports only. It does not infer starters or snaps from postgame participation. Validation stays in 2022–2024 so it is directly comparable with the v1.5 availability era.")
    if st.button("Run v1.6 Offensive Line Challenge",type="primary"):
        with st.spinner("Loading official OL injury reports and running paired walk-forward validation..."):
            ol=load_ol_availability_history(2022,2024); ox=v16_ol_backtest(g,ol,2022,2024)
        if ol.empty: st.error("Historical OL availability feed could not be loaded. Game Engine 1.0 remains unchanged.")
        elif ox.empty: st.error("OL validation could not be produced. Game Engine 1.0 remains unchanged.")
        else:
            st.caption(f"Official team-week OL availability rows loaded: {len(ol):,}")
            sm=v16_ol_summary(ox); show=sm.copy(); show["Winner Accuracy"]=show.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(show[["model","Games","Winner Accuracy","Brier"]].round(4),use_container_width=True,hide_index=True)
            st.subheader("Year-by-year OL stability")
            yy=v16_ol_yearly(ox); ys=yy.copy(); ys["Winner Accuracy"]=ys.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(ys[["season","model","Games","Winner Accuracy","Brier"]].round(4),use_container_width=True,hide_index=True)
            st.subheader("Paired bootstrap vs Game Engine 1.0")
            bt=v16_ol_bootstrap(ox); st.dataframe(bt.round(4),use_container_width=True,hide_index=True)
            base=sm[sm.model=="Game Engine 1.0"].iloc[0]; chal=sm[sm.model=="OL Availability Challenger"].iloc[0]
            st.markdown("### Offensive-line layer delta")
            st.write(f"Winner accuracy: **{(chal.Winner_Accuracy-base.Winner_Accuracy):+.2%}**"); st.write(f"Brier: **{(chal.Brier-base.Brier):+.4f}**")
            clear=bt[bt["Clear improvement"]]
            if chal.Brier < base.Brier and chal.Winner_Accuracy >= base.Winner_Accuracy:
                if len(clear): st.success("OL AVAILABILITY GATE: STRONG PASS — the pregame OL layer improves the profile and clears a paired 95% bootstrap bar.")
                else: st.info("OL AVAILABILITY GATE: PROMISING — aggregate profile improves, but certainty is not strong enough for automatic production promotion.")
            else: st.warning("OL AVAILABILITY GATE: HOLD — Game Engine 1.0 remains protected.")
    st.info("Research queue after v1.6: skill-position injuries → opponent adjustment → rest/travel → weather. QB Availability remains a promising shadow candidate, not a production feature.")


    st.divider()
    st.subheader("🎯 v1.7 Football Intelligence — Skill-Position Availability")
    st.write("Tests whether official pregame WR/RB/TE injury and practice burden adds independent signal beyond Game Engine 1.0.")
    st.caption("Uses official weekly injury/practice reports only; no postgame snaps, starter inference, or same-game production. Validation remains 2022–2024 for an apples-to-apples comparison with v1.5 and v1.6.")
    if st.button("Run v1.7 Skill-Position Challenge",type="primary"):
        with st.spinner("Loading official WR/RB/TE injury reports and running paired walk-forward validation..."):
            sk=load_skill_availability_history(2022,2024); sx=v17_skill_backtest(g,sk,2022,2024)
        if sk.empty: st.error("Historical skill-position availability feed could not be loaded. Game Engine 1.0 remains unchanged.")
        elif sx.empty: st.error("Skill-position validation could not be produced. Game Engine 1.0 remains unchanged.")
        else:
            st.caption(f"Official team-week skill-position availability rows loaded: {len(sk):,}")
            sm=v17_skill_summary(sx); show=sm.copy(); show["Winner Accuracy"]=show.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(show[["model","Games","Winner Accuracy","Brier"]].round(4),use_container_width=True,hide_index=True)
            st.subheader("Year-by-year skill-position stability")
            yy=v17_skill_yearly(sx); ys=yy.copy(); ys["Winner Accuracy"]=ys.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(ys[["season","model","Games","Winner Accuracy","Brier"]].round(4),use_container_width=True,hide_index=True)
            st.subheader("Paired bootstrap vs Game Engine 1.0")
            bt=v17_skill_bootstrap(sx); st.dataframe(bt.round(4),use_container_width=True,hide_index=True)
            base=sm[sm.model=="Game Engine 1.0"].iloc[0]; chal=sm[sm.model=="Skill Availability Challenger"].iloc[0]
            st.markdown("### Skill-position layer delta")
            st.write(f"Winner accuracy: **{(chal.Winner_Accuracy-base.Winner_Accuracy):+.2%}**"); st.write(f"Brier: **{(chal.Brier-base.Brier):+.4f}**")
            clear=bt[bt["Clear improvement"]]
            if chal.Brier < base.Brier and chal.Winner_Accuracy >= base.Winner_Accuracy:
                if len(clear): st.success("SKILL AVAILABILITY GATE: STRONG PASS — the pregame skill-position layer improves the profile and clears a paired 95% bootstrap bar.")
                else: st.info("SKILL AVAILABILITY GATE: PROMISING — aggregate profile improves, but certainty is not strong enough for automatic production promotion.")
            else: st.warning("SKILL AVAILABILITY GATE: HOLD — Game Engine 1.0 remains protected.")
    st.info("Research queue after v1.7: opponent adjustment → rest/travel → weather. QB Availability remains the leading promising shadow candidate; OL Availability remains HOLD.")


    st.divider()
    st.subheader("🧭 v1.8 Football Intelligence — Opponent Adjustment")
    st.write("Tests whether EPA performance becomes more predictive after accounting for the quality of opponents already faced.")
    st.caption("Leakage guard: every opponent-strength adjustment is built only from games completed before the game being predicted. Current-game and future results never enter the rating. Validation remains 2022–2024 for direct comparison with v1.5–v1.7.")
    if st.button("Run v1.8 Opponent Adjustment Challenge",type="primary"):
        with st.spinner("Building leakage-safe opponent-adjusted EPA ratings and running paired walk-forward validation..."):
            ax=v18_opponent_backtest(g,2022,2024)
        if ax.empty: st.error("Opponent-adjusted validation could not be produced. Game Engine 1.0 remains unchanged.")
        else:
            sm=v18_opponent_summary(ax); show=sm.copy(); show["Winner Accuracy"]=show.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(show[["model","Games","Winner Accuracy","Brier"]].round(4),use_container_width=True,hide_index=True)
            st.subheader("Year-by-year opponent-adjustment stability")
            yy=v18_opponent_yearly(ax); ys=yy.copy(); ys["Winner Accuracy"]=ys.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(ys[["season","model","Games","Winner Accuracy","Brier"]].round(4),use_container_width=True,hide_index=True)
            st.subheader("Paired bootstrap vs Game Engine 1.0")
            bt=v18_opponent_bootstrap(ax); st.dataframe(bt.round(4),use_container_width=True,hide_index=True)
            base=sm[sm.model=="Game Engine 1.0"].iloc[0]; chal=sm[sm.model=="Opponent-Adjusted Challenger"].iloc[0]
            st.markdown("### Opponent-adjustment layer delta")
            st.write(f"Winner accuracy: **{(chal.Winner_Accuracy-base.Winner_Accuracy):+.2%}**"); st.write(f"Brier: **{(chal.Brier-base.Brier):+.4f}**")
            clear=bt[bt["Clear improvement"]]
            if chal.Brier < base.Brier and chal.Winner_Accuracy >= base.Winner_Accuracy:
                if len(clear): st.success("OPPONENT ADJUSTMENT GATE: STRONG PASS — opponent-adjusted EPA improves the profile and clears a paired 95% bootstrap bar.")
                else: st.info("OPPONENT ADJUSTMENT GATE: PROMISING — aggregate profile improves, but certainty is not strong enough for automatic production promotion.")
            else: st.warning("OPPONENT ADJUSTMENT GATE: HOLD — Game Engine 1.0 remains protected.")
    st.info("Research queue after v1.8: rest/travel → weather. QB Availability remains the leading promising shadow candidate; OL and skill-position availability remain HOLD.")

    st.divider()
    st.subheader("🧪 v1.9 — 2026 Live Holdout Audit (Weeks 1–6)")
    st.write("Replays the first six completed weeks of 2026 exactly as a frozen Game Engine 1.0 would have seen them: each game is predicted using only information available before that kickoff.")
    st.caption("This is a true chronological holdout audit, not a refit on Weeks 1–6. Early-season games are intentionally included, so Week 1 starts from historical/preseason priors and later weeks accumulate only prior 2026 results.")
    if st.button("Run v1.9 2026 Weeks 1–6 Holdout Audit",type="primary"):
        with st.spinner("Replaying 2026 Weeks 1–6 chronologically with Game Engine 1.0 frozen..."):
            hx=v19_2026_holdout(g,1,6)
        if hx.empty:
            st.error("The 2026 Weeks 1–6 holdout could not be produced from the current schedule/results feed. Game Engine 1.0 remains unchanged.")
        else:
            sm=v19_holdout_summary(hx); show=sm.copy()
            for c in ["Winner Accuracy","ATS Accuracy","Market Favorite Accuracy"]:
                show[c]=show[c].map(lambda x:f"{x:.1%}" if pd.notna(x) else "—")
            st.markdown("### If NFL Edge had been live from Week 1")
            st.dataframe(show.round(3),use_container_width=True,hide_index=True)
            st.subheader("Week-by-week record")
            wk=v19_weekly(hx); ws=wk.copy(); ws["Winner Accuracy"]=ws.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(ws.round(3),use_container_width=True,hide_index=True)
            st.subheader("Confidence check")
            cf=v19_confidence(hx); cs=cf.copy(); cs["Average Confidence"]=cs.Average_Confidence.map(lambda x:f"{x:.1%}"); cs["Winner Accuracy"]=cs.Winner_Accuracy.map(lambda x:f"{x:.1%}")
            st.dataframe(cs[["Confidence band","Games","Average Confidence","Winner Accuracy"]],use_container_width=True,hide_index=True)
            st.subheader("Game-by-game audit trail")
            audit=hx[["week","gameday","Game","home_win_prob","pred_margin","actual_margin","pred_total","actual_total","winner_correct"]].copy()
            audit["home_win_prob"]=audit.home_win_prob.map(lambda x:f"{x:.1%}")
            st.dataframe(audit.round(1),use_container_width=True,hide_index=True)
            acc=float(hx.winner_correct.mean()); br=float(hx.brier.mean())
            if acc>=.64:
                st.success(f"2026 HOLDOUT: {acc:.1%} winner accuracy across {len(hx)} games. This supports the historical signal, but six weeks is still a small sample and not a guarantee of future performance.")
            elif acc>=.58:
                st.info(f"2026 HOLDOUT: {acc:.1%} winner accuracy across {len(hx)} games. Useful signal, but weaker than the long-run historical profile; keep monitoring before increasing trust.")
            else:
                st.warning(f"2026 HOLDOUT: {acc:.1%} winner accuracy across {len(hx)} games. Investigate early-season priors/calibration before treating the engine as production-trustworthy.")
            st.caption(f"Holdout Brier: {br:.4f}. No Weeks 1–6 outcome is allowed into its own prediction features or training set.")
    st.info("v1.9 changes the roadmap: validate the current engine on fresh 2026 games before adding more Football Intelligence layers. Rest/travel and weather remain queued until this audit is understood.")


    st.divider()
    st.subheader("🔎 v2.0 — 2026 Holdout Diagnostic")
    st.write("Explains *where* the frozen Game Engine 1.0 succeeded and failed across the same 93-game Weeks 1–6 holdout. This diagnostic does not refit the model or change any predictions.")
    st.caption("Primary question: is the engine materially stronger after it has current-season information, and do confidence/margin/market-disagreement signals behave consistently enough for production labels?")
    if st.button("Run v2.0 Holdout Diagnostic",type="primary"):
        with st.spinner("Diagnosing the frozen 2026 holdout..."):
            dx=v19_2026_holdout(g,1,6); d=v20_holdout_diagnostics(dx)
        if not d:
            st.error("The v2.0 diagnostic could not reproduce the v1.9 holdout. Game Engine 1.0 remains unchanged.")
        else:
            st.markdown("### Cold start vs informed engine")
            st.dataframe(v20_fmt_table(d["phase"]),use_container_width=True,hide_index=True)
            cold=d["phase"].loc[d["phase"].phase.str.startswith("Cold")].iloc[0]
            inf=d["phase"].loc[d["phase"].phase.str.startswith("Informed")].iloc[0]
            delta=float(inf.Winner_Accuracy-cold.Winner_Accuracy)
            st.metric("Weeks 3–6 accuracy lift vs Weeks 1–2",f"{delta:+.1%}")

            st.markdown("### Where winner accuracy comes from")
            st.caption("HOME/AWAY is the model's selected winner, not venue performance by itself.")
            st.dataframe(v20_fmt_table(d["pick"]),use_container_width=True,hide_index=True)
            st.markdown("#### Market favorite vs underdog selections")
            st.dataframe(v20_fmt_table(d["favorite"]),use_container_width=True,hide_index=True)
            st.markdown("#### Model agrees vs disagrees with market favorite")
            st.dataframe(v20_fmt_table(d["agreement"]),use_container_width=True,hide_index=True)

            st.markdown("### Projection-strength diagnostic")
            st.dataframe(v20_fmt_table(d["margin_band"]),use_container_width=True,hide_index=True)
            if not d["market_gap"].empty:
                st.markdown("#### Model vs market spread-gap diagnostic")
                st.dataframe(v20_fmt_table(d["market_gap"]),use_container_width=True,hide_index=True)

            cols=["week","Game","home_win_prob","pred_margin","actual_margin","market_spread","winner_correct","brier","margin_abs_error"]
            st.markdown("### Ten biggest probability/margin misses")
            mm=d["misses"][cols].copy(); mm["home_win_prob"]=mm.home_win_prob.map(lambda q:f"{q:.1%}")
            st.dataframe(mm.round(2),use_container_width=True,hide_index=True)
            st.markdown("### Ten cleanest correct calls")
            hh=d["hits"][cols].copy(); hh["home_win_prob"]=hh.home_win_prob.map(lambda q:f"{q:.1%}")
            st.dataframe(hh.round(2),use_container_width=True,hide_index=True)

            if delta>=.08:
                st.info("v2.0 DIAGNOSTIC: strong early evidence of a cold-start problem. Weeks 3–6 materially outperform Weeks 1–2. Treat early-season confidence as preliminary until this pattern is confirmed on more seasons.")
            elif delta>=.03:
                st.info("v2.0 DIAGNOSTIC: moderate evidence that current-season information improves the engine after the opening weeks.")
            else:
                st.warning("v2.0 DIAGNOSTIC: the Weeks 3–6 improvement is small. Do not attribute the holdout result mainly to cold start without more evidence.")
            st.caption("This is descriptive holdout analysis, not a new model-selection test. Subgroups are small and should not be converted into betting rules without independent validation.")
    st.info("After v2.0: decide whether to freeze Game Engine 1.0 for the live-product phase, while keeping QB Availability as a shadow candidate. Rest/travel and weather can remain queued rather than blocking product development.")


    st.divider()
    st.subheader("🎯 v2.1 — Player Projection Championship")
    st.write("Audits the player engine market-by-market before NFL Edge is allowed to treat any prop category as a betting edge. Projections use completed prior weeks only; current-week production never enters its own projection.")
    st.caption("This is projection validation, not sportsbook profitability validation. STRONG/USABLE means the stat projection has predictive structure; it does not yet mean a sportsbook line is beatable.")
    if st.button("Run v2.1 Player Projection Championship",type="primary"):
        with st.spinner("Backtesting QB/RB/WR/TE player markets chronologically..."):
            ps=players(); mk,pos,yr=v21_player_championship(ps,2022,2026,4); td=v21_td_audit(ps,2022,2026,4)
        if mk.empty:
            st.error("Player championship could not be produced from the current nflverse player feed. No production labels were changed.")
        else:
            show=mk.copy(); show["Correlation"]=show["Correlation"].map(lambda q:f"{q:.3f}" if pd.notna(q) else "—"); show["Normalized MAE"]=show["Normalized MAE"].map(lambda q:f"{q:.1%}" if pd.notna(q) else "—")
            st.markdown("### Market reliability board")
            st.dataframe(show.round(2),use_container_width=True,hide_index=True)
            st.markdown("### Fantasy projection error by position")
            st.dataframe(pos.round(2),use_container_width=True,hide_index=True)
            st.markdown("### Year-by-year player-engine stability")
            st.dataframe(yr.round(2),use_container_width=True,hide_index=True)
            st.markdown("### TD rare-event audit")
            st.caption("Touchdowns are separated because a low MAE on a rare event can be misleading. They remain research-only until probability calibration is validated.")
            st.dataframe(td.round(3),use_container_width=True,hide_index=True)
            strong=mk[mk["Research tier"]=="STRONG"]["Market"].tolist(); usable=mk[mk["Research tier"]=="USABLE"]["Market"].tolist()
            if strong:
                st.success("PLAYER ENGINE GATE: at least one market shows strong projection structure. Next step is sportsbook-line/price backtesting for: "+", ".join(strong)+".")
            elif usable:
                st.info("PLAYER ENGINE GATE: usable projection structure exists, but no market earns STRONG yet. Next step is targeted model upgrades before claiming betting value.")
            else:
                st.warning("PLAYER ENGINE GATE: HOLD. The current opportunity model is not strong enough to support prop-edge claims. Upgrade the player engine before odds integration.")
            st.caption("NFL Edge will not label a prop as +EV until historical lines/odds and forward frozen picks demonstrate value after vig.")


st.divider()
st.caption("NFL Edge v2.1 • Player Projection Championship • Game Engine 1.0 frozen • No prop is called +EV until price-aware validation proves it.")
