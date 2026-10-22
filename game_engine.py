"""NFL Edge Game Engine 1.0.

Extracted without model changes from the protected nfl_edge.py implementation so
new live-test and production surfaces can reuse the validated game model without
modifying the original application.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEAM_ALIASES={"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}
FEATURES=["elo_diff","pf_matchup","form_diff","winform_diff","home_pf5","away_pf5","home_pa5","away_pa5","home_games_prior","away_games_prior","rest_diff","home_rest","away_rest"]

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
    def team(t): return state.setdefault(t,{"games":0,"pf":[],"pa":[],"margin":[],"wins":[],"elo":1500.,"last_date":None})
    def avg(v,n,d): return float(np.mean(v[-n:])) if v else d
    for _,g in df.iterrows():
        season=int(g.season) if pd.notna(g.get("season",np.nan)) else None
        if last_season is not None and season is not None and season!=last_season:
            for s in state.values():
                s["elo"]=1500.+0.67*(s["elo"]-1500.); s["games"]=0; s["pf"]=[]; s["pa"]=[]; s["margin"]=[]; s["wins"]=[]; s["last_date"]=None
        last_season=season if season is not None else last_season
        h,a=team(g.home_team),team(g.away_team); gd=pd.to_datetime(g.get("gameday"),errors="coerce")
        def rest_days(s):
            if s["last_date"] is None or pd.isna(gd): return 7.0
            return float(np.clip((gd-s["last_date"]).days,3,21))
        hr,ar=rest_days(h),rest_days(a); r=g.to_dict()
        r.update({"home_elo":h["elo"],"away_elo":a["elo"],"elo_diff":h["elo"]-a["elo"],"home_pf5":avg(h["pf"],5,22),"away_pf5":avg(a["pf"],5,22),"home_pa5":avg(h["pa"],5,22),"away_pa5":avg(a["pa"],5,22),"home_margin5":avg(h["margin"],5,0),"away_margin5":avg(a["margin"],5,0),"home_win5":avg(h["wins"],5,.5),"away_win5":avg(a["wins"],5,.5),"home_games_prior":h["games"],"away_games_prior":a["games"],"home_rest":hr,"away_rest":ar,"rest_diff":hr-ar})
        rows.append(r)
        if pd.notna(g.home_score) and pd.notna(g.away_score):
            hs,aw=float(g.home_score),float(g.away_score); margin=hs-aw; exp=1/(1+10**((a["elo"]-(h["elo"]+55))/400)); actual=1 if margin>0 else (.5 if margin==0 else 0); delta=20*math.log(abs(margin)+1)*(actual-exp); h["elo"]+=delta; a["elo"]-=delta
            for s,pf,pa,m,w in [(h,hs,aw,margin,actual),(a,aw,hs,-margin,1-actual)]:
                s["games"]+=1; s["pf"].append(pf); s["pa"].append(pa); s["margin"].append(m); s["wins"].append(w); s["last_date"]=gd if pd.notna(gd) else s["last_date"]
    out=pd.DataFrame(rows); out["target_margin"]=out.home_score-out.away_score; out["target_total"]=out.home_score+out.away_score; out["pf_matchup"]=(out.home_pf5+out.away_pa5)-(out.away_pf5+out.home_pa5); out["form_diff"]=out.home_margin5-out.away_margin5; out["winform_diff"]=out.home_win5-out.away_win5
    return out

def pipe(alpha):
    return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))])

class NFLEdgeModel:
    def __init__(self):
        self.margin=pipe(25); self.total=pipe(30); self.win_model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(C=.45,max_iter=2000))])
    def fit(self,games):
        f=pregame_features(games); t=f.dropna(subset=["target_margin","target_total"]).copy(); mature=t[(t.home_games_prior>=2)&(t.away_games_prior>=2)]
        if len(mature)>100: t=mature
        self.margin.fit(t[FEATURES],t.target_margin); self.total.fit(t[FEATURES],t.target_total); pm=self.margin.predict(t[FEATURES]); pt=self.total.predict(t[FEATURES]); self.margin_residuals=np.asarray(t.target_margin-pm,dtype=float); self.total_residuals=np.asarray(t.target_total-pt,dtype=float); self.win_model.fit(t[FEATURES],(t.target_margin>0).astype(int)); self.feature_df=f; return self
    def predict(self,row,n=50000,seed=42):
        X=pd.DataFrame([{f:row[f] for f in FEATURES}]); margin=float(self.margin.predict(X)[0]); total=float(self.total.predict(X)[0]); rng=np.random.default_rng(seed); margins=margin+rng.choice(self.margin_residuals,size=n,replace=True); totals=total+rng.choice(self.total_residuals,size=n,replace=True); hs=np.maximum(0,(totals+margins)/2); aws=np.maximum(0,(totals-margins)/2); sim_home=float(np.mean(hs>aws)); calibrated=float(self.win_model.predict_proba(X)[0,1]); home_win=.5*sim_home+.5*calibrated
        out={"home_score":max(6,(total+margin)/2),"away_score":max(6,(total-margin)/2),"margin":margin,"total":total,"home_win_prob":home_win,"sim_home_win_prob":sim_home,"calibrated_home_win_prob":calibrated,"home_scores":hs,"away_scores":aws}
        spread=row.get("spread_line",np.nan)
        if pd.notna(spread): out["home_cover_prob"]=float(np.mean(margins-float(spread)>0))
        tl=row.get("total_line",np.nan)
        if pd.notna(tl): out["over_prob"]=float(np.mean(totals>float(tl)))
        if pd.notna(spread) and float(spread)!=0:
            fav_home=float(spread)>0; upset=(1-home_win) if fav_home else home_win; fav_model=margin if fav_home else -margin; out["upset_prob"]=upset; out["upset_score"]=min(100,100*upset+2.5*max(0,abs(float(spread))-fav_model))
        return out
