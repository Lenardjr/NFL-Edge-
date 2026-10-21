from __future__ import annotations
# NFL Edge v0.7 deployment edition — standalone Streamlit app
# Data source attribution: nflverse (CC-BY 4.0 where applicable).

import math
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEAM_ALIASES={"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}
FEATURES=["elo_diff","pf_matchup","form_diff","winform_diff",
          "home_pf5","away_pf5","home_pa5","away_pa5",
          "home_games_prior","away_games_prior"]

def load_games(url):
    df=pd.read_csv(url)
    for c in ["home_team","away_team"]:
        df[c]=df[c].map(lambda x: TEAM_ALIASES.get(str(x),str(x)))
    df["gameday"]=pd.to_datetime(df["gameday"],errors="coerce")
    for c in ["home_score","away_score","spread_line","total_line"]:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.sort_values(["gameday","gametime"],na_position="last").reset_index(drop=True)

def pregame_features(df):
    rows=[]; state={}
    def team(t):
        return state.setdefault(t,{"games":0,"pf":[],"pa":[],"margin":[],"wins":[],"elo":1500.})
    def avg(v,n,d): return float(np.mean(v[-n:])) if v else d
    for _,g in df.iterrows():
        h,a=team(g.home_team),team(g.away_team)
        r=g.to_dict()
        r.update({
            "home_elo":h["elo"],"away_elo":a["elo"],"elo_diff":h["elo"]-a["elo"],
            "home_pf5":avg(h["pf"],5,22),"away_pf5":avg(a["pf"],5,22),
            "home_pa5":avg(h["pa"],5,22),"away_pa5":avg(a["pa"],5,22),
            "home_margin5":avg(h["margin"],5,0),"away_margin5":avg(a["margin"],5,0),
            "home_win5":avg(h["wins"],5,.5),"away_win5":avg(a["wins"],5,.5),
            "home_games_prior":h["games"],"away_games_prior":a["games"]})
        rows.append(r)
        if pd.notna(g.home_score) and pd.notna(g.away_score):
            hs,aw=float(g.home_score),float(g.away_score); margin=hs-aw
            exp=1/(1+10**((a["elo"]-(h["elo"]+55))/400))
            actual=1 if margin>0 else (.5 if margin==0 else 0)
            delta=20*math.log(abs(margin)+1)*(actual-exp)
            h["elo"]+=delta; a["elo"]-=delta
            for s,pf,pa,m,w in [(h,hs,aw,margin,actual),(a,aw,hs,-margin,1-actual)]:
                s["games"]+=1; s["pf"].append(pf); s["pa"].append(pa); s["margin"].append(m); s["wins"].append(w)
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
    def fit(self,games):
        f=pregame_features(games); t=f.dropna(subset=["target_margin","target_total"])
        mature=t[(t.home_games_prior>=2)&(t.away_games_prior>=2)]
        if len(mature)>100: t=mature
        self.margin.fit(t[FEATURES],t.target_margin); self.total.fit(t[FEATURES],t.target_total)
        self.margin_sd=max(8,float(np.std(t.target_margin-self.margin.predict(t[FEATURES]))))
        self.total_sd=max(8,float(np.std(t.target_total-self.total.predict(t[FEATURES]))))
        self.feature_df=f; return self
    def predict(self,row,n=50000,seed=42):
        X=pd.DataFrame([{f:row[f] for f in FEATURES}])
        m=float(self.margin.predict(X)[0]); total=float(self.total.predict(X)[0])
        hp=max(6,(total+m)/2); ap=max(6,(total-m)/2)
        rng=np.random.default_rng(seed)
        margins=rng.normal(m,self.margin_sd,n); totals=rng.normal(total,self.total_sd,n)
        hs=np.maximum(0,(totals+margins)/2); aws=np.maximum(0,(totals-margins)/2)
        out={"home_score":hp,"away_score":ap,"margin":m,"total":total,
             "home_win_prob":float(np.mean(hs>aws)),"home_scores":hs,"away_scores":aws}
        spread=row.get("spread_line",np.nan)
        if pd.notna(spread):
            # nflverse spread_line is the home-team margin line: +3 means home favored by 3.
            out["home_cover_prob"]=float(np.mean(margins-float(spread)>0))
        tl=row.get("total_line",np.nan)
        if pd.notna(tl): out["over_prob"]=float(np.mean(totals>float(tl)))
        if pd.notna(spread) and float(spread)!=0:
            fav_home=float(spread)>0
            upset=(1-out["home_win_prob"]) if fav_home else out["home_win_prob"]
            fav_model=m if fav_home else -m
            disagreement=max(0,abs(float(spread))-fav_model)
            out["upset_prob"]=upset
            out["upset_score"]=min(100,100*upset+2.5*disagreement)
        return out

def walk_forward_backtest(games,first_test_season=2022):
    f=pregame_features(games); done=f.dropna(subset=["target_margin","target_total"])
    outs=[]
    for season in sorted(done.season.dropna().astype(int).unique()):
        if season<first_test_season: continue
        tr=done[done.season<season]
        te=done[(done.season==season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(tr)<300 or te.empty: continue
        mm,tm=pipe(25),pipe(30); mm.fit(tr[FEATURES],tr.target_margin); tm.fit(tr[FEATURES],tr.target_total)
        pm,pt=mm.predict(te[FEATURES]),tm.predict(te[FEATURES])
        for (_,r),a,b in zip(te.iterrows(),pm,pt):
            outs.append({"season":season,"week":r.week,"Game":f"{r.away_team} @ {r.home_team}",
                         "actual_margin":r.target_margin,"pred_margin":a,
                         "actual_total":r.target_total,"pred_total":b,
                         "winner_correct":bool((a>0)==(r.target_margin>0)),
                         "margin_abs_error":abs(a-r.target_margin),
                         "total_abs_error":abs(b-r.target_total),
                         "market_spread":r.get("spread_line",np.nan),
                         "market_total":r.get("total_line",np.nan),
                         "model_market_margin_gap":(a-r.get("spread_line",np.nan)) if pd.notna(r.get("spread_line",np.nan)) else np.nan})
    return pd.DataFrame(outs)


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
st.markdown('<div class="small">V0.7 • One engine. Games, props, fantasy and simulations.</div>',unsafe_allow_html=True)

page=st.radio("Navigation",["🏈 GAMES","📊 PROPS","🏆 FANTASY","🔬 SIMULATOR","📈 MODEL RECORD"],horizontal=True,label_visibility="collapsed")

def selectors(prefix=""):
    seasons=sorted(f.season.dropna().astype(int).unique(),reverse=True)
    season=st.selectbox("Season",seasons,key=prefix+"season")
    weeks=sorted(f[f.season==season].week.dropna().unique())
    week=st.selectbox("Week",weeks,index=len(weeks)-1,key=prefix+"week")
    slate=f[(f.season==season)&(f.week==week)]
    return season,week,slate

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
    st.header("Weekly Games")
    season,week,slate=selectors("g")
    rows=[]
    for _,row in slate.iterrows():
        r=m.predict(row,15000,seed=int(row.name)%10000)
        rows.append({
            "Game":f"{row.away_team} @ {row.home_team}",
            "NFL Edge Score":f"{row.away_team} {r['away_score']:.1f} – {row.home_team} {r['home_score']:.1f}",
            "Home Win":r["home_win_prob"],
            "Model Margin":r["margin"],
            "Market":row.get("spread_line",np.nan),
            "Total":r["total"],
            "Market O/U":row.get("total_line",np.nan),
            "Upset":r.get("upset_score",np.nan)})
    board=pd.DataFrame(rows)
    if not board.empty:
        view=board.copy()
        view["Home Win"]=view["Home Win"].map(lambda x:f"{x:.1%}")
        for c in ["Model Margin","Market","Total","Market O/U","Upset"]:
            view[c]=pd.to_numeric(view[c],errors="coerce").round(1)
        st.dataframe(view,use_container_width=True,hide_index=True)
        st.caption("Market columns are comparisons, not inputs to the pure model.")

elif page=="🔬 SIMULATOR":
    st.header("Game Simulator")
    _,_,slate=selectors("s")
    row=matchup_select(slate,"sm")
    if row is not None:
        n=st.slider("Monte Carlo simulations",10000,100000,50000,10000)
        r=game_card(row,n)
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
    st.header("Model Record")
    st.write("NFL Edge should earn trust through frozen, out-of-sample results—not by changing predictions after games.")
    if st.button("Run game-model walk-forward record",type="primary"):
        with st.spinner("Backtesting..."): bt=walk_forward_backtest(g,2022)
        if not bt.empty:
            a,b,c=st.columns(3)
            a.metric("Winner accuracy",f"{bt.winner_correct.mean():.1%}")
            b.metric("Margin MAE",f"{bt.margin_abs_error.mean():.2f}")
            c.metric("Total MAE",f"{bt.total_abs_error.mean():.2f}")
            st.dataframe(bt.sort_values(["season","week"],ascending=False).head(250),
                         use_container_width=True,hide_index=True)
    st.info("The permanent prediction ledger is the next deployment-stage feature: predictions will be timestamped and never overwritten after kickoff.")

st.divider()
st.caption("NFL Edge is probabilistic research software. A projection is not a guarantee, and PASS is a valid model decision.")
