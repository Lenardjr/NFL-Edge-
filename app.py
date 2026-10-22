from __future__ import annotations

import pandas as pd
import streamlit as st

from player_engine import normalize_player_schema, project_all_players_as_of, player_history_as_of
from player_validation import validate_player_feed
from player_backtest import backtest_fantasy_projections, fantasy_backtest_summary, fantasy_backtest_by_position
from live_game import build_live_game_forecast

PLAYER_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
BUILD_VERSION = "v1.7.0-game-night-beta"
QB_MODEL_VERSION = "QB Engine 1.0 • legacy champion frozen"
BACKTEST_CACHE_VERSION = "player-engine-1.0-champion-v1"
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")
OFFICIAL_MARKET = {"spread":"CHI -3.0","home_margin":3.0,"total":50.5,"ne_ml":"+132","chi_ml":"-156","captured":"2026-10-22"}
FROZEN_E10_MARGIN = 0.8
FROZEN_E11_MARGIN = 1.46

st.set_page_config(page_title="NFL Edge", page_icon="🏈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.block-container{max-width:1180px;padding-top:1.1rem;padding-bottom:3rem}[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);padding:12px;border-radius:12px}.small{opacity:.7;font-size:.9rem}.live-card{border:1px solid rgba(128,128,128,.35);border-radius:18px;padding:16px 18px;margin:10px 0}.hero{border:1px solid rgba(128,128,128,.32);border-radius:20px;padding:18px;margin:14px 0}.edge{font-size:1.08rem;font-weight:700}.frozen{font-size:.82rem;opacity:.75}@media(max-width:700px){.block-container{padding-left:.7rem;padding-right:.7rem}[data-testid="stMetricValue"]{font-size:1.25rem}}</style>""",unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_player_stats(start_season,end_season):
    frames=[]
    for s in range(start_season,end_season+1):
        try: frames.append(pd.read_csv(PLAYER_URL_TEMPLATE.format(season=s),low_memory=False))
        except Exception: continue
    return normalize_player_schema(pd.concat(frames,ignore_index=True)) if frames else pd.DataFrame()

@st.cache_data(ttl=900,show_spinner=False)
def live_forecast(): return build_live_game_forecast(2026,7,"NE","CHI",simulations=50000,seed=42)

def fantasy_only(df):
    if df is None or df.empty or "position" not in df.columns:return pd.DataFrame() if df is None else df.copy()
    d=df.copy();d["position"]=d["position"].astype(str).str.upper().str.strip();return d[d["position"].isin(FANTASY_POSITIONS)].copy()

@st.cache_data(ttl=900,show_spinner=False)
def run_backtest(stats,season,week,lookback,min_games,cache_version):
    _=cache_version;return fantasy_only(backtest_fantasy_projections(fantasy_only(stats),start_season=max(2022,int(season)-1),end_season=int(season),start_week=4,end_week=min(max(int(week)-1,4),18),lookback=lookback,min_games=min_games))

st.title("🏈 NFL EDGE");st.markdown(f'<div class="small">Game Engine 1.0 + frozen 1.1-B challenger + Player Engine 1.0 • <b>{BUILD_VERSION}</b><br><b>{QB_MODEL_VERSION}</b></div>',unsafe_allow_html=True)
st.success("GAME-NIGHT BETA • Live Test #001 is frozen before kickoff. Tonight is prospective validation, not retrospective tuning.")
with st.sidebar:
    st.header("Projection Settings");season=st.selectbox("Season",list(range(2026,2019,-1)),index=0);week=st.slider("Project week",1,22,7);lookback=st.slider("Recent games used",3,12,8);min_games=st.slider("Minimum prior games",2,8,3)
with st.spinner("Loading nflverse player data..."):stats=load_player_stats(max(2020,season-1),season)
health=validate_player_feed(stats)
if not health["ok"]:st.error("Player data feed is not projection-ready.");st.stop()
history=player_history_as_of(stats,season=season,week=week)
if history.empty:st.warning("No completed player history available.");st.stop()
with st.spinner("Building leakage-safe player projections..."):rankings=project_all_players_as_of(stats,season=season,week=week,lookback=lookback,min_games=min_games)
if rankings.empty:st.warning("Not enough player history.");st.stop()
st.caption(f"As-of cutoff: {season} Week {week} • {health['rows']:,} feed rows • target-week results excluded")

if season==2026 and week==7:
    st.divider();st.header("🏟️ Game Center — Patriots @ Bears");st.caption("Thursday, Oct. 22, 2026 • 8:15 PM ET • Soldier Field • Live Test #001")
    with st.spinner("Loading frozen-seed Game Engine forecast..."):game=live_forecast()
    if not game.get("ok"):st.error("Game Engine forecast unavailable: "+game.get("error","unknown error"))
    else:
        st.markdown("<div class='hero'><b>OFFICIAL FROZEN FORECAST</b><br><span class='frozen'>Prediction locked before kickoff • 50,000 simulations • seed 42</span></div>",unsafe_allow_html=True)
        a,b,c,d=st.columns(4);a.metric("Projected score",f"NE {game['away_score']:.1f} — CHI {game['home_score']:.1f}");b.metric("CHI win probability",f"{game['home_win_prob']:.1%}");c.metric("Engine 1.0 fair margin",f"CHI by {FROZEN_E10_MARGIN:.1f}");d.metric("Projected total",f"{game['total']:.1f}")
        st.subheader("Model vs market")
        a,b,c,d=st.columns(4);a.metric("Market spread",OFFICIAL_MARKET['spread']);b.metric("1.1-B challenger",f"CHI by {FROZEN_E11_MARGIN:.2f}");c.metric("Spread difference",f"{OFFICIAL_MARKET['home_margin']-FROZEN_E10_MARGIN:.1f} pts");d.metric("Total difference",f"{OFFICIAL_MARKET['total']-game['total']:.1f} pts")
        x,y=st.columns(2)
        x.info("**SPREAD LEAN: NE +3**\n\nNFL Edge makes Chicago only a slight favorite. Engine 1.0 is 2.2 points inside the market spread; the frozen 1.1-B challenger is Chicago by 1.46.")
        y.info("**TOTAL LEAN: UNDER 50.5**\n\nNFL Edge projects 42.8 total points, 7.7 below the frozen market total.")
        st.caption("Leans are model-vs-market research signals, not guarantees. The market snapshot is frozen external context; it was not an Engine 1.0 input.")
        st.subheader("Frozen market snapshot")
        a,b,c,d=st.columns(4);a.metric("Spread",OFFICIAL_MARKET['spread']);b.metric("Total",f"{OFFICIAL_MARKET['total']:.1f}");c.metric("NE moneyline",OFFICIAL_MARKET['ne_ml']);d.metric("CHI moneyline",OFFICIAL_MARKET['chi_ml'])
        st.subheader("Availability snapshot")
        st.markdown("NE is without Ben Brown, Alijah Vera-Tucker, Carlton Davis III, Channing Canada and Karon Prunty. CHI is without Kyle Monangai and Jack Sanborn; T.J. Edwards is doubtful. Drake Maye, A.J. Brown and TreVeyon Henderson have no game-status designation.")
        matchup=rankings[rankings["team"].isin(["NE","CHI"])].copy();st.subheader("Player Engine — game-night board")
        qbs=matchup[matchup["position"].astype(str).str.upper().eq("QB")]
        for _,p in qbs.sort_values("projected_ppr",ascending=False).iterrows():st.markdown(f"<div class='live-card'><b>{p.player_name}</b> · {p.team} — <b>{p.projected_ppr:.1f} PPR</b> · range {p.floor:.1f}–{p.ceiling:.1f} · confidence {p.confidence:.0%}</div>",unsafe_allow_html=True)
        for club,label in [("NE","New England skill players"),("CHI","Chicago skill players")]:
            show=matchup[(matchup.team==club)&matchup.position.isin(["RB","WR","TE"])][["player_name","position","projected_ppr","floor","ceiling","confidence"]].head(10).copy();show.columns=["Player","Pos","PPR","Floor","Ceiling","Confidence"]
            if not show.empty:show["Confidence"]=show.Confidence.map(lambda z:f"{z:.0%}");st.markdown(f"**{label}**");st.dataframe(show,use_container_width=True,hide_index=True)
        st.warning("Known Player Engine limitation: the frozen audit retains players later ruled OUT rather than silently changing the pregame record. Kyle Monangai remains visible for audit integrity.")
        st.subheader("Prospective model record")
        a,b,c=st.columns(3);a.metric("Registered tests","1");b.metric("Completed","0");c.metric("Status","Awaiting result")
        st.caption("After the final score, a separate result artifact will grade Engine 1.0 and frozen Engine 1.1-B. The prediction itself will never be rewritten.")

st.divider();st.subheader(f"Fantasy Projection Board — {season} Week {week}")
positions=sorted([str(x) for x in rankings.position.dropna().unique()]);teams=sorted([str(x) for x in rankings.team.dropna().unique()]);c1,c2,c3=st.columns([1,1,2]);position=c1.selectbox("Position",["ALL"]+positions);team=c2.selectbox("Team",["ALL"]+teams);search=c3.text_input("Player search",placeholder="e.g. Drake Maye")
filtered=rankings.copy()
if position!="ALL":filtered=filtered[filtered.position==position]
if team!="ALL":filtered=filtered[filtered.team==team]
if search.strip():filtered=filtered[filtered.player_name.str.contains(search.strip(),case=False,na=False)]
show_cols=[c for c in ["player_name","position","team","projected_ppr","floor","ceiling","recent_avg","trend","confidence","games_used"] if c in filtered.columns];board=filtered[show_cols].head(100).copy();st.dataframe(board,use_container_width=True,hide_index=True)
st.divider();st.subheader("Historical Accuracy Gate — Player Engine 1.0")
if week>=5 and st.button("Run Player Engine 1.0 historical validation",type="secondary"):
    with st.spinner("Replaying frozen Player Engine 1.0..."):results=run_backtest(stats,season,week,lookback,min_games,BACKTEST_CACHE_VERSION)
    if not results.empty:
        summary=fantasy_backtest_summary(results);a,b,c,d,e=st.columns(5);a.metric("Samples",f"{summary['n']:,}");b.metric("MAE",f"{summary['mae']:.2f}");c.metric("Correlation",f"{summary['correlation']:.2f}" if pd.notna(summary['correlation']) else "—");d.metric("Coverage",f"{summary['interval_coverage']:.0%}" if pd.notna(summary['interval_coverage']) else "—");e.metric("Reliability",summary['tier'])
        by_position=fantasy_backtest_by_position(results)
        if not by_position.empty:st.dataframe(by_position.round(3),use_container_width=True,hide_index=True)
st.caption(f"{BUILD_VERSION} • Live Test #001 frozen • prospective scoring follows after the game is final.")