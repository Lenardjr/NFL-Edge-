from __future__ import annotations

import pandas as pd
import streamlit as st

from player_engine import normalize_player_schema, project_all_players_as_of, player_history_as_of
from player_validation import validate_player_feed
from player_backtest import backtest_fantasy_projections, fantasy_backtest_summary, fantasy_backtest_by_position
from live_game import build_live_game_forecast

PLAYER_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
BUILD_VERSION = "v1.5.0-game-player-connected"
QB_MODEL_VERSION = "QB Engine 1.0 • legacy champion frozen"
BACKTEST_CACHE_VERSION = "player-engine-1.0-champion-v1"
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")

st.set_page_config(page_title="NFL Edge", page_icon="🏈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.1rem;padding-bottom:3rem}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);padding:12px;border-radius:12px}
.small{opacity:.7;font-size:.9rem}.live-card{border:1px solid rgba(128,128,128,.35);border-radius:18px;padding:16px 18px;margin:10px 0}
@media(max-width:700px){.block-container{padding-left:.7rem;padding-right:.7rem}[data-testid="stMetricValue"]{font-size:1.25rem}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_player_stats(start_season: int, end_season: int) -> pd.DataFrame:
    frames=[]
    for s in range(start_season,end_season+1):
        try: frames.append(pd.read_csv(PLAYER_URL_TEMPLATE.format(season=s),low_memory=False))
        except Exception: continue
    return normalize_player_schema(pd.concat(frames,ignore_index=True)) if frames else pd.DataFrame()

@st.cache_data(ttl=900,show_spinner=False)
def live_forecast():
    return build_live_game_forecast(2026,7,"NE","CHI",simulations=50000,seed=42)

def fantasy_only(df):
    if df is None or df.empty or "position" not in df.columns: return pd.DataFrame() if df is None else df.copy()
    d=df.copy(); d["position"]=d["position"].astype(str).str.upper().str.strip(); return d[d["position"].isin(FANTASY_POSITIONS)].copy()

@st.cache_data(ttl=900,show_spinner=False)
def run_backtest(stats,season,week,lookback,min_games,cache_version):
    _=cache_version
    return fantasy_only(backtest_fantasy_projections(fantasy_only(stats),start_season=max(2022,int(season)-1),end_season=int(season),start_week=4,end_week=min(max(int(week)-1,4),18),lookback=lookback,min_games=min_games))

st.title("🏈 NFL EDGE")
st.markdown(f'<div class="small">Game Engine 1.0 + Player Engine 1.0 • <b>{BUILD_VERSION}</b><br><b>{QB_MODEL_VERSION}</b></div>',unsafe_allow_html=True)
st.info("Live Test #001 now connects the protected Game Engine forecast to the leakage-safe Player Engine. The test remains unfrozen until the pregame snapshot is formally recorded.")

with st.sidebar:
    st.header("Projection Settings"); st.caption(f"Live build: {BUILD_VERSION}"); st.caption(QB_MODEL_VERSION)
    season=st.selectbox("Season",list(range(2026,2019,-1)),index=0); week=st.slider("Project week",1,22,7); lookback=st.slider("Recent games used",3,12,8); min_games=st.slider("Minimum prior games",2,8,3)

with st.spinner("Loading nflverse player data..."): stats=load_player_stats(max(2020,season-1),season)
health=validate_player_feed(stats)
if not health["ok"]: st.error("Player data feed is not projection-ready."); st.stop()
history=player_history_as_of(stats,season=season,week=week)
if history.empty: st.warning(f"No completed player history is available before {season} Week {week}."); st.stop()
with st.spinner("Building leakage-safe player projections..."): rankings=project_all_players_as_of(stats,season=season,week=week,lookback=lookback,min_games=min_games)
if rankings.empty: st.warning("Not enough completed player games are available."); st.stop()
st.caption(f"As-of cutoff: {season} Week {week} • {health['rows']:,} feed rows • {health['teams']} normalized teams • target-week results excluded")

if season==2026 and week==7:
    st.divider(); st.header("🧪 Live Test #001 — Patriots @ Bears"); st.caption("Thursday, Oct. 22, 2026 • 8:15 PM ET • Soldier Field • pregame research workspace")
    with st.spinner("Running Game Engine 1.0 and 50,000 simulations..."): game=live_forecast()
    if not game.get("ok"):
        st.error("Game Engine forecast unavailable: "+game.get("error","unknown error"))
    else:
        st.subheader("NFL Edge Game Forecast")
        a,b,c,d=st.columns(4)
        a.metric("Projected score",f"NE {game['away_score']:.1f} — CHI {game['home_score']:.1f}")
        b.metric("Bears win probability",f"{game['home_win_prob']:.1%}")
        c.metric("Model margin",f"CHI {game['home_margin']:+.1f}" if game['home_margin']>=0 else f"NE {-game['home_margin']:+.1f}")
        d.metric("Model total",f"{game['total']:.1f}")
        st.caption(f"Game Engine 1.0 • 50,000 simulations • {game['training_games']:,} completed pregame training games • target result excluded")

        st.subheader("Model vs Market")
        a,b,c,d=st.columns(4)
        market_spread=game.get("market_spread"); market_total=game.get("market_total")
        a.metric("Schedule market spread",f"{market_spread:+.1f}" if market_spread is not None else "—")
        b.metric("Schedule market total",f"{market_total:.1f}" if market_total is not None else "—")
        c.metric("Home cover probability",f"{game['home_cover_prob']:.1%}" if game.get('home_cover_prob') is not None else "—")
        d.metric("Over probability",f"{game['over_prob']:.1%}" if game.get('over_prob') is not None else "—")
        if game.get("upset_prob") is not None: st.caption(f"Model upset probability: {game['upset_prob']:.1%} • vulnerability score: {game['upset_score']:.1f}/100")
        st.warning("Market fields above come from the nflverse schedule feed and are comparison context only. They are not Game Engine input features. We will capture the official pregame market separately when the test is frozen.")

    matchup=rankings[rankings["team"].isin(["NE","CHI"])].copy()
    st.subheader("Connected Player Forecast")
    if matchup.empty: st.warning("No NE/CHI players were found in the current projection board.")
    else:
        qbs=matchup[matchup["position"].astype(str).str.upper().eq("QB")].copy()
        if not qbs.empty:
            cols=st.columns(max(1,min(2,len(qbs))))
            for i,(_,p) in enumerate(qbs.sort_values("projected_ppr",ascending=False).iterrows()):
                with cols[i%len(cols)]: st.markdown(f"<div class='live-card'><b>{p.player_name}</b> · {p.team}<br>Projected PPR: <b>{p.projected_ppr:.1f}</b><br>Range: <b>{p.floor:.1f}–{p.ceiling:.1f}</b> · Confidence: <b>{p.confidence:.0%}</b></div>",unsafe_allow_html=True)
        for club,label in [("NE","New England"),("CHI","Chicago")]:
            club_rows=matchup[matchup["team"]==club].copy(); st.markdown(f"**{label} skill players**")
            show=club_rows[club_rows["position"].isin(["RB","WR","TE"])][["player_name","position","projected_ppr","floor","ceiling","confidence"]].head(10).copy()
            show=show.rename(columns={"player_name":"Player","position":"Pos","projected_ppr":"PPR","floor":"Floor","ceiling":"Ceiling","confidence":"Confidence"})
            if not show.empty: show["Confidence"]=show["Confidence"].map(lambda x:f"{x:.0%}"); st.dataframe(show,use_container_width=True,hide_index=True)

    st.subheader("Live Test #001 status")
    st.info("GAME + PLAYER CONNECTION COMPLETE • Still UNFROZEN")
    st.markdown("**Remaining before freeze:** capture current official market snapshot, add injury/availability context, write the immutable pregame record, then lock the test before kickoff.")

st.divider(); st.subheader(f"Fantasy Projection Board — {season} Week {week}")
positions=sorted([str(x) for x in rankings.position.dropna().unique()]) if "position" in rankings else []; teams=sorted([str(x) for x in rankings.team.dropna().unique()]) if "team" in rankings else []
c1,c2,c3=st.columns([1,1,2]); position=c1.selectbox("Position",["ALL"]+positions); team=c2.selectbox("Team",["ALL"]+teams); search=c3.text_input("Player search",placeholder="e.g. Drake Maye")
filtered=rankings.copy()
if position!="ALL": filtered=filtered[filtered.position==position]
if team!="ALL": filtered=filtered[filtered.team==team]
if search.strip(): filtered=filtered[filtered.player_name.str.contains(search.strip(),case=False,na=False)]
show_cols=[c for c in ["player_name","position","team","projected_ppr","floor","ceiling","recent_avg","trend","confidence","games_used"] if c in filtered.columns]
board=filtered[show_cols].head(100).copy().rename(columns={"player_name":"Player","position":"Pos","team":"Team","projected_ppr":"Projection","floor":"Floor","ceiling":"Ceiling","recent_avg":"Recent Avg","trend":"Trend","confidence":"Confidence","games_used":"Games"})
if "Confidence" in board: board["Confidence"]=board["Confidence"].map(lambda x:f"{x:.0%}")
st.dataframe(board,use_container_width=True,hide_index=True)

st.divider(); st.subheader("Historical Accuracy Gate — Player Engine 1.0")
if week>=5 and st.button("Run Player Engine 1.0 historical validation",type="secondary"):
    with st.spinner("Replaying frozen Player Engine 1.0 on historical samples..."): results=run_backtest(stats,season,week,lookback,min_games,BACKTEST_CACHE_VERSION)
    if not results.empty:
        summary=fantasy_backtest_summary(results); a,b,c,d,e=st.columns(5); a.metric("Samples",f"{summary['n']:,}"); b.metric("MAE",f"{summary['mae']:.2f}"); c.metric("Correlation",f"{summary['correlation']:.2f}" if pd.notna(summary['correlation']) else "—"); d.metric("Range coverage",f"{summary['interval_coverage']:.0%}" if pd.notna(summary['interval_coverage']) else "—"); e.metric("Reliability",summary["tier"])
        by_position=fantasy_backtest_by_position(results)
        if not by_position.empty: st.dataframe(by_position.round(3),use_container_width=True,hide_index=True)

st.caption(f"Live build: {BUILD_VERSION} • {QB_MODEL_VERSION}. Game + Player connection is active; Live Test #001 remains unfrozen pending the official pregame snapshot.")
