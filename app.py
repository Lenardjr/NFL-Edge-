from __future__ import annotations

import pandas as pd
import streamlit as st

from player_engine import normalize_player_schema, project_all_players_as_of, player_history_as_of
from player_validation import validate_player_feed
from player_backtest import backtest_fantasy_projections, fantasy_backtest_summary, fantasy_backtest_by_position

PLAYER_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
BUILD_VERSION = "v1.4.0-live-test-001"
QB_MODEL_VERSION = "QB Engine 1.0 • legacy champion frozen"
BACKTEST_CACHE_VERSION = "player-engine-1.0-champion-v1"
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")

st.set_page_config(page_title="NFL Edge", page_icon="🏈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.1rem;padding-bottom:3rem}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);padding:12px;border-radius:12px}
.small{opacity:.7;font-size:.9rem}.edge-card{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:14px 16px;margin:8px 0}
.live-card{border:1px solid rgba(128,128,128,.35);border-radius:18px;padding:16px 18px;margin:10px 0}
@media(max-width:700px){.block-container{padding-left:.7rem;padding-right:.7rem}[data-testid="stMetricValue"]{font-size:1.25rem}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_player_stats(start_season: int, end_season: int) -> pd.DataFrame:
    frames = []
    for season in range(start_season, end_season + 1):
        try:
            frames.append(pd.read_csv(PLAYER_URL_TEMPLATE.format(season=season), low_memory=False))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return normalize_player_schema(pd.concat(frames, ignore_index=True))


def fantasy_only(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "position" not in df.columns:
        return pd.DataFrame() if df is None else df.copy()
    d = df.copy()
    d["position"] = d["position"].astype(str).str.upper().str.strip()
    return d[d["position"].isin(FANTASY_POSITIONS)].copy()


@st.cache_data(ttl=900, show_spinner=False)
def run_backtest(stats: pd.DataFrame, season: int, week: int, lookback: int, min_games: int, cache_version: str) -> pd.DataFrame:
    _ = cache_version
    return fantasy_only(backtest_fantasy_projections(
        fantasy_only(stats), start_season=max(2022, int(season)-1), end_season=int(season),
        start_week=4, end_week=min(max(int(week)-1, 4), 18), lookback=lookback, min_games=min_games,
    ))

st.title("🏈 NFL EDGE")
st.markdown(f'<div class="small">Player Engine 1.0 • Game + Player integration build • <b>{BUILD_VERSION}</b><br><b>{QB_MODEL_VERSION}</b></div>', unsafe_allow_html=True)
st.info("Live Test #001 is now the development target. Game Engine 1.0 remains protected in nfl_edge.py while this workspace connects its matchup concept to the validated Player Engine.")

current_season = 2026
with st.sidebar:
    st.header("Projection Settings")
    st.caption(f"Live build: {BUILD_VERSION}")
    st.caption(QB_MODEL_VERSION)
    season = st.selectbox("Season", list(range(current_season, 2019, -1)), index=0)
    week = st.slider("Project week", 1, 22, 7)
    lookback = st.slider("Recent games used", 3, 12, 8)
    min_games = st.slider("Minimum prior games", 2, 8, 3)

with st.spinner("Loading nflverse player data..."):
    stats = load_player_stats(max(2020, season - 1), season)
health = validate_player_feed(stats)
if not health["ok"]:
    st.error("Player data feed is not projection-ready: " + (", ".join(health["missing"]) if health["missing"] else "unknown feed issue") + ".")
    st.stop()

history = player_history_as_of(stats, season=season, week=week)
if history.empty:
    st.warning(f"No completed player history is available before {season} Week {week}.")
    st.stop()
with st.spinner("Building leakage-safe player projections..."):
    rankings = project_all_players_as_of(stats, season=season, week=week, lookback=lookback, min_games=min_games)
if rankings.empty:
    st.warning("Not enough completed player games are available for this projection setup.")
    st.stop()

st.caption(f"As-of cutoff: {season} Week {week} • {health['rows']:,} feed rows • {health['teams']} normalized teams • target-week results excluded")

# First connected live-test workspace. Market values are deliberately displayed as a snapshot,
# not used to tune the model. They can be updated immediately before the official freeze.
if season == 2026 and week == 7:
    st.divider()
    st.header("🧪 Live Test #001 — Patriots @ Bears")
    st.caption("Thursday, Oct. 22, 2026 • 8:15 PM ET • Soldier Field • pregame research workspace")
    a,b,c = st.columns(3)
    a.metric("Reference spread", "CHI -3.0")
    b.metric("Reference total", "50.5")
    c.metric("Model status", "PREGAME / UNFROZEN")
    st.caption("Reference market is context only. The official model snapshot will be frozen separately before kickoff so the prediction cannot be changed after seeing the result.")

    matchup = rankings[rankings["team"].isin(["NE", "CHI"])].copy()
    if matchup.empty:
        st.warning("No NE/CHI players were found in the current projection board.")
    else:
        for club, label in [("NE", "New England Patriots"), ("CHI", "Chicago Bears")]:
            club_rows = matchup[matchup["team"] == club].copy()
            st.subheader(label)
            if club_rows.empty:
                st.info("No eligible projections available for this team under the current minimum-games setting.")
                continue
            show = club_rows[[c for c in ["player_name","position","projected_ppr","floor","ceiling","recent_avg","trend","confidence","games_used"] if c in club_rows.columns]].copy()
            show = show.rename(columns={"player_name":"Player","position":"Pos","projected_ppr":"PPR Projection","floor":"Floor","ceiling":"Ceiling","recent_avg":"Recent Avg","trend":"Trend","confidence":"Confidence","games_used":"Games"})
            if "Confidence" in show:
                show["Confidence"] = show["Confidence"].map(lambda x: f"{x:.0%}")
            st.dataframe(show.head(15), use_container_width=True, hide_index=True)

        qbs = matchup[matchup["position"].astype(str).str.upper().eq("QB")].copy()
        if not qbs.empty:
            st.subheader("Quarterback checkpoint")
            for _, p in qbs.sort_values("projected_ppr", ascending=False).iterrows():
                st.markdown(f"<div class='live-card'><b>{p.player_name}</b> · {p.team}<br>Projected PPR: <b>{p.projected_ppr:.1f}</b> · Range: <b>{p.floor:.1f}–{p.ceiling:.1f}</b> · Confidence: <b>{p.confidence:.0%}</b></div>", unsafe_allow_html=True)

        st.subheader("Freeze checklist")
        st.markdown("1. Connect Game Engine score/win probability.  \n2. Capture official model spread/total.  \n3. Capture NE/CHI player projections.  \n4. Record market snapshot and injury context.  \n5. Freeze Live Test #001 before kickoff.")

st.divider()
positions = sorted([str(x) for x in rankings.get("position", pd.Series(dtype=str)).dropna().unique() if str(x)])
team_values = sorted([str(x) for x in rankings.get("team", pd.Series(dtype=str)).dropna().unique() if str(x)])
c1,c2,c3 = st.columns([1,1,2])
position = c1.selectbox("Position", ["ALL"] + positions)
team = c2.selectbox("Team", ["ALL"] + team_values)
search = c3.text_input("Player search", placeholder="e.g. Drake Maye")
filtered = rankings.copy()
if position != "ALL": filtered = filtered[filtered["position"] == position]
if team != "ALL": filtered = filtered[filtered["team"] == team]
if search.strip(): filtered = filtered[filtered["player_name"].str.contains(search.strip(), case=False, na=False)]

st.subheader(f"Fantasy Projection Board — {season} Week {week}")
a,b,c,d = st.columns(4)
a.metric("Players ranked", len(filtered))
b.metric("Top projection", f"{filtered.projected_ppr.max():.1f}" if len(filtered) else "—")
c.metric("Median projection", f"{filtered.projected_ppr.median():.1f}" if len(filtered) else "—")
d.metric("Avg confidence", f"{filtered.confidence.mean():.0%}" if len(filtered) else "—")
show_cols = [c for c in ["player_name","position","team","projected_ppr","floor","ceiling","recent_avg","trend","confidence","games_used"] if c in filtered.columns]
board = filtered[show_cols].head(100).copy().rename(columns={"player_name":"Player","position":"Pos","team":"Team","projected_ppr":"Projection","floor":"Floor","ceiling":"Ceiling","recent_avg":"Recent Avg","trend":"Trend","confidence":"Confidence","games_used":"Games"})
if "Confidence" in board: board["Confidence"] = board["Confidence"].map(lambda x: f"{x:.0%}")
st.dataframe(board, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Historical Accuracy Gate — Player Engine 1.0")
st.caption("QB legacy champion is frozen. Validation remains fantasy-only and leakage-safe; opponent/injury/market adjustments are not yet claimed as validated edge.")
if week >= 5 and st.button("Run Player Engine 1.0 historical validation", type="secondary"):
    with st.spinner("Replaying frozen Player Engine 1.0 on historical samples..."):
        results = run_backtest(stats, season, week, lookback, min_games, BACKTEST_CACHE_VERSION)
    if results.empty:
        st.warning("No historical projection matches were available for this setup.")
    else:
        summary = fantasy_backtest_summary(results)
        a,b,c,d,e = st.columns(5)
        a.metric("Fantasy samples", f"{summary['n']:,}")
        b.metric("MAE", f"{summary['mae']:.2f}")
        c.metric("Correlation", f"{summary['correlation']:.2f}" if pd.notna(summary['correlation']) else "—")
        d.metric("Range coverage", f"{summary['interval_coverage']:.0%}" if pd.notna(summary['interval_coverage']) else "—")
        e.metric("Reliability", summary["tier"])
        by_position = fantasy_backtest_by_position(results)
        if not by_position.empty:
            st.dataframe(by_position.round(3), use_container_width=True, hide_index=True)

st.caption(f"Live build: {BUILD_VERSION} • {QB_MODEL_VERSION}. Live Test #001 remains unfrozen until Game Engine output, market snapshot and player board are captured together.")
