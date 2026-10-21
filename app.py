from __future__ import annotations

import pandas as pd
import streamlit as st

from player_engine import normalize_player_schema, project_all_players

PLAYER_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"

st.set_page_config(page_title="NFL Edge", page_icon="🏈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.1rem;padding-bottom:3rem}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);padding:12px;border-radius:12px}
.small{opacity:.7;font-size:.9rem}.edge-card{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:14px 16px;margin:8px 0}
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

st.title("🏈 NFL EDGE")
st.markdown('<div class="small">Player Projection Engine v1 • Fantasy + Props foundation</div>', unsafe_allow_html=True)
st.info("This build exposes the new player engine for validation. Game Engine 1.0 remains protected in nfl_edge.py while the player layer is tested.")

current_season = 2026
with st.sidebar:
    st.header("Projection Settings")
    season = st.selectbox("Season", list(range(current_season, 2019, -1)), index=0)
    lookback = st.slider("Recent games used", 3, 12, 8)
    min_games = st.slider("Minimum games", 2, 8, 3)

with st.spinner("Loading nflverse player data..."):
    stats = load_player_stats(max(2020, season - 1), season)

if stats.empty:
    st.error("Player data could not be loaded from nflverse.")
    st.stop()

season_stats = stats[stats["season"] == season].copy() if "season" in stats.columns else stats.copy()
if season_stats.empty:
    st.warning(f"No player rows are available for {season} yet.")
    st.stop()

with st.spinner("Building player projections..."):
    rankings = project_all_players(season_stats, lookback=lookback, min_games=min_games)

if rankings.empty:
    st.warning("Not enough completed player games are available for this projection setup.")
    st.stop()

positions = sorted([str(x) for x in season_stats.get("position", pd.Series(dtype=str)).dropna().unique()])
team_values = sorted([str(x) for x in season_stats.get("recent_team", pd.Series(dtype=str)).dropna().unique()])

c1, c2, c3 = st.columns([1,1,2])
position = c1.selectbox("Position", ["ALL"] + positions)
team = c2.selectbox("Team", ["ALL"] + team_values)
search = c3.text_input("Player search", placeholder="e.g. Drake Maye")

filtered = rankings.copy()
# Attach latest position because the engine intentionally keeps its projection output compact.
if "position" in season_stats.columns:
    id_col = next((c for c in ("player_id", "gsis_id", "player_display_name", "player_name") if c in season_stats.columns), None)
    if id_col:
        latest = season_stats.sort_values([c for c in ("season", "week") if c in season_stats.columns]).groupby(id_col, dropna=True).tail(1)
        name_col = next((c for c in ("player_display_name", "player_name") if c in latest.columns), None)
        if name_col:
            pos_map = latest.drop_duplicates(name_col).set_index(name_col)["position"].to_dict()
            filtered["position"] = filtered["player_name"].map(pos_map).fillna("")

if position != "ALL" and "position" in filtered.columns:
    filtered = filtered[filtered["position"] == position]
if team != "ALL":
    filtered = filtered[filtered["team"] == team]
if search.strip():
    filtered = filtered[filtered["player_name"].str.contains(search.strip(), case=False, na=False)]

st.subheader("Fantasy Projection Board")
a,b,c,d = st.columns(4)
a.metric("Players ranked", len(filtered))
b.metric("Top projection", f"{filtered.projected_ppr.max():.1f}" if len(filtered) else "—")
c.metric("Median projection", f"{filtered.projected_ppr.median():.1f}" if len(filtered) else "—")
d.metric("Avg confidence", f"{filtered.confidence.mean():.0%}" if len(filtered) else "—")

show_cols = [c for c in ["player_name","position","team","projected_ppr","floor","ceiling","recent_avg","trend","confidence","games_used"] if c in filtered.columns]
board = filtered[show_cols].head(100).copy()
board = board.rename(columns={"player_name":"Player","position":"Pos","team":"Team","projected_ppr":"Projection","floor":"Floor","ceiling":"Ceiling","recent_avg":"Recent Avg","trend":"Trend","confidence":"Confidence","games_used":"Games"})
if "Confidence" in board:
    board["Confidence"] = board["Confidence"].map(lambda x: f"{x:.0%}")
st.dataframe(board, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Player Detail")
names = filtered.player_name.tolist()
if names:
    selected = st.selectbox("Player", names)
    p = filtered[filtered.player_name == selected].iloc[0]
    a,b,c,d = st.columns(4)
    a.metric("Projection", f"{p.projected_ppr:.1f} PPR")
    b.metric("Floor", f"{p.floor:.1f}")
    c.metric("Ceiling", f"{p.ceiling:.1f}")
    d.metric("Confidence", f"{p.confidence:.0%}")
    direction = "📈" if p.trend > 0 else ("📉" if p.trend < 0 else "➡️")
    st.markdown(f"<div class='edge-card'><b>{selected}</b> · {p.team or '—'}<br>{direction} Recent trend: <b>{p.trend:+.1f}</b> PPR vs earlier games · Volatility: <b>{p.volatility:.1f}</b> · Games used: <b>{int(p.games_used)}</b></div>", unsafe_allow_html=True)

st.caption("Projection Engine v1 uses completed-game production with recency weighting. Opponent, injury, depth-chart and sportsbook layers will be added only after this baseline passes validation.")
