from __future__ import annotations

import pandas as pd
import streamlit as st

from player_engine import normalize_player_schema, project_all_players_as_of, player_history_as_of
from player_validation import validate_player_feed
from player_backtest import backtest_fantasy_projections, fantasy_backtest_summary, fantasy_backtest_by_position
try:
    from player_backtest import qb_model_audit
except ImportError:
    def qb_model_audit(results: pd.DataFrame) -> dict:
        return {"n": 0, "deployment_pending": True}

PLAYER_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
BUILD_VERSION = "v1.3.0-qb-v3-championship"
QB_MODEL_VERSION = "QB Engine v3 • volume-stability challenger"
BACKTEST_CACHE_VERSION = "qb-v3-vs-legacy-championship-v1"
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")

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


def fantasy_only(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "position" not in df.columns:
        return pd.DataFrame() if df is None else df.copy()
    d = df.copy()
    d["position"] = d["position"].astype(str).str.upper().str.strip()
    return d[d["position"].isin(FANTASY_POSITIONS)].copy()


@st.cache_data(ttl=900, show_spinner=False)
def run_backtest(stats: pd.DataFrame, season: int, week: int, lookback: int, min_games: int, cache_version: str) -> pd.DataFrame:
    _ = cache_version
    fantasy_stats = fantasy_only(stats)
    end_week = min(max(int(week) - 1, 4), 18)
    start_season = max(2022, int(season) - 1)
    results = backtest_fantasy_projections(
        fantasy_stats,
        start_season=start_season,
        end_season=int(season),
        start_week=4,
        end_week=end_week,
        lookback=lookback,
        min_games=min_games,
    )
    return fantasy_only(results)

st.title("🏈 NFL EDGE")
st.markdown(f'<div class="small">Player Projection Engine v1 • Leakage-safe weekly validation build • <b>{BUILD_VERSION}</b><br><b>{QB_MODEL_VERSION}</b></div>', unsafe_allow_html=True)
st.info("This validation dashboard projects a selected NFL week using only games completed before that week. Game Engine 1.0 remains protected in nfl_edge.py.")

current_season = 2026
with st.sidebar:
    st.header("Projection Settings")
    st.caption(f"Live build: {BUILD_VERSION}")
    st.caption(QB_MODEL_VERSION)
    season = st.selectbox("Season", list(range(current_season, 2019, -1)), index=0)
    week = st.slider("Project week", 1, 22, 8)
    lookback = st.slider("Recent games used", 3, 12, 8)
    min_games = st.slider("Minimum prior games", 2, 8, 3)

with st.spinner("Loading nflverse player data..."):
    stats = load_player_stats(max(2020, season - 1), season)

health = validate_player_feed(stats)
if not health["ok"]:
    missing = ", ".join(health["missing"]) if health["missing"] else "unknown feed issue"
    st.error(f"Player data feed is not projection-ready: {missing}.")
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

positions = sorted([str(x) for x in rankings.get("position", pd.Series(dtype=str)).dropna().unique() if str(x)])
team_values = sorted([str(x) for x in rankings.get("team", pd.Series(dtype=str)).dropna().unique() if str(x)])

st.caption(f"As-of cutoff: {season} Week {week} • {health['rows']:,} feed rows • {health['teams']} normalized teams • future/current-week results excluded")

c1, c2, c3 = st.columns([1,1,2])
position = c1.selectbox("Position", ["ALL"] + positions)
team = c2.selectbox("Team", ["ALL"] + team_values)
search = c3.text_input("Player search", placeholder="e.g. Drake Maye")

filtered = rankings.copy()
if position != "ALL" and "position" in filtered.columns:
    filtered = filtered[filtered["position"] == position]
if team != "ALL":
    filtered = filtered[filtered["team"] == team]
if search.strip():
    filtered = filtered[filtered["player_name"].str.contains(search.strip(), case=False, na=False)]

st.subheader(f"Fantasy Projection Board — {season} Week {week}")
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
    st.markdown(f"<div class='edge-card'><b>{selected}</b> · {p.position or '—'} · {p.team or '—'}<br>{direction} Recent trend: <b>{p.trend:+.1f}</b> PPR vs earlier games · Volatility: <b>{p.volatility:.1f}</b> · Prior games used: <b>{int(p.games_used)}</b></div>", unsafe_allow_html=True)
else:
    st.info("No players match the selected filters.")

st.divider()
st.subheader("Historical Accuracy Gate")
st.caption(f"Build {BUILD_VERSION} • {QB_MODEL_VERSION} • Fantasy-only validation: QB / RB / WR / TE. Walk-forward projections only see information available before each target week.")
if week < 5:
    st.info("Historical accuracy becomes available from Week 5 onward so at least one in-season validation week exists.")
else:
    if st.button("Run QB v3 vs legacy historical championship", type="secondary"):
        with st.spinner("Replaying QB v3 and legacy QB projections on identical historical samples..."):
            results = run_backtest(stats, season, week, lookback, min_games, BACKTEST_CACHE_VERSION)
        if results.empty:
            st.warning("No historical projection matches were available for this setup.")
        else:
            positions_in_results = sorted(set(results["position"].dropna().astype(str))) if "position" in results.columns else []
            invalid_positions = sorted(set(positions_in_results) - set(FANTASY_POSITIONS))
            if invalid_positions:
                st.error(f"Validation guard blocked non-fantasy positions: {', '.join(invalid_positions)}")
                st.stop()
            st.caption(f"Validated positions in this run: {', '.join(positions_in_results)} • QB model: {QB_MODEL_VERSION}")
            summary = fantasy_backtest_summary(results)
            a,b,c,d,e = st.columns(5)
            a.metric("Fantasy samples", f"{summary['n']:,}")
            b.metric("MAE", f"{summary['mae']:.2f}")
            c.metric("Correlation", f"{summary['correlation']:.2f}" if pd.notna(summary['correlation']) else "—")
            d.metric("Range coverage", f"{summary['interval_coverage']:.0%}" if pd.notna(summary['interval_coverage']) else "—")
            e.metric("Reliability", summary["tier"])
            if summary["tier"] == "STRONG":
                st.success("Fantasy-only historical projection performance clears the STRONG research gate for this sample.")
            elif summary["tier"] == "USABLE":
                st.warning("Fantasy-only historical projection performance clears the USABLE research gate, but it is not a betting-edge claim.")
            else:
                st.error("Fantasy-only historical performance remains in RESEARCH status. NFL Edge should not treat these projections as a proven edge.")
            by_position = fantasy_backtest_by_position(results)
            if not by_position.empty:
                display = by_position[[c for c in ["position","n","mae","rmse","bias","correlation","nmae","interval_coverage","tier"] if c in by_position.columns]].copy()
                display = display.rename(columns={"position":"Pos","n":"Samples","mae":"MAE","rmse":"RMSE","bias":"Bias","correlation":"Correlation","nmae":"Normalized MAE","interval_coverage":"Range Coverage","tier":"Reliability"})
                st.dataframe(display.round(3), use_container_width=True, hide_index=True)

            st.subheader("QB v3 vs Legacy Championship")
            audit = qb_model_audit(results)
            if audit.get("n", 0):
                a,b,c,d,e = st.columns(5)
                a.metric("QB samples", f"{audit['n']:,}")
                b.metric("QB v3 MAE", f"{audit['v2_mae']:.3f}")
                c.metric("Legacy QB MAE", f"{audit['legacy_mae']:.3f}")
                d.metric("Avg projection Δ", f"{audit['mean_abs_projection_delta']:.3f}")
                e.metric("Predictions changed", f"{audit['changed_predictions']}/{audit['n']}")
                st.caption(f"Largest absolute QB projection change: {audit['max_abs_projection_delta']:.3f} PPR points.")
                if audit['changed_predictions'] == 0:
                    st.error("Audit failure: QB v3 did not change any historical QB predictions versus legacy.")
                elif audit['v2_mae'] < audit['legacy_mae']:
                    st.success(f"QB v3 BEATS the legacy champion by {audit['legacy_mae'] - audit['v2_mae']:.3f} PPR MAE on identical QB samples.")
                elif audit['v2_mae'] > audit['legacy_mae']:
                    st.warning(f"QB v3 changed predictions but trails legacy by {audit['v2_mae'] - audit['legacy_mae']:.3f} PPR MAE.")
                else:
                    st.info("QB v3 changed predictions, but aggregate MAE is tied with legacy on this sample.")
            elif audit.get("deployment_pending"):
                st.warning("QB audit helper is still syncing on the deployment. Core dashboard remains available; reboot after the latest deploy completes.")
            else:
                st.warning("QB audit could not form a matched v3-versus-legacy sample.")

st.caption(f"Live build: {BUILD_VERSION} • {QB_MODEL_VERSION}. Projection Engine v1 is a validation layer, not a betting-edge claim. Opponent, injury, depth-chart, market-price and vig-aware validation remain gated behind model testing.")
