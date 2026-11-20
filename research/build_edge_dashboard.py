"""Build static, read-only NFL Edge dashboard from verified research snapshots."""
from __future__ import annotations
import argparse,html,json
from pathlib import Path
import pandas as pd

def build(snapshot_dir,output_dir):
    src=Path(snapshot_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    reports=sorted(src.glob("moneyline_snapshot_*.csv"))
    if not reports:raise ValueError("Missing sportsbook snapshot")
    q=pd.read_csv(reports[-1])
    if q.empty:raise ValueError("No sportsbook quotes")
    if q.game_id.duplicated().all() and len(q)<3:raise ValueError("Too few quotes")
    generated=str(q.retrieved_at_utc.iloc[0])
    games=[]
    for game,g in q.groupby("game_id",sort=True):
        p=float(g.model_home_win_probability.iloc[0])
        fresh=g.copy()
        now=pd.to_datetime(fresh.retrieved_at_utc,utc=True,errors="coerce")
        update=pd.to_datetime(fresh.market_last_update_utc,utc=True,errors="coerce")
        kickoff=pd.to_datetime(fresh.kickoff_utc,utc=True,errors="coerce")
        fresh=fresh.loc[update.notna()&(update<=now)&((now-update)<=pd.Timedelta(minutes=90))&(now<kickoff)]
        market=float(fresh.home_no_vig_probability.median()) if len(fresh)>=3 else None
        games.append({"id":str(game),"away":str(g.away_team.iloc[0]),"home":str(g.home_team.iloc[0]),
          "kickoff":str(g.kickoff_utc.iloc[0]),"home_probability":p,
          "market_home_probability":market,"fresh_books":int(fresh.sportsbook.nunique()),
          "best_home_moneyline":int(g.loc[g.home_ev_per_unit.idxmax(),"home_moneyline_american"]),
          "best_away_moneyline":int(g.loc[g.away_ev_per_unit.idxmax(),"away_moneyline_american"]),
          "best_home_book":str(g.loc[g.home_ev_per_unit.idxmax(),"sportsbook_title"]),
          "best_away_book":str(g.loc[g.away_ev_per_unit.idxmax(),"sportsbook_title"])})
    extras=sorted(src.glob("spread_total_snapshot_*.csv"))
    additional=pd.read_csv(extras[-1]) if extras else pd.DataFrame()
    for item in games:
        item["spread_quotes"]=[]
        item["total_quotes"]=[]
        if additional.empty:continue
        subset=additional.loc[additional.game_id==item["id"]].copy()
        if subset.empty:continue
        now=pd.to_datetime(subset.retrieved_at_utc,utc=True,errors="coerce")
        updated=pd.to_datetime(subset.market_last_update_utc,utc=True,errors="coerce")
        subset=subset.loc[updated.notna()&(updated<=now)&((now-updated)<=pd.Timedelta(minutes=90))]
        for kind,target in (("spreads","spread_quotes"),("totals","total_quotes")):
            rows=subset.loc[subset.market==kind]
            item[target]=[{"book":str(x.sportsbook_title),"selection":str(x.selection),
               "line":float(x.point),"odds":int(x.american_odds)} for x in rows.itertuples(index=False)]
    data={"updated_at":generated,"games":games,"note":"Frozen research probabilities; sportsbook quotes are snapshots, not live executable prices. No betting recommendation."}
    (out/"data.json").write_text(json.dumps(data,indent=2))
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL Edge | Research Dashboard</title><meta name="theme-color" content="#101b30">
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#0a1020;color:#f0f4fc}
*{box-sizing:border-box}body{margin:0}main{max-width:1000px;margin:auto;padding:24px 16px 72px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
h1{font-size:clamp(28px,6vw,42px);letter-spacing:-1px;margin:4px 0}h2{font-size:18px;margin:0}
.muted{color:#a6b6ce}.tag{color:#9ce2d5;font-size:12px;letter-spacing:2px;font-weight:700}
.notice{background:#15233a;border:1px solid #32435d;padding:14px;border-radius:12px;line-height:1.5;margin:22px 0}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:18px 0 26px}
.stat,.game{background:#111d31;border:1px solid #293a52;border-radius:14px;padding:16px}
.stat strong{font-size:23px;display:block}.stat span{font-size:12px;color:#a6b6ce}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:12px}
.match{font-size:21px;font-weight:750;margin:10px 0}.game small{color:#a6b6ce}
.bar{display:flex;height:10px;border-radius:8px;overflow:hidden;background:#30435c;margin:14px 0}
.bar>span{background:#42c9aa;display:block}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.metric{background:#0b1526;border-radius:10px;padding:11px}.metric b{font-size:21px;display:block;margin-top:5px}
.delta{margin-top:12px;font-weight:650}.warn{color:#f4c479}.good{color:#9ce2d5}
footer{font-size:12px;color:#a6b6ce;margin-top:26px;line-height:1.6}
@media(max-width:400px){.stats{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main><header><div><div class="tag">RESEARCH / NFL EDGE</div><h1>NFL Edge</h1><div class="muted">Pregame probability tracker</div></div><div class="muted" id="updated">Loading snapshot…</div></header>
<div class="notice">Model predictions are frozen for evaluation. Sportsbook prices are periodic snapshots, not live executable quotes. Differences do not establish a profitable betting edge.</div>
<section class="stats"><div class="stat"><strong id="count">—</strong><span>Upcoming matchups</span></div><div class="stat"><strong id="books">—</strong><span>Games with fresh market consensus</span></div><div class="stat"><strong id="largest">—</strong><span>Largest model-market gap</span></div></section>
<div class="cards" id="games"><p>Loading games…</p></div>
<footer>Research-only, no wagers. Model win probabilities are fixed to the archived cohort; market consensus uses the median no-vig home win probability across at least three books with updates within 90 minutes of snapshot retrieval. Games shown reflect the latest snapshot, which can omit already-started games.</footer></main>
<script>
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=n=>(n*100).toFixed(1)+'%';
const when=s=>new Date(s).toLocaleString(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'});
fetch('data.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Data unavailable');return r.json()}).then(data=>{
 document.getElementById('updated').textContent='Odds snapshot: '+when(data.updated_at);
 const games=data.games.slice().sort((a,b)=>new Date(a.kickoff)-new Date(b.kickoff));
 document.getElementById('count').textContent=games.length;
 document.getElementById('books').textContent=games.filter(g=>g.market_home_probability!==null).length;
 const gaps=games.filter(g=>g.market_home_probability!==null).map(g=>Math.abs(g.home_probability-g.market_home_probability));
 document.getElementById('largest').textContent=gaps.length?(Math.max(...gaps)*100).toFixed(1)+' pts':'—';
 document.getElementById('games').innerHTML=games.map(g=>{
 const home=g.home_probability>=.5,team=home?g.home:g.away,prob=home?g.home_probability:1-g.home_probability;
 const m=g.market_home_probability,delta=m===null?null:(g.home_probability-m)*100;
 const lines=(arr,kind)=>{
   if(!arr?.length)return '<small>No fresh '+kind+' quotes available</small>';
   const byBook={};
   for(const x of arr){if(!byBook[x.book])byBook[x.book]=[];byBook[x.book].push(x);}
   return Object.entries(byBook).slice(0,4).map(([book,quotes])=>
     '<div style="padding:8px 0;border-bottom:1px solid #293a52"><small>'+esc(book)+'</small><div>'+
     quotes.map(x=>esc(x.selection)+' '+(x.line>0?'+':'')+x.line+' ('+(x.odds>0?'+':'')+x.odds+')').join(' · ')+'</div></div>').join('');
 };

 return '<article class="game"><small>'+esc(when(g.kickoff))+' · '+esc(g.fresh_books)+' fresh books</small><div class="match">'+esc(g.away)+' @ '+esc(g.home)+'</div><small>EDGE PROJECTED WINNER</small><h2>'+esc(team)+' · '+pct(prob)+'</h2><div class="bar"><span style="width:'+Math.round(g.home_probability*100)+'%"></span></div><div class="grid"><div class="metric"><small>Edge · home</small><b>'+pct(g.home_probability)+'</b></div><div class="metric"><small>Market · home</small><b>'+(m===null?'Unavailable':pct(m))+'</b></div></div><div class="delta '+(delta!==null&&Math.abs(delta)>=15?'warn':'good')+'">'+(delta===null?'Not enough fresh quotes':'Edge minus market (home): '+(delta>0?'+':'')+delta.toFixed(1)+' pts')+'</div><details style="margin-top:16px"><summary style="cursor:pointer;font-weight:650">Point spreads &amp; over/under</summary><div style="margin-top:12px"><b>Point spreads</b>'+lines(g.spread_quotes,'spread')+'<div style="margin-top:12px"><b>Game totals (O/U)</b>'+lines(g.total_quotes,'total')+'</div><small style="display:block;margin-top:12px">Sportsbook lines only. NFL Edge has not validated spread or total probabilities.</small></div></details></article>';
 }).join('');
}).catch(e=>document.getElementById('games').textContent='Unable to load the current research snapshot.');
</script></body></html>'''
    (out/"index.html").write_text(page)
    (out/".nojekyll").write_text("")
    print(json.dumps({"games":len(games),"snapshot":reports[-1].name,"output":str(out)}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--snapshot-dir",required=True);p.add_argument("--output-dir",required=True)
    a=p.parse_args();build(a.snapshot_dir,a.output_dir)
