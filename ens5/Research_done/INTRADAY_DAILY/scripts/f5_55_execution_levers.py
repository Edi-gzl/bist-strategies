from _rp import _RP  # portable-path shim (added when vendoring)
"""
f5_55 — execution levers: rolling daily hedge, weekday return distribution, and
closing-print-quality conditioning (theoretical-matching / gap idea, realized full-sample proxy).
Reuses the EXACT f5_53/f5_52 loading so books stay identical to intraday_reversal_report.html.
Writes execution_levers.html (+ full_picks.csv with closing-quality signals).
All tables use FULL descriptive column names with legends (no shorthand).
"""
import io, base64, warnings
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
BG, FG = "#0e1116", "#c9d1d9"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#2b333c","axes.linewidth":0.6,
    "axes.titlesize":10.5,"axes.titleweight":"bold","figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
    "text.color":FG,"axes.labelcolor":FG,"xtick.color":"#8a93a0","ytick.color":"#8a93a0","grid.color":"#222932","legend.labelcolor":FG})
ACC,MUT,RED,BLU="#3fd99a","#7d8794","#e0795a","#6cb6ff"
PKG=Path(_RP(r"\Research_done\INTRADAY_DAILY")); OUT=Path(_RP(r"\Research_done\Fable-5\results"))
R15=Path(_RP(r"\Research_done\claude_data\results\F_intraday\03_residuals\15min\rank_l1"))
IS1=pd.Timestamp("2025-01-02"); LOCK=0.094; SIDES=2; NET5=SIDES*5.0/1e4
def sh(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    return x.mean()/x.std(ddof=1)*np.sqrt(252) if len(x)>20 and x.std(ddof=1)>0 else np.nan
def mdd(s): eq=s.cumsum(); return (eq.cummax()-eq).max()
def b64(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=120,bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
def clean(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.22,lw=0.5)

print("loading records ...")
recs=pd.read_parquet(OUT/"f5_15_daily_records.parquet"); recs["date"]=pd.to_datetime(recs["date"])
op=pd.read_parquet(OUT/"f5_44_open_prices.parquet"); op["date"]=pd.to_datetime(op["date"])
recs=recs.merge(op,on=["ticker","date"],how="left").sort_values(["ticker","date"])
recs["official"]=recs["p_auct"].where(recs["p_auct"].notna(),recs["p1800"])
recs["prev_official"]=recs.groupby("ticker")["official"].shift(1)
recs["next_p_auct"]=recs.groupby("ticker")["p_auct"].shift(-1)
recs["next_open"]=recs.groupby("ticker")["p_open"].shift(-1)
recs["next_date"]=recs.groupby("ticker")["date"].shift(-1)
adj=pd.read_parquet(_RP(r"\Research_done\claude_data\raw\bist50_prices_daily.parquet"))
A=adj.pivot_table(index="date",columns="ticker",values="close_adj"); A.index=pd.to_datetime(A.index)
mcap=adj.pivot_table(index="date",columns="ticker",values="mktcap_try"); mcap.index=pd.to_datetime(mcap.index)
# closing-quality panels (adjusted, intra-day consistent)
HI=adj.pivot_table(index="date",columns="ticker",values="high_adj"); HI.index=pd.to_datetime(HI.index)
LO=adj.pivot_table(index="date",columns="ticker",values="low_adj"); LO.index=pd.to_datetime(LO.index)
def cfac(row):
    tk=row.ticker
    if tk not in A.columns or not(np.isfinite(row.p_auct) and np.isfinite(row.next_p_auct) and row.p_auct>0): return 1.0
    a0=A[tk].asof(row.date); a1=A[tk].asof(row.next_date)
    if not(np.isfinite(a0) and np.isfinite(a1) and a0>0): return 1.0
    f=(a1/a0)/(row.next_p_auct/row.p_auct); return f if np.isfinite(f) and 0.05<f<20 else 1.0
recs["F"]=[cfac(r) for r in recs.itertuples()]
recs["r_on"]=(recs.next_open*recs.F)/recs.p_auct-1.0
recs["r_id"]=recs.next_p_auct/recs.next_open-1.0
recs["r_24"]=(recs.next_p_auct*recs.F)/recs.p_auct-1.0
recs["move"]=recs.p1800/recs.prev_official-1.0
recs["auction_pop"]=recs.p_auct/recs.p1759-1.0          # closing auction clear vs last continuous print (imbalance proxy)
recs["mc"]=[mcap[r.ticker].asof(r.date) if r.ticker in mcap.columns else np.nan for r in recs.itertuples()]
# close position within the day's range (adjusted) -> "closed at the high?"
cl=A.stack(future_stack=True).rename("close_adj"); hi=HI.stack(future_stack=True).rename("high_adj"); lo=LO.stack(future_stack=True).rename("low_adj")
rng=pd.concat([cl,hi,lo],axis=1).reset_index(); rng.columns=["date","ticker","close_adj","high_adj","low_adj"]
rng["close_to_high"]=rng.close_adj/rng.high_adj
rng["range_pos"]=(rng.close_adj-rng.low_adj)/(rng.high_adj-rng.low_adj)
recs=recs.merge(rng[["date","ticker","close_to_high","range_pos"]],on=["date","ticker"],how="left")

print("loading residual models ...")
def load_rank(mid):
    rr=pd.read_parquet(R15/f"{mid}_residuals.parquet",columns=["ticker","date","period","residual"])
    rr["date"]=pd.to_datetime(rr.date); rr=rr[(rr.period==30)&rr.residual.notna()]
    return rr.set_index(["ticker","date"])["residual"].groupby(level=1).rank(pct=True)
ranks={m:load_rank(m) for m in ["M05","BETA_126","M10","N02_mom61_only"]}
m15=pd.read_parquet(OUT/"f5_37_M15v2_residuals.parquet"); m15["date"]=pd.to_datetime(m15["date"])
ranks["M15v2"]=m15.set_index(["ticker","date"])["residual"].groupby(level=1).rank(pct=True)
ens=pd.concat(list(ranks.values()),axis=1)
ens=ens.mean(axis=1).where(ens.notna().sum(axis=1)>=3); ens.index.names=["ticker","date"]
ri=recs.set_index(["ticker","date"])
df=ri.join(ens.rename("s"),how="inner"); df=df[df.s.notna()&df.r_24.notna()&df.p_auct.notna()].reset_index()

# benchmarks
e50on=recs.dropna(subset=["r_on"]).groupby("date").r_on.mean()
e50id=recs.dropna(subset=["r_id"]).groupby("date").r_id.mean()
ew24=recs.dropna(subset=["r_24"]).groupby("date").r_24.mean()
for s in (e50on,e50id,ew24): s.index=pd.to_datetime(s.index)
e50on,e50id,ew24=(s[s.index>="2023-01-01"] for s in (e50on,e50id,ew24))

# book + FULL per-pick panel
N=3; rows_book=[]; picks=[]
for d,g in df.groupby("date"):
    if len(g)<2*N+2 or pd.Timestamp(d)<pd.Timestamp("2023-01-01"): continue
    srt=g.sort_values("s")
    no_buy=(srt.move>=LOCK).fillna(False); no_sell=(srt.move<=-LOCK).fillna(False)
    L=srt[~no_buy].head(N); S=srt[~no_sell].tail(N)
    if len(L)<2 or len(S)<2: continue
    rows_book.append((d,L.r_on.mean(),L.r_id.mean(),-S.r_on.mean(),-S.r_id.mean()))
    for leg,frame,sgn in (("LONG",L,1),("SHORT",S,-1)):
        for r in frame.itertuples():
            picks.append(dict(date=d,leg=leg,ticker=r.ticker,signal=r.s,move=r.move,auction_pop=r.auction_pop,
                close_to_high=r.close_to_high,range_pos=r.range_pos,
                r_on=r.r_on,r_id=r.r_id,r_24=r.r_24))
b=pd.DataFrame(rows_book,columns=["date","L_on","L_id","S_on","S_id"]).set_index("date").sort_index()
b["pm"]=0.5*b.L_on+0.5*b.S_id
P=pd.DataFrame(picks); P.to_csv(PKG/"full_picks.csv",index=False)
pmn=(b.pm-NET5).dropna()
print(f"reconcile PM top3 net5: IS {sh(pmn[pmn.index<=IS1]):+.2f} OOS {sh(pmn[pmn.index>IS1]):+.2f} (report OOS +3.01)")

# market series aligned
M24=ew24.reindex(b.index); Mon=e50on.reindex(b.index); Mid=e50id.reindex(b.index)

# ============== 1. ROLLING DAILY HEDGE vs STATIC ==============
def beta_series_static(s,m):
    d=pd.concat([s,m],axis=1).dropna(); return d.iloc[:,0].cov(d.iloc[:,1])/d.iloc[:,1].var()
bstat=beta_series_static(b.pm,M24)
def rolling_beta(s,m,win):
    d=pd.concat([s.rename("s"),m.rename("m")],axis=1).dropna()
    cov=d["s"].rolling(win,min_periods=max(20,win//2)).cov(d["m"])
    var=d["m"].rolling(win,min_periods=max(20,win//2)).var()
    bt=(cov/var).shift(1)                       # CAUSAL: estimate through T-1, apply on T
    return bt.reindex(s.index)
def hedged_net(beta_t):
    # daily-rebalanced index hedge: hedge return_t = -beta_t * market_t ; cost on book(2 sides)+hedge(|beta|*2 sides)
    bt=beta_t.reindex(b.index)
    if np.isscalar(bt) or isinstance(beta_t,float):
        bt=pd.Series(bstat,index=b.index)
    hedge=-bt*M24
    cost=NET5+bt.abs()*2*5/1e4
    return (b.pm+hedge-cost).dropna()
hedges={"unhedged (no hedge)":pmn,
        "static beta (full-sample, look-ahead)":hedged_net(pd.Series(bstat,index=b.index)),
        "rolling beta 21d (causal)":hedged_net(rolling_beta(b.pm,M24,21)),
        "rolling beta 42d (causal)":hedged_net(rolling_beta(b.pm,M24,42)),
        "rolling beta 63d (causal)":hedged_net(rolling_beta(b.pm,M24,63)),
        "rolling beta 126d (causal)":hedged_net(rolling_beta(b.pm,M24,126))}
worst20_idx=pmn.sort_values().head(20).index
def hedge_row(nm,s):
    return dict(**{"hedge scheme":nm},
        **{"Sharpe (in-sample)":f"{sh(s[s.index<=IS1]):+.2f}","Sharpe (out-of-sample)":f"{sh(s[s.index>IS1]):+.2f}",
           "Sharpe (full)":f"{sh(s):+.2f}","annualized return":f"{s.mean()*252:+.0%}","annualized volatility":f"{s.std()*np.sqrt(252):.0%}",
           "maximum drawdown":f"{mdd(s):.0%}","worst single day":f"{s.min()*100:+.1f}%",
           "mean of worst-20 days":f"{s.reindex(worst20_idx).mean()*100:+.2f}%"})
hedge_tbl=pd.DataFrame([hedge_row(k,v) for k,v in hedges.items()])
rb63=rolling_beta(b.pm,M24,63)

# ============== 2. WEEKDAY RETURN DISTRIBUTION (EW BIST50) ==============
DOW=["Monday","Tuesday","Wednesday","Thursday","Friday"]
def dist_block(series,label):
    out=[]
    s=series.dropna(); s.index=pd.to_datetime(s.index)
    for i in range(5):
        x=s[s.index.dayofweek==i]
        out.append(dict(**{"weekday":DOW[i],"phase":label,"days":len(x),
            "mean %":f"{x.mean()*100:+.3f}","median %":f"{x.median()*100:+.3f}","std %":f"{x.std()*100:.3f}",
            "share positive":f"{(x>0).mean():.0%}","5th pctile %":f"{x.quantile(.05)*100:+.2f}","95th pctile %":f"{x.quantile(.95)*100:+.2f}",
            "annualized %":f"{x.mean()*252*100:+.0f}"}))
    return out
wk_rows=dist_block(Mon,"overnight (close auction -> next open)")+dist_block(Mid,"intraday (open -> next close auction)")+dist_block(M24,"full day (24h, close -> next close)")
wk_tbl=pd.DataFrame(wk_rows)

# ============== 3. CLOSING-PRINT-QUALITY CONDITIONING (theoretical-matching / gap, realized proxy) ==============
# For SHORT picks (winners we short): does a STRONG close predict the short loses intraday next day?
#   short P&L intraday = -r_id ; "blowup" = winner keeps ripping intraday (r_id large +).
# For LONG picks (losers we buy): does a strong close predict CONTINUATION (so holding past the open captures extra r_id)?
S=P[P.leg=="SHORT"].dropna(subset=["r_id","close_to_high","auction_pop"]).copy()
L=P[P.leg=="LONG"].dropna(subset=["r_id","r_on","close_to_high","auction_pop"]).copy()
S["short_pnl_id"]=-S.r_id
# quartile conditioning on close_to_high and auction_pop for the SHORT leg intraday P&L
def quartile_cond(frame,sig,target,tlabel):
    f=frame.dropna(subset=[sig,target]).copy()
    f["q"]=pd.qcut(f[sig],4,labels=["Q1 (weakest close)","Q2","Q3","Q4 (closed strongest)"],duplicates="drop")
    g=f.groupby("q",observed=True)[target]
    return pd.DataFrame([dict(**{"signal quartile":str(q),"n picks":int(g.size()[q]),
        "mean "+tlabel+" %":f"{g.mean()[q]*100:+.3f}","median %":f"{g.median()[q]*100:+.3f}","share positive":f"{(f[f.q==q][target]>0).mean():.0%}"}) for q in g.size().index])
short_cth=quartile_cond(S,"close_to_high","short_pnl_id","short intraday P&L")
short_pop=quartile_cond(S,"auction_pop","short_pnl_id","short intraday P&L")
long_cth_extra=quartile_cond(L,"close_to_high","r_id","extra intraday return if held past open")

# Skip-short filter feasibility: skip shorts that "closed strong" (top decile close_to_high OR auction_pop>thr)
def skip_short_eval(thr_cth=0.995, thr_pop=0.01):
    s=S.copy()
    s["closed_strong"]= (s.close_to_high>=thr_cth) | (s.auction_pop>=thr_pop)
    blow=s.r_id>0.02   # winner ripped >2% intraday = bad short (event to avoid)
    tp=int((s.closed_strong & blow).sum()); fp=int((s.closed_strong & ~blow).sum())
    fn=int((~s.closed_strong & blow).sum()); tn=int((~s.closed_strong & ~blow).sum())
    prec= tp/(tp+fp) if tp+fp else np.nan; rec= tp/(tp+fn) if tp+fn else np.nan
    base_pnl=s.short_pnl_id.mean(); filt_pnl=s[~s.closed_strong].short_pnl_id.mean(); skipped=s[s.closed_strong].short_pnl_id.mean()
    return dict(threshold=f"close_to_high>={thr_cth} or auction_pop>={thr_pop:.0%}",
        skipped=int(s.closed_strong.sum()), kept=int((~s.closed_strong).sum()),
        precision=f"{prec:.0%}", recall=f"{rec:.0%}",
        mean_short_pnl_all=f"{base_pnl*100:+.3f}%", mean_short_pnl_kept=f"{filt_pnl*100:+.3f}%", mean_short_pnl_skipped=f"{skipped*100:+.3f}%",
        TP=tp,FP=fp,FN=fn,TN=tn)
skip_rows=pd.DataFrame([skip_short_eval(0.999,0.015),skip_short_eval(0.995,0.010),skip_short_eval(0.990,0.005)])

# market-level skip: when EW50 itself closed strong on T (mean close_to_high across names), does the SHORT leg do worse next day?
mkt_cth=P.dropna(subset=["close_to_high"]).groupby("date").close_to_high.mean()
short_leg_id=(-b.S_id)  # NOTE b.S_id already = -S.r_id.mean(); short leg intraday P&L = b.S_id
shortleg=b.S_id.copy()
mk=pd.concat([mkt_cth.rename("mkt_close_strength"),shortleg.rename("short_leg_intraday_pnl")],axis=1).dropna()
mk["tercile"]=pd.qcut(mk.mkt_close_strength,3,labels=["market closed weak","market closed mid","market closed strong"])
mkt_tbl=pd.DataFrame([dict(**{"market close strength (T)":str(t),"days":int((mk.tercile==t).sum()),
    "mean short-leg intraday P&L %":f"{mk[mk.tercile==t].short_leg_intraday_pnl.mean()*100:+.3f}",
    "Sharpe of short leg":f"{sh(mk[mk.tercile==t].short_leg_intraday_pnl):+.2f}"}) for t in ["market closed weak","market closed mid","market closed strong"]])

# ============== PLOTS ==============
imgs={}
# rolling vs static beta
fig,axes=plt.subplots(1,2,figsize=(11,3.3))
axes[0].plot(rb63.index,rb63.values,color=ACC,lw=1.3,label="rolling 63d beta (causal)")
axes[0].axhline(bstat,color=BLU,ls="--",lw=1,label=f"static full-sample beta = {bstat:+.2f}")
axes[0].axhline(0,color="#3a424c",lw=.6); axes[0].axvline(IS1,color="#2b333c",lw=1)
axes[0].set_title("phase-matched beta vs EW BIST50 — rolling vs static"); axes[0].legend(fontsize=7,frameon=False); clean(axes[0])
for nm,col in [("unhedged (no hedge)",MUT),("static beta (full-sample, look-ahead)",BLU),("rolling beta 63d (causal)",ACC)]:
    axes[1].plot(hedges[nm].cumsum().index,hedges[nm].cumsum().values,lw=1.4,color=col,label=nm)
axes[1].axvline(IS1,color="#2b333c",lw=1); axes[1].set_title("cumulative net5 P&L — hedge schemes"); axes[1].legend(fontsize=6.5,frameon=False); clean(axes[1])
imgs["hedge"]=b64(fig)
# weekday distribution box-ish (mean bars on/id/24)
fig,ax=plt.subplots(figsize=(11,3.2)); x=np.arange(5); w=0.26
for k,(ser,col,lbl) in enumerate([(Mon,ACC,"overnight"),(Mid,RED,"intraday"),(M24,BLU,"full day")]):
    vals=[ser[pd.to_datetime(ser.index).dayofweek==i].mean()*100 for i in range(5)]
    ax.bar(x+(k-1)*w,vals,w,color=col,label=lbl)
ax.set_xticks(x); ax.set_xticklabels(DOW); ax.axhline(0,color="#3a424c",lw=.6)
ax.set_title("EW BIST50 mean return by weekday & phase (%)"); ax.legend(frameon=False,fontsize=8); clean(ax)
imgs["weekday"]=b64(fig)
# short-leg blowup vs close strength
fig,axes=plt.subplots(1,2,figsize=(11,3.3))
axes[0].scatter(S.close_to_high,S.r_id*100,s=6,alpha=.25,color=RED)
axes[0].axhline(2,color=ACC,ls="--",lw=.8,label="+2% = short blows up"); axes[0].axhline(0,color="#3a424c",lw=.5)
axes[0].set_xlabel("close / day-high  (1.0 = closed at high)"); axes[0].set_ylabel("next-day intraday return of shorted name %")
axes[0].set_title("SHORT picks: does closing at the high predict the short ripping?"); axes[0].legend(fontsize=7,frameon=False); clean(axes[0])
axes[1].scatter(L.close_to_high,L.r_id*100,s=6,alpha=.25,color=ACC); axes[1].axhline(0,color="#3a424c",lw=.5)
axes[1].set_xlabel("close / day-high (1.0 = closed at high)"); axes[1].set_ylabel("extra intraday return if long held past open %")
axes[1].set_title("LONG picks: closing strong vs continuation past the open"); clean(axes[1])
imgs["closequal"]=b64(fig)

# ============== HTML ==============
def htbl(d): return d.to_html(index=False,border=0,escape=False)
html=f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>ENS5v2 — execution levers</title><style>
body{{font-family:'Segoe UI',system-ui,sans-serif;margin:46px auto;max-width:1180px;color:#c9d1d9;background:#0e1116;padding:0 24px}}
h1{{font-size:22px;margin-bottom:2px}} .sub{{color:#7d8794;font-size:13px;margin-bottom:22px;line-height:1.55}}
h2{{font-size:12px;text-transform:uppercase;letter-spacing:.13em;color:#7d8794;margin:40px 0 12px;border-bottom:1px solid #1f262e;padding-bottom:6px}}
table{{border-collapse:collapse;font-size:12px;width:100%;margin:10px 0;background:#0c1014;border:1px solid #1f262e;border-radius:10px;overflow:hidden}}
td,th{{padding:8px 12px;border-bottom:1px solid #1a212a;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
tr:first-child th{{background:#12171e;border-bottom:1px solid #2b333c}}
th{{color:#8a93a0;font-weight:600;font-size:10px;text-transform:none;letter-spacing:.02em}}
td:first-child,th:first-child{{text-align:left;color:#c9d1d9}} tr:hover td{{background:#141b23}}
.note{{background:#161b22;border-left:3px solid #3fd99a;padding:10px 14px;margin:13px 0;font-size:12.7px;line-height:1.55}}
.note.warn{{border-left-color:#e0795a}} .lgnd{{color:#7d8794;font-size:11.5px;margin:4px 0}} code{{color:#3fd99a;font-size:11.5px}} img{{max-width:100%;margin:6px 0}}
</style></head><body>
<h1>ENS5v2 — execution levers</h1>
<div class='sub'>Subject: phase-matched top3 (0.5 long held overnight + 0.5 short held intraday). Net = 5 bps/side, 2 sides/day.
Books identical to intraday_reversal_report.html (reconciliation: PM top3 out-of-sample Sharpe net 5bps =
<code>{sh(pmn[pmn.index>IS1]):+.2f}</code>, report shows +3.01). Full descriptive column names throughout.</div>

<h2>1 · rolling daily hedge vs static full-sample beta</h2>
<div class='note'>The report's beta-hedge used a single <b>full-sample</b> beta (uses the whole history at once — look-ahead, not deployable).
Here the hedge beta is re-estimated every day from a <b>trailing causal window</b> (cov/var through T−1, applied on T), then a
daily-rebalanced index leg of size −beta&times;EW BIST50 is added. Hedge cost = book (2 sides) + hedge notional
(|beta|&times;2 sides) at 5 bps/side. Static full-sample beta of the phase-matched book = <code>{bstat:+.2f}</code>.</div>
{htbl(hedge_tbl)}
<img src='data:image/png;base64,{imgs["hedge"]}'>
<div class='lgnd'>Column key: in-sample = through 2024 · out-of-sample = 2025+ · "mean of worst-20 days" uses the unhedged book's worst-20 dates,
so each scheme is judged on the SAME calendar days.</div>

<h2>2 · EW BIST50 return distribution by weekday and phase</h2>
<div class='note'>Daily biases in the market itself. Phases: <b>overnight</b> (close auction T → next open), <b>intraday</b>
(open → next close auction), <b>full day</b> (24h). If a weekday has a systematic overnight or intraday drift, that interacts
with our net-short-intraday book (which profits when the session falls and bleeds when it rises).</div>
{htbl(wk_tbl)}
<img src='data:image/png;base64,{imgs["weekday"]}'>

<h2>3 · closing-print quality — theoretical-matching / gap conditioning (realized, full sample)</h2>
<div class='note'>Tests the hypothesis: a name that <b>closed strong</b> (at/near the day's high, or with a positive closing-auction pop)
carries momentum into the next session. Two signals available at the closing auction on day T:
<code>close_to_high</code> = closing price ÷ day's high (1.0 = closed exactly at the high);
<code>auction_pop</code> = closing-auction price ÷ last continuous print at 17:59 − 1 (how much the auction itself jumped, an imbalance proxy).
For <b>shorts</b> (we short winners) the risk is the winner keeps ripping intraday (a "blowup", the worst-day driver).
For <b>longs</b> (we buy losers) a strong close may signal continuation, so holding past the open could capture extra return.</div>

<div class='lgnd'><b>3a — short leg intraday P&amp;L by closing strength quartile</b> (Q4 = names that closed strongest; more negative = worse for the short):</div>
{htbl(short_cth)}
<div class='lgnd'><b>3b — short leg intraday P&amp;L by closing-auction pop quartile</b> (Q4 = biggest upward auction jump):</div>
{htbl(short_pop)}
<div class='lgnd'><b>3c — long leg: extra intraday return if held past the open, by closing strength quartile</b> (this is the "don't exit at MOC, hold later" payoff for longs):</div>
{htbl(long_cth_extra)}
<img src='data:image/png;base64,{imgs["closequal"]}'>

<div class='note'><b>3d — skip-short filter: false-positive / false-negative accounting.</b> Rule: skip a short if the name "closed strong"
(close at/above threshold of the high, OR a positive auction pop above threshold). Event to avoid ("blowup") = shorted winner
rises &gt;2% intraday. Precision = of shorts we skip, fraction that were genuine blowups. Recall = of all blowups, fraction we skip.
Compare mean short-leg intraday P&amp;L for kept vs skipped names.</div>
{htbl(skip_rows)}

<div class='note'><b>3e — market-level skip ("if the market is jumping, don't short").</b> Group days by how strongly the
whole EW BIST50 closed on day T (mean close_to_high across names), and look at the next-day short-leg intraday P&amp;L.</div>
{htbl(mkt_tbl)}

<div class='note warn'><b>Data note on the live (theoretical) version:</b> the signals above use the <i>realized</i> closing print
(known the instant the auction uncrosses). The <i>indicative</i> equilibrium price during the closing-auction call phase
(<code>equilibrium_price</code> in the BBO feed, state P_KAPANIS_EMIR_TPL) is observable a few minutes <b>before</b> uncross and is
what makes a live hold/skip decision possible — but the tick data only covers 2024-11 → 2026-04. f5_56 validates that the indicative
price tracks the realized close on a sample, i.e. the rule is actually tradeable live.</div>
</body></html>"""
(PKG/"execution_levers.html").write_text(html,encoding="utf-8")
print("WROTE execution_levers.html size",len(html)//1024,"KB images",html.count("base64,"))
print("\nHEDGE TABLE:\n",hedge_tbl.to_string(index=False))
print("\nSHORT close_to_high quartiles:\n",short_cth.to_string(index=False))
print("\nLONG hold-later extra:\n",long_cth_extra.to_string(index=False))
print("\nSKIP-SHORT eval:\n",skip_rows[["threshold","skipped","kept","precision","recall","mean_short_pnl_all","mean_short_pnl_kept","mean_short_pnl_skipped"]].to_string(index=False))
print("\nMARKET-level skip:\n",mkt_tbl.to_string(index=False))
print("\nWEEKDAY (full-day phase only):\n",wk_tbl[wk_tbl.phase.str.startswith('full')].to_string(index=False))
