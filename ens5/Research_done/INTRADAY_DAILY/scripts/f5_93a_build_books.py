from _rp import _RP  # portable-path shim (added when vendoring)
"""f5_93a — rebuild full top3 & top5 daily books (reuse f5_86 engine), save with all columns for the report."""
import numpy as np, pandas as pd
from pathlib import Path
OUT=Path(_RP(r"\Research_done\Fable-5\results")); R15=Path(_RP(r"\Research_done\claude_data\results\F_intraday\03_residuals\15min\rank_l1"))
PKG=Path(_RP(r"\Research_done\INTRADAY_DAILY")); LOCK=0.094
print("loading records + residuals ...")
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
def cfac(r):
    tk=r.ticker
    if tk not in A.columns or not(np.isfinite(r.p_auct) and np.isfinite(r.next_p_auct) and r.p_auct>0): return 1.0
    a0=A[tk].asof(r.date); a1=A[tk].asof(r.next_date)
    if not(np.isfinite(a0) and np.isfinite(a1) and a0>0): return 1.0
    f=(a1/a0)/(r.next_p_auct/r.p_auct); return f if np.isfinite(f) and 0.05<f<20 else 1.0
recs["F"]=[cfac(r) for r in recs.itertuples()]
recs["r_on"]=(recs.next_open*recs.F)/recs.p_auct-1.0
recs["r_id"]=recs.next_p_auct/recs.next_open-1.0
recs["r_24"]=(recs.next_p_auct*recs.F)/recs.p_auct-1.0
recs["move"]=recs.p1800/recs.prev_official-1.0
def load_rank(mid):
    rr=pd.read_parquet(R15/f"{mid}_residuals.parquet",columns=["ticker","date","period","residual"])
    rr["date"]=pd.to_datetime(rr.date); rr=rr[(rr.period==30)&rr.residual.notna()]
    return rr.set_index(["ticker","date"])["residual"].groupby(level=1).rank(pct=True)
rk=pd.concat([load_rank("M05"),load_rank("BETA_126"),load_rank("M10"),load_rank("N02_mom61_only")],axis=1)
m15=pd.read_parquet(OUT/"f5_37_M15v2_residuals.parquet"); m15["date"]=pd.to_datetime(m15["date"])
rk=pd.concat([rk,m15.set_index(["ticker","date"])["residual"].groupby(level=1).rank(pct=True)],axis=1)
rk.columns=["M05","BETA_126","M10","N02","M15v2"]
ens=rk.mean(axis=1).where(rk.notna().sum(axis=1)>=3); ens.index.names=["ticker","date"]
df=recs.set_index(["ticker","date"]).join(ens.rename("s"),how="inner")
df=df[df.s.notna()&df.r_24.notna()&df.p_auct.notna()].reset_index()
ew_on=recs.dropna(subset=["r_on"]).groupby("date").r_on.mean(); ew_on.index=pd.to_datetime(ew_on.index)
ew24=recs.dropna(subset=["r_24"]).groupby("date").r_24.mean(); ew24.index=pd.to_datetime(ew24.index)
breadth=recs.dropna(subset=["r_on"]).groupby("date").r_on.apply(lambda x:(x>0).mean()); breadth.index=pd.to_datetime(breadth.index)
disp=df.groupby("date").r_24.std()
def build(N):
    out=[]
    for d,g in df.groupby("date"):
        if len(g)<2*N+2 or pd.Timestamp(d)<pd.Timestamp("2023-01-01"): continue
        srt=g.sort_values("s"); L=srt[~(srt.move>=LOCK).fillna(False)].head(N); S=srt[~(srt.move<=-LOCK).fillna(False)].tail(N)
        if len(L)<2 or len(S)<2: continue
        out.append((d,L.r_on.mean(),L.r_id.mean(),-S.r_on.mean(),-S.r_id.mean()))
    b=pd.DataFrame(out,columns=["date","L_on","L_id","S_on","S_id"]).set_index("date").sort_index()
    b["pm"]=0.5*b.L_on+0.5*b.S_id; b["lon"]=0.5*b.L_on; b["sid"]=0.5*b.S_id
    b["spec"]=0.5*(b.L_on+b.L_id)+0.5*(b.S_on+b.S_id)
    b["gap"]=ew_on.reindex(b.index); b["mkt"]=ew24.reindex(b.index); b["breadth"]=breadth.reindex(b.index); b["disp"]=disp.reindex(b.index)
    return b
for N in (3,5):
    b=build(N); b.to_parquet(PKG/"data"/f"book_top{N}.parquet")
    print(f"saved book_top{N}.parquet rows {len(b)}")
print("done")
