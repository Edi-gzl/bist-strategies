from _rp import _RP  # portable-path shim (added when vendoring)
"""FINAL CLEAN audit (leak-free). halve-long-down REMOVED (added no value once timing-correct).
Recommended = S10c = top3-both + A3(disp-low->top1) + B4(extend Lmean_hi) + mom-halve-short + gap-skip(short).
All long gates verified NOT contemporaneous with L_on[T]. Short gates use gap known at open T+1. cost5+5, guard, 0.5/0.5.
Also return-tilt variant with A6 short-dispersion concentrate."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, _RP(r"\Research_done"))
sys.path.insert(0, _RP(r"\Research_done\INTRADAY_DAILY\scripts"))
from test_intra_daily_parity_battery import build_pick_panel, zser, e42_weight
IS1=pd.Timestamp("2025-01-02"); YTD=pd.Timestamp("2026-01-01")
SIDE=5e-4; FULL=2*SIDE; SKIP=1*SIDE; WCAP=0.15; LEG=0.5
def guard(v): v=np.asarray(v,float); return np.where(np.abs(v)>WCAP,0.0,v)
def blk(x):
    x=pd.Series(x).dropna()
    if len(x)<6: return None
    eq=x.cumsum(); dd=(eq.cummax()-eq).max()
    return dict(n=int(len(x)),ann=round(float(x.mean()*252*100),1),sh=round(float(x.mean()/x.std(ddof=1)*np.sqrt(252)),2),
                mdd=round(float(dd*100),2),hit=round(float((x>0).mean()*100),1),total=round(float(x.sum()*100),1))
def slc(s):
    s=pd.Series(np.asarray(s,float),index=idx)
    return {k:blk(v) for k,v in [("FULL",s),("IS",s[s.index<=IS1]),("OOS",s[s.index>IS1]),("YTD",s[s.index>=YTD])]}
def beta(s,m):
    d=pd.concat([pd.Series(s,index=idx).rename("s"),m.rename("m")],axis=1).dropna()
    return round(float(np.polyfit(d.m,d.s,1)[0]),3) if len(d)>20 else None
def cz(series):
    s=pd.to_numeric(series,errors="coerce"); return ((s-s.rolling(63,min_periods=30).mean().shift(1))/s.rolling(63,min_periods=30).std().shift(1))
p=build_pick_panel(); idx=p.index; w=e42_weight(p); one=np.ones(len(p))
r1L=guard(p["r1_lon"].values);r1S=guard(p["r1_sid"].values)
r2L=guard(p["r2_lon"].values);r2S=guard(p["r2_sid"].values)
r3L=guard(p["r3_lon"].values);r3S=guard(p["r3_sid"].values)
r1lid=guard(p["r1_lid"].values);r2lid=guard(p["r2_lid"].values);r3lid=guard(p["r3_lid"].values)
Lon_b=guard(p["L_on"].values);Sid_b=guard(p["S_id"].values)
r1l=LEG*r1L;r1s=LEG*r1S
tl3=LEG*(r1L+r2L+r3L)/3; ts3=LEG*(r1S+r2S+r3S)/3; lid3=LEG*(r1lid+r2lid+r3lid)/3
gap_short=(zser(p["gap"])>0.5).fillna(False).values     # SHORT gate: OK (known at open T+1)
mom=p["mom_streak"].fillna(False).values; sc_b=np.where(mom,0.5,1.0)
Le=p[["long1_eps","long2_eps","long3_eps"]]; zLstd=cz(Le.std(axis=1)); zLmean=cz(Le.mean(axis=1))
Ldisp_lo=(zLstd<-0.5).fillna(False).values; Lmean_hi=(zLmean>0.5).fillna(False).values
Se=p[["short1_eps","short2_eps","short3_eps"]]; zSstd=cz(Se.std(axis=1)); Sdisp_hi=(zSstd>0.5).fillna(False).values  # A6 (short gate, clean)
ew=pd.read_csv(_RP(r"\Research_done\INTRADAY_DAILY\data\EW_BIST50_daily.csv"),parse_dates=["date"]).set_index("date")["r"].reindex(idx)
def book(lon_leg,sid_leg,skip,ssc=one,lid_add=None):
    L=w*lon_leg
    if lid_add is not None: L=L+w*lid_add
    S=np.where(skip,0.0,sid_leg*ssc)
    cost=SKIP*w+np.where(skip,0.0,(FULL-SKIP)*np.abs(ssc)); return L+S-cost
A3lon=np.where(Ldisp_lo,r1l,tl3); ext=np.where(Lmean_hi,lid3,0.0)
CFG={
 "CURRENT champion (ENS5 basket)": book(LEG*Lon_b,LEG*Sid_b,gap_short),
 "LOCKED name-rank (r1/r1)":       book(r1l,r1s,gap_short),
 "top3-both (gap-skip only)":      book(tl3,ts3,gap_short),
 "+ mom-halve-short":              book(tl3,ts3,gap_short,ssc=sc_b),
 "+ A3 (disp-low->top1 long)":     book(A3lon,ts3,gap_short,ssc=sc_b),
 "+ B4 (extend Lmean_hi)":         book(tl3,ts3,gap_short,ssc=sc_b,lid_add=ext),
 "S10c = A3+B4+mom [RECOMMENDED]": book(A3lon,ts3,gap_short,ssc=sc_b,lid_add=ext),
 "S10c + A6 short-conc [RET-TILT]":book(A3lon,np.where(Sdisp_hi,r1s,ts3),gap_short,ssc=sc_b,lid_add=ext),
}
RES={k:slc(v) for k,v in CFG.items()}; BETA={k:beta(v,ew) for k,v in CFG.items()}
BENCH={"BIST50_EW_longonly":slc(ew.values)}
neut=int(sum((np.abs(p[c].values)>WCAP).sum() for c in ["r1_lon","r1_sid","r2_lon","r2_sid","r3_lon","r3_sid"]))
meta=dict(n=len(p),d0=str(idx[0].date()),d1=str(idx[-1].date()),n_is=int((idx<=IS1).sum()),n_oos=int((idx>IS1).sum()),
          n_ytd=int((idx>=YTD).sum()),IS1=str(IS1.date()),guard_neutralized=neut,side_bps=5.0,
          universe="BIST50 (48 names; TUPRS never picked -> 47 active)",
          leak_note="halve-long-down REMOVED: gap realized at open T+1 is contemporaneous with overnight long (corr 0.87) - cannot gate the long. Only short gate uses gap (known at open before intraday short).")
json.dump(dict(meta=meta,configs=RES,beta_vs_EW=BETA,benchmark=BENCH),
          open(_RP(r"\Research_done\INTRADAY_DAILY\results\results_intra_FINAL_audit.json"),"w"),indent=1)
cur=pd.Series(CFG["CURRENT champion (ENS5 basket)"],index=idx)
rec=pd.Series(CFG["S10c = A3+B4+mom [RECOMMENDED]"],index=idx)
t3=pd.Series(CFG["top3-both (gap-skip only)"],index=idx)
rt=pd.Series(CFG["S10c + A6 short-conc [RET-TILT]"],index=idx)
pd.DataFrame({"CURRENT":cur,"TOP3":t3,"S10c":rec,"RETTILT":rt,"EW":ew}).to_csv(_RP(r"\Research_done\INTRADAY_DAILY\results\FINAL_daily_series.csv"))
print("META leak_note:",meta["leak_note"][:80],"...")
print(f"\n{'config':<34}{'FULL a/s/m':>16}{'IS':>16}{'OOS':>16}{'YTD':>16}{'bEW':>7}")
for k in CFG:
    v=RES[k]
    def c(s): d=v[s]; return f"{d['ann']:.0f}/{d['sh']}/{d['mdd']}" if d else "-"
    print(f"{k:<34}{c('FULL'):>16}{c('IS'):>16}{c('OOS'):>16}{c('YTD'):>16}{BETA[k]:>7}")
e=BENCH["BIST50_EW_longonly"]; print(f"{'BIST50 EW bench':<34}{e['FULL']['ann']:.0f}/{e['FULL']['sh']}/{e['FULL']['mdd']}  OOS {e['OOS']['ann']:.0f}/{e['OOS']['sh']}")
print("WROTE clean FINAL audit (leak-fixed, halve-down removed)")
