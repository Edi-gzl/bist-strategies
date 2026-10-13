"""Mirror daily leg / rank-slot / BEST-overlay battery on INTRADAY champion book.

Baseline: ENS5 top3, gap skip (roll-63 z>0.5), E42 long tilt Mon/Thu/Fri.
Cost: 2 x 5 bps x (long_notional + short_notional), unhedged (deploy champion).
IS/OOS split: 2025-01-02 (intraday standard).
"""
from __future__ import annotations
from _rp import _RP  # portable-path shim (added when vendoring)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, _RP(r"\Research_done"))
from report_lib import sh

PKG = Path(_RP(r"\Research_done\INTRADAY_DAILY"))
OUT = PKG / "results"
BOOK = PKG / "data" / "book_top3.parquet"
PICKS = PKG / "full_picks.csv"
IS1 = pd.Timestamp("2025-01-02")
IS_END = pd.Timestamp("2024-12-31")
BPS = 5.0
LEG, HALF = 0.5, 0.25
Z_WIN = 63
START = pd.Timestamp("2023-01-01")
RANKS = 3
WCAP = 0.15  # corporate-action guard: |leg per-unit return| > 15% -> 0


def _guard(a):
    """Corporate-action guard: neutralize per-unit leg returns beyond +/-WCAP."""
    a = np.asarray(a, float)
    return np.where(np.abs(a) > WCAP, 0.0, a)


def zser(g: pd.Series) -> pd.Series:
    mu = g.rolling(Z_WIN, min_periods=30).mean().shift(1)
    sd = g.rolling(Z_WIN, min_periods=30).std().shift(1)
    return (g - mu) / sd


def roll_z(s: pd.Series, win: int = Z_WIN) -> pd.Series:
    return zser(s)


def mdd(s: pd.Series) -> float:
    s = s.dropna()
    eq = s.cumsum()
    return float((eq.cummax() - eq).max())


def slice_stats(net: np.ndarray, idx: pd.DatetimeIndex, name: str) -> list[dict]:
    s = pd.Series(net, index=idx).dropna()
    rows = []
    for slc, x in [
        ("FULL", s),
        ("IS", s[s.index <= IS1]),
        ("OOS", s[s.index > IS1]),
    ]:
        if len(x) < 10:
            continue
        rows.append(
            dict(
                rule=name,
                slice=slc,
                n=len(x),
                ann=round(x.mean() * 252 * 100, 1),
                sharpe=round(sh(x), 2),
                mdd=round(mdd(x) * 100, 2),
                hit=round((x > 0).mean() * 100, 1),
            )
        )
    return rows


def build_pick_panel() -> pd.DataFrame:
    B = pd.read_parquet(BOOK)
    B.index = pd.to_datetime(B.index)
    P = pd.read_csv(PICKS, parse_dates=["date"])

    rows = []
    for d, g in P.groupby("date"):
        if d < START:
            continue
        gl = g[g.leg == "LONG"].sort_values("signal", ascending=False)
        gs = g[g.leg == "SHORT"].sort_values("signal", ascending=False)
        if len(gl) < RANKS or len(gs) < RANKS:
            continue
        if d not in B.index:
            continue
        rec = dict(date=d)
        for k in range(RANKS):
            lk, sk = gl.iloc[k], gs.iloc[k]
            rk = k + 1
            rec[f"long{rk}_eps"] = float(lk.signal)
            rec[f"short{rk}_eps"] = float(sk.signal)
            rec[f"r{rk}_lon"] = float(lk.r_on)
            rec[f"r{rk}_lid"] = float(lk.r_id)
            rec[f"r{rk}_sid"] = float(-sk.r_id)
            rec[f"r{rk}_son"] = float(-sk.r_on)
            rec[f"long{rk}_ret_day"] = float(lk.r_on)
            rec[f"short{rk}_ret_day"] = float(sk.r_on)
        rec["eps_spread_long"] = rec["long1_eps"] - rec["long2_eps"]
        rec["eps_spread_short"] = rec["short1_eps"] - rec["short2_eps"]
        rec["book_spread"] = rec["long1_eps"] - rec["short1_eps"]
        rows.append(rec)

    p = pd.DataFrame(rows).set_index("date").sort_index()
    p = p.join(B, how="inner", rsuffix="_bk")

    for k in range(1, RANKS + 1):
        p[f"z_long{k}_eps"] = roll_z(p[f"long{k}_eps"])
        p[f"z_short{k}_eps"] = roll_z(p[f"short{k}_eps"])
    p["z_long_eps"] = roll_z(p["long1_eps"])
    p["z_spread_long"] = roll_z(p["eps_spread_long"])
    p["z_book_spread"] = roll_z(p["book_spread"])
    p["z_rel_l2_l1"] = p["z_long2_eps"] - p["z_long1_eps"]
    p["z_rel_s2_s1"] = p["z_short2_eps"] - p["z_short1_eps"]

    # short pick overnight return z (daily short_z63 analogue)
    sz = []
    for i, row in p.iterrows():
        past = p["short1_ret_day"].loc[:i].iloc[:-1].dropna().tail(Z_WIN)
        r = row["short1_ret_day"]
        if len(past) < Z_WIN // 3 or not np.isfinite(r):
            sz.append(np.nan)
            continue
        mu, sdv = float(past.mean()), float(past.std(ddof=1))
        sz.append((r - mu) / sdv if sdv > 1e-12 else np.nan)
    p["short_z63"] = sz

    # long pick return z
    lz = []
    for i, row in p.iterrows():
        past = p["long1_ret_day"].loc[:i].iloc[:-1].dropna().tail(Z_WIN)
        r = row["long1_ret_day"]
        if len(past) < Z_WIN // 3 or not np.isfinite(r):
            lz.append(np.nan)
            continue
        mu, sdv = float(past.mean()), float(past.std(ddof=1))
        lz.append((r - mu) / sdv if sdv > 1e-12 else np.nan)
    p["long_pick_z63"] = lz

    # mom streak proxy: prior 5d long1 overnight returns (fast; ticker-change noise ok)
    r = p["long1_ret_day"]
    pos5 = (r > 0).astype(float).rolling(5, min_periods=4).sum().shift(1)
    ret5 = r.rolling(5, min_periods=4).sum().shift(1)
    p["mom_streak"] = (ret5 > 0) & (pos5 >= 4)

    fit = p.index <= IS_END
    p["long_weak"] = p["long1_eps"] < p["long1_eps"].loc[fit].quantile(0.5)
    p["spread_wide"] = p["eps_spread_long"] > p["eps_spread_long"].loc[fit].quantile(0.75)
    p["spread_tight"] = p["eps_spread_long"] < p["eps_spread_long"].rolling(Z_WIN, min_periods=21).quantile(0.25).shift(1)

    intra = ((1 + p["mkt"]) / (1 + p["gap"]) - 1).shift(1)
    p["zin"] = zser(intra)
    p["dow"] = p.index.dayofweek
    return p


def e42_weight(p: pd.DataFrame) -> np.ndarray:
    mtf = p["dow"].isin([0, 3, 4]).values
    w = (1 + 0.5 * p["zin"].fillna(0)).clip(0.3, 2.0).fillna(1.0)
    return w.where(mtf, 1.0).values


def net_book(
    p: pd.DataFrame,
    long_ret: np.ndarray,
    short_ret: np.ndarray,
    skip: np.ndarray,
    tilt: bool = True,
    long_scale: np.ndarray | None = None,
    short_scale: np.ndarray | None = None,
) -> np.ndarray:
    """long_ret/short_ret: per-unit returns (not yet x LEG)."""
    w = e42_weight(p) if tilt else np.ones(len(p))
    ls = np.ones(len(p)) if long_scale is None else long_scale
    ss = np.ones(len(p)) if short_scale is None else short_scale
    long_ret = _guard(long_ret)
    short_ret = _guard(short_ret)
    lon_g = w * ls * LEG * long_ret
    sid_g = np.where(skip, 0.0, ss * LEG * short_ret)
    long_not = w * ls * LEG
    short_not = np.where(skip, 0.0, ss * LEG)
    cost = 2 * BPS / 1e4 * (long_not + short_not)
    return lon_g + sid_g - cost


def net_simple(long_g: np.ndarray, short_g: np.ndarray, skip: np.ndarray, tilt: bool, p: pd.DataFrame) -> np.ndarray:
    """Pre-weighted gross legs (already LEG * ret)."""
    w = e42_weight(p) if tilt else np.ones(len(p))
    long_g = np.where(np.abs(long_g) > WCAP * LEG, 0.0, long_g)
    short_g = np.where(np.abs(short_g) > WCAP * LEG, 0.0, short_g)
    g = w * long_g + np.where(skip, 0.0, short_g)
    long_not = w * LEG
    short_not = np.where(skip, 0.0, LEG)
    return g - 2 * BPS / 1e4 * (long_not + short_not)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = build_pick_panel()
    idx = p.index
    skip = (zser(p["gap"]) > 0.5).fillna(False).values
    w = e42_weight(p)

    leg_rows, overlay_rows, slot_rows = [], [], []

    def run_leg(name, net, tilted=True):
        leg_rows.extend(slice_stats(net, idx, name))

    def run_overlay(name, net):
        overlay_rows.extend(slice_stats(net, idx, name))

    # --- 1. Leg decomposition (book columns + champion wrapper) ---
    B = p
    leg_specs = [
        ("L_on only (overnight long)", B.L_on.values, np.zeros(len(p)), True),
        ("L_id only (intraday extend)", B.L_id.values, np.zeros(len(p)), False),
        ("L_on+L_id", B.L_on.values + B.L_id.values, np.zeros(len(p)), True),
        ("S_id only (intraday short)", np.zeros(len(p)), B.S_id.values, False),
        ("S_on only (overnight short)", np.zeros(len(p)), B.S_on.values, False),
        ("S_id+S_on", np.zeros(len(p)), B.S_id.values + B.S_on.values, False),
        ("PM L_on+S_id no skip", B.L_on.values, B.S_id.values, True),
        ("CHAMPION (gap skip + E42)", B.L_on.values, B.S_id.values, True),
        ("CHAMPION no tilt", B.L_on.values, B.S_id.values, True),
    ]
    for name, lr, sr, tilt in leg_specs:
        if name == "CHAMPION no tilt":
            net = net_book(p, lr, sr, skip, tilt=False)
        elif name == "PM L_on+S_id no skip":
            net = net_book(p, lr, sr, np.zeros(len(p), dtype=bool), tilt=True)
        elif "only" in name or "L_on+L_id" in name or "S_id+S_on" in name:
            # single-leg style: no tilt on pure leg tests
            lg, sg = LEG * lr, LEG * sr
            if "L_on only" in name or "L_id" in name or "L_on+L_id" in name:
                net = lg - 2 * BPS / 1e4 * LEG
            else:
                net = sg - 2 * BPS / 1e4 * LEG
        else:
            net = net_book(p, lr, sr, skip, tilt=tilt)
        run_leg(name, net)

    # CHAMPION + L_id variants
    for name, lid_cond in [
        ("CHAMP + L_id always", np.ones(len(p), dtype=bool)),
        ("CHAMP + L_id if z_long1>0", (p["z_long1_eps"].fillna(0) > 0).values),
        ("CHAMP + L_id if z_long1>1", (p["z_long1_eps"].fillna(0) > 1).values),
        ("CHAMP + L_id if long_pick_z63>0", (p["long_pick_z63"].fillna(0) > 0).values),
    ]:
        lr = B.L_on.values + np.where(lid_cond, B.L_id.values, 0)
        net = net_book(p, lr, B.S_id.values, skip, tilt=True)
        run_leg(name, net)

    # Gap alternatives
    gap_alt = [
        ("Gap: L_on+L_id no short", B.L_on.values + B.L_id.values, B.S_id.values, skip),
        ("Gap: rank2 SID not skip", B.L_on.values, p["r2_sid"].values, np.zeros(len(p), dtype=bool)),
        ("Gap: S_on not skip", B.L_on.values, B.S_on.values, np.zeros(len(p), dtype=bool)),
        ("Gap: half rank2 SID", B.L_on.values, np.where(skip, p["r2_sid"].values, B.S_id.values), skip),
    ]
    for name, lr, sr, sk in gap_alt:
        run_leg(name, net_book(p, lr, sr, sk, tilt=True))

    # --- 2. Per-rank leg quality ---
    rank_rows = []
    for k in range(1, RANKS + 1):
        for leg, col in [
            (f"rank{k} LON", f"r{k}_lon"),
            (f"rank{k} LID", f"r{k}_lid"),
            (f"rank{k} SID", f"r{k}_sid"),
            (f"rank{k} SON", f"r{k}_son"),
        ]:
            g = LEG * p[col].values
            net = g - 2 * BPS / 1e4 * LEG
            s = pd.Series(net, index=idx)
            oos = s[s.index > IS1]
            rank_rows.append(
                dict(rank=k, leg=leg, sh_full=round(sh(s), 2), sh_oos=round(sh(oos), 2), ann_oos=round(oos.mean() * 252 * 100, 1))
            )

    # --- 3. BEST-style on intraday (CHAMP base) ---
    r1l, r2l = p["r1_lon"].values, p["r2_lon"].values
    r1s, r2s = p["r1_sid"].values, p["r2_sid"].values
    dual_l = HALF * r1l + HALF * r2l
    dual_s = HALF * r1s + HALF * r2s
    weak = p["long_weak"].fillna(False).values
    wide = p["spread_wide"].fillna(False).values
    wide_z = (p["z_spread_long"] > 0.5).fillna(False).values
    mom = np.array(p["mom_streak"], dtype=bool)

    base_best_lr = np.where(weak, dual_l, r1l)
    base_best_sr = np.where(wide_z | mom, dual_s, r1s)
    base_skip = skip | (p["short_z63"] > -0.5).fillna(False).values
    base_scale = np.where(mom, 0.5, 1.0)

    def best_net(lr, sr, sk, ss=None):
        ss = np.ones(len(p)) if ss is None else ss
        return net_book(p, lr, sr, sk, tilt=True, long_scale=np.ones(len(p)), short_scale=ss)

    champ_net = net_book(p, B.L_on.values, B.S_id.values, skip, tilt=True)
    champ_oos = sh(pd.Series(champ_net, index=idx)[idx > IS1])

    configs = {
        "BASE: CHAMPION": champ_net,
        "CURRENT: CHAMP no overlays": net_book(p, r1l, r1s, skip, tilt=True),
        "BEST-adapt: weak dual + z_spread/mom short": best_net(base_best_lr, base_best_sr * base_scale, base_skip, base_scale),
        "BEST-adapt: no short_z skip": best_net(base_best_lr, base_best_sr * base_scale, skip, base_scale),
        "CHAMP + weak dual long": net_book(p, np.where(weak, dual_l, r1l), r1s, skip, tilt=True),
        "CHAMP + wide dual short": net_book(p, B.L_on.values, np.where(wide_z, dual_s, r1s), skip, tilt=True),
        "CHAMP + dual_Lweak_Swide (IS pct)": net_book(p, np.where(weak, dual_l, r1l), np.where(wide, dual_s, r1s), skip, tilt=True),
        "CHAMP + dual_Lweak_Swide_z": net_book(
            p, np.where(weak, dual_l, r1l), np.where(wide_z, dual_s, r1s), skip, tilt=True
        ),
        "CHAMP + mom halve short": net_book(p, B.L_on.values, np.where(mom, dual_s * 0.5, r1s), skip, tilt=True, short_scale=np.where(mom, 0.5, 1.0)),
        "CHAMP + short_z skip": net_book(p, B.L_on.values, r1s, skip | (p["short_z63"] > -0.5).fillna(False).values, tilt=True),
        "CHAMP + short_z AND z_short1<-0.5": net_book(
            p,
            B.L_on.values,
            r1s,
            skip | ((p["short_z63"] > -0.5).fillna(False).values & (p["z_short1_eps"].fillna(0) < -0.5).values),
            tilt=True,
        ),
    }

    for name, net in configs.items():
        run_overlay(name, net)

    # Slot-z overlays on champion
    z2 = p["z_long2_eps"].fillna(0).values
    zrel = p["z_rel_l2_l1"].fillna(0).values
    zs2 = p["z_short2_eps"].fillna(0).values
    tight = p["spread_tight"].fillna(False).values

    slot_rules = [
        ("CHAMP + dual lon if z_long2>0.5", np.where(z2 > 0.5, dual_l, r1l), r1s, skip),
        ("CHAMP + dual lon if z_rel>0.5", np.where(zrel > 0.5, dual_l, r1l), r1s, skip),
        ("CHAMP + dual lon if z_long2>0.5 & tight", np.where((z2 > 0.5) & tight, dual_l, r1l), r1s, skip),
        ("CHAMP + dual short if z_short2>0.5", B.L_on.values, np.where(zs2 > 0.5, dual_s, r1s), skip),
        ("CHAMP + halve long if z_long1>1", B.L_on.values, r1s, skip),
        ("BEST + dual lon z_rel>0.5", np.where(zrel > 0.5, dual_l, base_best_lr), base_best_sr * base_scale, base_skip, base_scale),
        ("BEST + crowded dual lon", np.where(tight, dual_l, base_best_lr), base_best_sr * base_scale, base_skip, base_scale),
    ]
    for item in slot_rules:
        name = item[0]
        lr, sr, sk = item[1], item[2], item[3]
        ss = item[4] if len(item) > 4 else None
        if ss is not None:
            net = best_net(lr, sr, sk, ss)
        else:
            net = net_book(p, lr, sr, sk, tilt=True)
        run_overlay(name, net)
        oos_sh = sh(pd.Series(net, index=idx)[idx > IS1])
        slot_rows.append(dict(rule=name, sh_oos=round(oos_sh, 2), delta=round(oos_sh - champ_oos, 2)))

    # max |z_long_k| dynamic rank (ranks 1-3)
    zcols = [p[f"z_long{k}_eps"] for k in range(1, RANKS + 1)]
    zmat = pd.concat(zcols, axis=1)
    zmat.columns = range(1, RANKS + 1)
    ext_k = zmat.abs().fillna(-np.inf).idxmax(axis=1)
    lon_ext = np.array([p[f"r{int(k)}_lon"].iloc[i] for i, k in enumerate(ext_k)])
    net = net_book(p, lon_ext, r1s, skip, tilt=True)
    oos_sh = sh(pd.Series(net, index=idx)[idx > IS1])
    slot_rows.append(dict(rule="CHAMP + LON at max|z_long_k|", sh_oos=round(oos_sh, 2), delta=round(oos_sh - champ_oos, 2)))

    leg_df = pd.DataFrame(leg_rows)
    overlay_df = pd.DataFrame(overlay_rows)
    rank_df = pd.DataFrame(rank_rows)
    slot_df = pd.DataFrame(slot_rows).sort_values("sh_oos", ascending=False)

    leg_df.to_csv(OUT / "results_intra_leg_overnight_intraday.csv", index=False)
    overlay_df.to_csv(OUT / "results_intra_overlays_best.csv", index=False)
    rank_df.to_csv(OUT / "results_intra_per_rank_leg_quality.csv", index=False)
    slot_df.to_csv(OUT / "results_intra_rank_slot_z.csv", index=False)

    print("=" * 95)
    print(f"INTRADAY PARITY BATTERY — n={len(p)} | CHAMPION OOS Sh={champ_oos:.2f}")
    print("=" * 95)

    print("\n--- 1. LEG DECOMPOSITION (OOS) ---")
    l1 = leg_df[leg_df.slice == "OOS"].sort_values("sharpe", ascending=False)
    print(l1[["rule", "ann", "sharpe", "mdd", "hit"]].head(18).to_string(index=False))

    print("\n--- 2. PER-RANK LEG QUALITY (OOS Sharpe) ---")
    print(rank_df.pivot_table(index="rank", columns="leg", values="sh_oos").to_string())

    print("\n--- 3. OVERLAYS / BEST-ADAPT (OOS vs CHAMPION) ---")
    o3 = overlay_df[overlay_df.slice == "OOS"].copy()
    o3["delta"] = o3["sharpe"] - champ_oos
    print(o3.sort_values("sharpe", ascending=False)[["rule", "ann", "sharpe", "mdd", "delta"]].to_string(index=False))

    print("\n--- 4. RANK SLOT Z (OOS) ---")
    print(slot_df.to_string(index=False))

    print(f"\nWrote {OUT}/results_intra_*.csv")


if __name__ == "__main__":
    main()
