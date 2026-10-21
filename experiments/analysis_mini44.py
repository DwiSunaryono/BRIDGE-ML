"""Locked analysis of MINI-44 (PRAREG_MINI44.md): H1 tuned baseline, H2 per-task tuned gate, and (q, eps) on real data.

Settings (A) and (D) are analyzed separately, with datasets as clusters. Bootstrap uses B = 10,000 and seed 20261018.
Usage: python analysis_mini44.py -> prints and writes out/analysis_mini44_*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import summarize_paper as SP
from verify_exp41 import kl_ratio_up as up

HERE = Path(__file__).resolve().parent
B, SEED, EQ = 10_000, 20261018, 2.0
QS, ES, DEF = (0.025, 0.05, 0.1), (0.0025, 0.005, 0.0075, 0.01), (0.05, 0.005)


def load():
    d = pd.concat([pd.read_csv(HERE / "out" / f"mini44_m{i}.csv", float_precision="round_trip") for i in (1, 2, 3)])
    d = d[d.status == "ok"].copy(); d["S"] = d.setting.replace({"D-mc": "D"}); d["has"] = d.f_cfg.notna() & (d.f_cfg != "None")
    return d.reset_index(drop=True)


def crit(acc, wrong, unit):
    acc, wrong = np.asarray(acc, bool), np.asarray(wrong, bool); NT = len(acc)
    per = pd.Series(unit).value_counts(); u = np.asarray(unit)
    n, k = int(acc.sum()), int((acc & wrong).sum())
    kl = up(k, n, NT, 1); lo = SP.loco(pd.Series(acc), pd.Series(wrong), pd.Series(u), per, NT)
    r0 = max(up(0, int((acc & (u != x)).sum()), NT - nx, 1) if (acc & (u != x)).sum() else 1.0 for x, nx in per.items())
    y = 100 * n / NT; dsy = sum(100 * (acc & (u == x)).sum() / nx >= 5 for x, nx in per.items())
    return dict(accepted=n, harmful=k, harm_pct=100 * k / max(n, 1), kl=100 * kl, loco=100 * lo, r0=100 * r0, yield_=y, ds_yield5=int(dsy),
                meets=bool(kl < .1 and lo < .1 and r0 < .1 and y >= 5 and dsy >= 3))


def boot(m, fn):
    rng = np.random.default_rng(SEED); idx = [np.flatnonzero((m.dataset == u).to_numpy()) for u in m.dataset.unique()]
    out = []
    for _ in range(B):
        s = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]); v = fn(s)
        if np.isfinite(v):
            out.append(v)
    out = np.array(out); return fn(np.arange(len(m))), np.percentile(out, 2.5), np.percentile(out, 97.5), out


def hp(acc, wrong):
    return lambda s: 100 * wrong[s][acc[s]].mean() if acc[s].any() else np.nan


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def main():
    d = load(); H1, H2, QE = [], [], []
    for S, m in d.groupby("S"):
        m = m.reset_index(drop=True); u = m.dataset.to_numpy()
        fref, href = m.f_gain_ref.to_numpy(), m.h_gain_ref.to_numpy()
        f_acc = (m.has & (m.f_lb05 > DEF[1])).to_numpy(); f_nv = (m.f_gain_pool > 0).to_numpy()
        h_acc = (m.h_accept == 1).to_numpy(); h_nv = (m.h_naive == 1).to_numpy()
        # H1
        cb, cn = crit(h_acc, href <= 0, u), crit(h_nv, href <= 0, u)
        dd = lambda s: hp(h_acc, href <= 0)(s) - hp(h_nv, href <= 0)(s)
        e, lo, hi, _ = boot(m, dd)
        t = lambda s: hp(h_acc, href <= 0)(s) - hp(f_acc, fref <= 0)(s)
        te, tlo, thi, tb = boot(m, t)
        man = lambda s: (m.h_base_ref.to_numpy()[s] - m.f_base_ref.to_numpy()[s]).mean() * 100
        me, mlo, mhi, _ = boot(m, man)
        fin_f = np.where(f_acc, m.f_dep_ref, m.f_base_ref); fin_h = np.where(h_acc, m.h_dep_ref, m.h_base_ref)
        fe, flo, fhi, _ = boot(m, lambda s: (fin_h[s] - fin_f[s]).mean() * 100)
        H1.append(dict(setting=S, decisions=len(m), naive_acc=cn["accepted"], naive_harm_pct=cn["harm_pct"],
                       **{f"bridge_{k}": v for k, v in cb.items()}, bridge_minus_naive_pp=e, bm_lo=lo, bm_hi=hi,
                       default_bridge_harm_pct=100 * (f_acc & (fref <= 0)).sum() / f_acc.sum(), default_bridge_acc=int(f_acc.sum()),
                       tuned_minus_default_pp=te, td_lo=tlo, td_hi=thi, p_tost=max((tb <= -EQ).mean(), (tb >= EQ).mean()),
                       baseline_gain_from_tuning_wf1_pts=me, bg_lo=mlo, bg_hi=mhi,
                       final_model_tuned_minus_default_pts=fe, fm_lo=flo, fm_hi=fhi,
                       net_gain_bridge_tuned=href[h_acc].sum(), net_gain_bridge_default=fref[f_acc].sum(),
                       net_gain_naive_tuned=href[h_nv].sum()))
        # H2
        t_acc = (m.t_accept == 1).to_numpy()
        ct = crit(t_acc, fref <= 0, u)
        he, hlo, hhi, _ = boot(m, lambda s: hp(t_acc, fref <= 0)(s) - hp(f_acc, fref <= 0)(s))
        ge, glo, ghi, _ = boot(m, lambda s: 100 * (fref[s][t_acc[s]].sum() - fref[s][f_acc[s]].sum()) / len(s))
        H2.append(dict(setting=S, tuned_acc=ct["accepted"], tuned_harm=ct["harmful"], tuned_harm_pct=ct["harm_pct"], tuned_kl=ct["kl"],
                       tuned_meets=ct["meets"], frozen_acc=int(f_acc.sum()), frozen_harm=int((f_acc & (fref <= 0)).sum()),
                       harm_diff_pp=he, hd_lo=hlo, hd_hi=hhi, netgain_diff_per100=ge, ng_lo=glo, ng_hi=ghi,
                       netgain_ratio=fref[t_acc].sum() / fref[f_acc].sum(),
                       chosen_default_pct=100 * ((m.t_q == DEF[0]) & (m.t_eps == DEF[1])).mean(),
                       no_benefit_supported=bool(not (hhi < 0) and not (glo > 0))))
        # (q, eps)
        rows = []
        for q in QS:
            for eps in ES:
                acc = (m.has & (m[f"f_q{q}"] > eps)).to_numpy(); c = crit(acc, fref <= 0, u)
                r = dict(setting=S, q=q, eps=eps, **c, netgain_ratio=fref[acc].sum() / fref[f_acc].sum())
                if (q, eps) != DEF:
                    de, dlo, dhi, db = boot(m, lambda s, a=acc: hp(a, fref <= 0)(s) - hp(f_acc, fref <= 0)(s))
                    r |= dict(diff_pp=de, d_lo=dlo, d_hi=dhi, p_tost=max((db <= -EQ).mean(), (db >= EQ).mean()))
                rows.append(r)
        r = pd.DataFrame(rows); v = r.p_tost.notna(); r.loc[v, "p_tost_holm"] = holm(r.loc[v, "p_tost"]); r["equivalent"] = r.p_tost_holm < .05
        QE.append(r)
    pd.set_option("display.width", 250)
    for nm, x in (("h1", pd.DataFrame(H1)), ("h2", pd.DataFrame(H2)), ("qeps", pd.concat(QE))):
        x.to_csv(HERE / "out" / f"analysis_mini44_{nm}.csv", index=False); print(f"== {nm}"); print(x.round(4).T.to_string() if nm != "qeps" else x.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
