"""Generate the first version of the figures (PDF and PNG) and LaTeX tables of the paper from the CSVs in out/.

    python paper_assets.py          # writes to out/paper_assets_regenerated/

The final tables of the paper are in ../paper_assets/. Several of them were later set, merged, or composed by hand
(first line of each file); paper_assets/TABLE_MAP.md lists, for every table, how it was produced, whether this
generator reproduces it byte for byte, and the script and CSV that recompute its numbers. The generator writes to a
separate folder so that it never overwrites the final tables.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize_paper as SP  # noqa: E402
from verify_exp41 import kl_ratio_up as up  # noqa: E402

OUT = HERE / "out" / "paper_assets_regenerated"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
RD = SP.rd


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png"); plt.close(fig)



def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion, in percent. Decisions are treated as independent; cluster
    intervals for the main comparisons are in the CI table."""
    k, n = np.asarray(k, float), np.asarray(n, float); p = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    den = 1 + z ** 2 / np.maximum(n, 1); c = (p + z ** 2 / (2 * np.maximum(n, 1))) / den
    h = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z ** 2 / (4 * np.maximum(n, 1) ** 2)) / den
    return 100 * (c - h), 100 * (c + h)


class Raw(str):
    """A table cell that is already LaTeX: passed through by _esc/fmt without escaping."""
    pass


def _esc(x):
    x = str(x)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        x = x.replace(a, b)
    x = x.replace("5x2cv", r"$5\times2$~cv")
    x = re.sub(r"(?<![0-9A-Za-z#$])F1\b", r"$F_1$", x)  # F1 score in math, as in the text
    x = re.sub(r"\b(paired|resampled|Corrected|corrected) t\b", r"\1 $t$", x)  # t-test statistic in italics, as in the text
    return x.replace("Δ", r"$\Delta$").replace("≥", r"$\geq$").replace("–", "--")


# One-line captions (aturan_paper.md section 4); the rest of the generator caption becomes the table footnote.
CAPS = {
    "tab1_rules_identical_decisions": "Acceptance rules on identical decisions in (A): BRIDGE has the lowest risk bound, also when any one dataset is left out (LOCO).",
    "tab1b_hyperparameters": "Frozen parameters of BRIDGE and of the evaluation protocol.",
    "tab2_synthetic_population": "Acceptance rules on the synthetic tasks (B): BRIDGE has the lowest Proposition 1 bound.",
    "tab4_holdout": "BRIDGE compared with an honest holdout (part of the same labeled data set aside to test the chosen update): 1.4–1.9 times the accepted updates at a lower risk bound.",
    "tab5b_generalisation": "Conditions of use on (A) and (D): the frozen gate meets the criterion across metrics, training sizes and multiclass tasks, and with feature selection for KNN.",
    "tab5_clinical": "Retrospective clinical evaluation by metric: BRIDGE meets the criterion on every metric.",
    "tab6_clinical_tasks": "Clinical tasks.",
    "tab8_complexity": "Wall-clock cost of BRIDGE.",
    "tab9_ablation_factorial": "Factorial ablation (each of the three BRIDGE components switched on or off): the full gate has the lowest risk bound.",
    "tab10_candidate_space": "Effect of candidate-space size: with each rule accepting its own number of updates, BRIDGE keeps the lowest risk bound at every size.",
    "tab11_base_learner": "Base learners: the gate meets the criterion for HGB, both MLPs and XGBoost with early stopping or 30 rounds; adding early stopping is sufficient for XGBoost.",
    "tabA_dataset_splits": "Data partition per dataset or task.",
    "tabS2_base_learner_audit": "Audit of the base learners.",
    "tabS3_identity_audit": "Patient-identity audit of the clinical sources.",
}

def tex(df, name, caption, label, rc=None, pw=None, hdroverrides=None, bold=(), bold_sel=None, bold_note=None, crit_note=None, foot=None, groupcol=None, grouppal=None, firstcol="l", fixed=False, natural=False, narrow=False):
    """LaTeX table in the CORE / IEEE Access house style (booktabs, @{} margins, table* when wide, no
    resizebox), without a jinja2 dependency. rc: optional list of \\rowcolor names, one per data row;
    pw: optional dict of explicit p-column widths in pt per column (minimum point to fit). Direction
    arrows (ARROWS, {down}/{up} in the header), bold best values (bold columns, best among bold_sel
    rows), a standard direction and criterion footnote, and check/x marks for criterion columns are
    generated for the results tables. Both require the manuscript preamble, with colors defined there
    as in CORE's irlow/irmoderate/irextreme."""
    YN = {"YA": r"$\checkmark$", "TIDAK": r"$\times$", True: r"$\checkmark$", False: r"$\times$"}
    def fmt(v, d=2):
        if isinstance(v, Raw):
            return str(v)
        if isinstance(v, (bool, np.bool_)) or v in ("YA", "TIDAK"):
            return YN[v if not isinstance(v, np.bool_) else bool(v)]
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, float):
            return "--" if v != v else f"{v:.{d}f}"
        return _esc(v)
    def decimals(col):  # 2 decimals for columns with values >= 1, else 3 so small values (margin, seconds) stay visible
        x = pd.to_numeric(df[col], errors="coerce").abs().dropna()
        if len(x) == 0 or x.max() >= 1 or np.allclose(x * 100, np.round(x * 100), atol=1e-9):
            return 2  # large values, or a column already rounded to 2 decimals on purpose
        return 3
    def hdr(c, w=9, arrow=""):  # stack a long header into short lines; arrow goes on the last line
        out, cur = [], ""
        for t in _esc(c).split():
            if cur and len(cur) + 1 + len(t) > w:
                out.append(cur); cur = t
            else:
                cur = (cur + " " + t).strip()
        out.append(cur)
        if arrow:
            out[-1] = out[-1] + " $" + arrow + "$"
        return out[0] if len(out) == 1 else r"\begin{tabular}[b]{@{}c@{}}" + r"\\".join(out) + r"\end{tabular}"
    # house style (aturan_paper.md section 4): one-line caption; method detail goes in a justified footnote
    cap = CAPS.get(name, caption)
    note = caption.replace("Clinical validation:", "Retrospective clinical evaluation:") if cap != caption else ""
    cells = pd.DataFrame({c: [fmt(v, decimals(c)) for v in df[c]] for c in df.columns}, index=df.index)
    if bold:  # bold the best value per column among the selected comparable rows
        sel = pd.Series(bold_sel, index=df.index) if bold_sel is not None else pd.Series(True, index=df.index)
        for col in bold:
            if col not in df.columns:
                continue
            s = pd.to_numeric(pd.Series(df[col].values, index=df.index), errors="coerce")
            use = sel.fillna(False) & s.notna()
            if not use.any():
                continue
            best = s[use].min() if col in ARROWS and ARROWS[col] == "\\downarrow" else s[use].max()
            for i in use.index[use & (s == best)]:
                cells.loc[i, col] = r"\textbf{" + cells.loc[i, col] + "}"
    longtxt = [c for c in df.columns if (pw is not None and c in pw) or cells[c].str.len().max() > 28]
    # rough width estimate in pt decides one-column table vs two-column table*
    tw, cw = 516.0, 252.0  # IEEE Access text / column width (pt)
    nw = lambda c: 4.0 * max(cells[c].str.len().max(), min(9, max(len(t) for t in str(c).split()))) + 6
    pwd = {c: pw[c] for c in (pw or {})}
    for c in longtxt:
        pwd.setdefault(c, min(4.0 * cells[c].str.len().max() / 2.2, 150.0))  # long text wrapped to ~2-3 lines
    est = sum(pwd[c] + 6 if c in longtxt else nw(c) for c in df.columns)
    wide = est > 0.9 * cw  # safety margin: estimate is rough
    labels_only = natural or (not fixed and all(c == df.columns[0] for c in longtxt))
    if labels_only:  # the only text is the row label: natural widths, no wrapping; float width from that width
        longtxt = []
        wide = sum(nw(c) for c in df.columns) > cw
    wide = wide and not narrow  # narrow: force one IEEE column (text columns wrap)
    if narrow:  # every text column may wrap inside the column width
        longtxt = [c for c in df.columns if cells[c].str.contains(r"[A-Za-z]{3}").any() and cells[c].str.len().max() > 12]
        for c in longtxt:
            pwd.setdefault(c, min(4.0 * cells[c].str.len().max() / 2.2, 150.0))
    env, size = ("table*", r"\scriptsize") if wide else ("table", r"\scriptsize" if narrow else r"\footnotesize")
    if est > tw and longtxt:  # shrink text columns to fit (only proxy-estimated ones)
        f = max(0.35, 1 - (est - tw) / sum(v for c, v in pwd.items() if (pw is None or c not in pw)))
        pwd = {c: v * f if (pw is None or c not in pw) else v for c, v in pwd.items()}
    spec = "".join(f">{{\\raggedright\\arraybackslash}}p{{{pwd[c]:.0f}pt}}" if c in longtxt else ("l" if i == 0 else "c") for i, c in enumerate(df.columns))
    if natural:  # text columns left-aligned at natural width
        isnum = lambda c: pd.api.types.is_numeric_dtype(df[c]) or all(re.fullmatch(r"[\d,.]*", str(v)) for v in df[c])
        spec = "".join("r" if isnum(c) else "l" for c in df.columns)
    hc = (lambda c: (hdroverrides or {}).get(c)) if hdroverrides else (lambda c: None)
    def hcell(c):
        v = hc(c)
        return v if v is not None else hdr(c, arrow=ARROWS.get(c, ""))
    # colour policy: the criterion is carried by the check/x column and by bold type; only the
    # partition table keeps group tints.
    rc = None if not grouppal else rc
    crit = False
    groups = list(df[groupcol]) if groupcol else [None] * len(df)
    if grouppal:  # each group gets its own colour family; no rules between groups
        rc, k = [], 0
        for i, g in enumerate(groups):
            k = 0 if i and g != groups[i - 1] else k
            rc.append(f"bg{g}0"); k += 1
    elif False:
        rc, k = [], 0
        for i, g in enumerate(groups):
            k = 0 if i and g != groups[i - 1] else k
            rc.append("bgzebra" if k % 2 else ""); k += 1
    # tables with text columns use tabulary (preamble: \usepackage{tabulary}): LaTeX sizes each column
    # from its content and fills the line, so only the longest text wraps.
    tbl = "tabulary" if longtxt and not fixed else "tabular"
    if fixed:  # explicit p-widths (pw) for tables tabulary would balance badly
        longtxt = list(pw)
        spec = "".join(f">{{\\raggedright\\arraybackslash}}p{{{pw[c]:.0f}pt}}" if c in pw else "c" for c in df.columns)
    if tbl == "tabulary":
        # first column: natural width (labels never wrap); other text columns share the rest; numbers centred
        spec = "".join(firstcol if i == 0 else ("L" if c in longtxt else "c") for i, c in enumerate(df.columns))
        begin = (r"\setlength{\tymin}{28pt}\begin{tabulary}{" + (r"\textwidth" if wide else r"\columnwidth") + "}{@{}" + spec + "@{}}")
    else:
        begin = r"\begin{tabular}{@{}" + spec + "@{}}"
    lines = [begin, r"\toprule", " & ".join(hcell(c) for c in df.columns) + r" \\", r"\midrule"]
    for i, (row, color) in enumerate(zip(cells.itertuples(index=False), rc if rc is not None else [None] * len(df))):
        if groupcol and not grouppal and i and groups[i] != groups[i - 1]:
            lines.append(r"\midrule")
        if color:
            lines.append("\\rowcolor{" + {"tabpass": "bgpass", "tabfail": "bgfail"}.get(color, color) + "}")
        row = list(row)
        if row[0] == "BRIDGE":
            row[0] = r"\textbf{BRIDGE}"
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{" + tbl + "}"]
    fin = _esc(note).replace(r"\textbackslash{}", "\\") if note else ""  # notes may carry inline math such as $|\\mathcal{C}|$
    std = DIR_NOTE + ((bold_note or BOLD_NOTE) if bold else "") + (CRIT_NOTE if crit_note is None else crit_note)
    if any(c in ARROWS for c in df.columns) or bold:
        fin = (fin + "  " + std) if fin else std
    if crit:
        fin = (fin.rstrip(". ") + COLOR_NOTE) if fin else COLOR_NOTE.lstrip("; ").capitalize()
    if foot:
        fin = (fin + "  " + foot) if fin else foot
    (OUT / f"{name}.tex").write_text("% generated by paper_assets.py (requires \\usepackage{booktabs,array,ragged2e} and, for colored rows, \\usepackage[table]{xcolor}; row colors defined in the manuscript preamble)\n\\begin{" + env + "}[!t]\n\\centering\n" + size +
                                     "\n\\providecolor{bghead}{HTML}{DBEFF8}\\providecolor{bgzebra}{HTML}{F2F9FC}\\providecolor{bgpass}{HTML}{D7F0D2}\\providecolor{bgfail}{HTML}{FCDCD7}" + "".join(f"\\providecolor{{bg{g}{j}}}{{HTML}}{{{c}}}" for g, cs in (grouppal or {}).items() for j, c in enumerate(cs)) + "\n\\setlength{\\tabcolsep}{3pt}\n\\caption{" + _esc(cap) + "}\n\\label{" + label + "}\n" + "\n".join(lines) +
                                     ("\n\n\\vspace{2pt}\n\\parbox{\\linewidth}{\\justifying\\scriptsize *" + fin + "}" if fin else "") + "\n\\end{" + env + "}\n")


def rcolor(df, col, mapping, default=""):
    """Per-row \\rowcolor names from a category column (see CORE's category tints)."""
    return [mapping.get(v, default) for v in df[col]]


def rule_color(name):
    """Row tint for the rule tables: naive red, significance tests grey, honest variants light, BRIDGE green."""
    n = str(name).lower()
    if n.startswith("naive"):
        return "tabfail"
    if n.startswith("bridge without margin"):
        return "tabhy"
    if n.startswith("bridge"):
        return "tabpass"
    if "per-split" in n:
        return "tabhy"
    return "tabmid"


def yn_color(v):
    """Row tint from a binary criterion column (values may be YA/TIDAK, Yes/No, or 1/0)."""
    if isinstance(v, (bool, np.bool_)):
        return "tabpass" if v else "tabfail"
    if isinstance(v, (float, np.floating)):
        v = int(v)
    if v in ("YA", "TIDAK"):
        return "tabpass" if v == "YA" else "tabfail"
    if v in ("Yes", "No"):
        return "tabpass" if v == "Yes" else "tabfail"
    if v in (1, "1"):
        return "tabpass"
    if v in (0, "0"):
        return "tabfail"
    return ""


def phase_color(v):
    """Row tint for the dataset-split table by phase."""
    return {"confirmation (34)": "tabbgconfirm", "new binary (11)": "tabbgnewbin",
            "new multiclass (5)": "tabbgnewmc", "clinical (12 tasks)": "tabbgclin"}.get(v, "")


ARROWS = {
    "Bound %": "\\downarrow",
    "Harmful %": "\\downarrow", "% harmful": "\\downarrow", "Risk bound %": "\\downarrow", "LOCO %": "\\downarrow",
    "LOCO-source %": "\\downarrow", "Proposition 1 bound %": "\\downarrow",
    "Conservative bound %": "\\downarrow", "Conservative %": "\\downarrow",
    "Total loss": "\\downarrow", "Naive harmful %": "\\downarrow",
    "Harmful % at BRIDGE yield": "\\downarrow", "Gate / fit (%)": "\\downarrow",
    "Fit models (s)": "\\downarrow", "Naive selection (s)": "\\downarrow",
    "Cross-fitted selection (s)": "\\downarrow", "Bootstrap (s)": "\\downarrow",
    "BRIDGE gate total (s)": "\\downarrow", "Inference, baseline (ms/row)": "\\downarrow",
    "Inference, 3-model ensemble (ms/row)": "\\downarrow",
    "Log-loss": "\\downarrow", "Probabilities <0.01 or >0.99 %": "\\downarrow",
    "Fit time per bag (s)": "\\downarrow",
    "Real: bound %": "\\downarrow", "Q: Proposition 1 bound %": "\\downarrow",
    "Yield/100": "\\uparrow", "Total gain": "\\uparrow",
    "Naive: harmful %": "\\downarrow", "BRIDGE: harmful %": "\\downarrow", "Reduction in harmful %, pp [95% CI]": "\\uparrow",
    "Real: total gain": "\\uparrow", "Q: total gain": "\\uparrow",
    "AUC": "\\uparrow", "Weighted F1 at 0.5": "\\uparrow",
    "Median POOL log-loss skill": "\\uparrow",
}
DIR_NOTE = r"$\downarrow$/$\uparrow$ = smaller/larger is better"
COLOR_NOTE = r"; row tint: green = pre-registered criterion met, red = not met (as in the last column)."
BOLD_NOTE = r"; \textbf{bold} = best value per column"
BOLD_NOTE_NONAIVE = r"; \textbf{bold} = best value per column, naive acceptance excluded"
MIXCRIT_NOTE = "; bound: one-sided 95\\% upper bound on the harmful proportion, the uncertainty measure of each rule; criterion: Table~\\ref{tab:criterion}."
QCRIT_NOTE = "; bound: one-sided 95\\% upper bound on the harmful proportion, the uncertainty measure of each rule; criterion: Table~\\ref{tab:criterion}."
CRIT_NOTE = "; bound: one-sided 95\\% upper bound on the harmful proportion, the uncertainty measure of each rule; criterion: Table~\\ref{tab:criterion}."


# Figure 1: harm of the naive rule vs. size of the promised validation gain (MINI-27)
m27 = RD("mini27_m1.csv", "mini27_m2.csv", "mini27_m3.csv")
b24 = RD("mini24_m1.csv", "mini24_m2.csv", "mini24_m3.csv")
d = m27.merge(b24[["dataset", "rep", "bridge"]], on=["dataset", "rep"]); d = d[d.cfg != "None"]
bins = [0, .005, .01, .02, .05, 1]; lab = ["(0,0.005]", "(0.005,0.01]", "(0.01,0.02]", "(0.02,0.05]", "(0.05,1]"]
d["band"] = pd.cut(d.dpool_best, bins, labels=lab)
g = d.groupby("band", observed=True).apply(lambda s: pd.Series({"n": len(s), "harm": 100 * (s.dref_deploy <= 0).mean(),
                                                               "k": int((s.dref_deploy <= 0).sum()), "bridge": int(s.bridge.sum())}))
d["harmful"] = d.dref_deploy <= 0
_q1 = pd.read_csv(HERE / "out" / "mini32_Q.csv")
_c1 = RD("mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"); _c1 = _c1[~_c1.kelas.isin(["error", "skip"])]; _c1 = _c1[_c1.metric == "nb20"]
panels = [("(A) 11 new datasets, weighted-F1", d.dpool_best, d.harmful, "#8e1b1b"),
          ("(B) Synthetic tasks, weighted-F1", _q1.pt_pool, _q1.ref <= 0, "#d2452f"),
          ("(C) Clinical tasks, net benefit", _c1.dpool, _c1.ref <= 0, "#f0a35e")]
_d1 = pd.concat([pd.read_csv(HERE / "out" / f"mini43_m{i}.csv") for i in (1, 2, 3)])
panels.append(("(D) 39 supporting datasets, weighted-F1", _d1.gain_pool.where(_d1.naive_accept == 1), _d1.gain_ref <= 0, "#c9788a"))
_e1 = pd.concat([pd.read_csv(HERE / "out" / f) for f in ("settingE_laptop.csv", "settingE_part2_laptop.csv")]); _e1 = _e1[_e1.status == "ok"]
_e1nv = _e1.naive_accept.astype(str).str.lower().isin(["true", "1"])
panels.append(("(E) 14 OpenML datasets, weighted-F1", _e1.gain_pool.astype(float).where(_e1nv), _e1.gain_ref.astype(float) <= 0, "#7b6fa8"))
names = ["All (= naive false-\nimprovement %)", "+0.5 or less", "+0.5 to +1", "+1 to +2", "more than +2"]
bins1 = [0, .005, .01, .02, 1]; lab1 = ["(0,0.005]", "(0.005,0.01]", "(0.01,0.02]", "(0.02,1]"]
fig, ax = plt.subplots(figsize=(5.6, 3.6)); y = np.array([4.25, 3, 2, 1, 0]); h = .17
for k, (ttl, gain, bad, col) in enumerate(panels):
    t = pd.DataFrame({"band": pd.cut(gain, bins1, labels=lab1), "bad": bad}).dropna().groupby("band", observed=False).bad.agg(["size", "sum"])
    pct = np.concatenate([[100 * t["sum"].sum() / t["size"].sum()], (100 * t["sum"] / t["size"]).to_numpy()]); off = (2 - k) * h
    ax.barh(y + off, pct, h, color=col, label=ttl)
    for yi, v in zip(y + off, pct):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=5.8)
ax.axhline(3.62, color="#999999", lw=.6, ls="--")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7); ax.set_xlim(0, 80); ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_ylabel("Validation score increase\n(0-100 scale)", fontsize=7)
ax.set_xlabel("False improvements among accepted changes, on new data (%)", fontsize=7.2)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="x", labelsize=6.5)
ax.legend(fontsize=6.3, frameon=False, loc="lower right")
fig.tight_layout()
save(fig, "fig1_harm_vs_validation_gain")

# Figure 2: yield-matched comparison (real data and population Q)
q25 = RD("mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv")
fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
for ax, (nm, m, Ns) in zip(axs, (("(A) 11 new datasets", b24, range(50, 576, 25)), ("(B) Synthetic tasks", q25, range(50, 701, 25)))):
    stat = {"BRIDGE": m.lo95, "Corrected resampled t": -m.p_crt_final, "5x2cv t": -m.p_5x2, "McNemar (same data)": -m.p_mcn_pool}
    ok = m.cfg != "None"
    for (k, s), c in zip(stat.items(), ("#1f4e79", "#e69138", "#6aa84f", "#999999")):
        order = s[ok].sort_values(ascending=False).index
        ax.plot(list(Ns), [int((m.loc[order[:N]].ref <= 0).sum()) for N in Ns], label=k, color=c, lw=1.6 if k == "BRIDGE" else 1)
    ax.set_title(nm, fontsize=8.5); ax.set_xlabel("Number of accepted changes (equal yield)"); ax.set_ylabel("False improvements accepted $\\downarrow$")
axs[0].legend(fontsize=6.5, frameon=False)
save(fig, "fig2_yield_matched")

# Figure 3: clinical, harm by metric, naive vs BRIDGE (MINI-30)
m30 = RD("mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"); m30 = m30[~m30.kelas.isin(["error", "skip"])]
mets = [("nb20", "Net benefit\n$p_t$=0.2"), ("bacc", "Balanced\naccuracy"), ("wF1", "Weighted F1"), ("nb10", "Net benefit\n$p_t$=0.1"), ("F2", "F2")]
nk = np.array([[((m30.metric == k) & (m30.cfg != "None") & (m30.ref <= 0)).sum(), ((m30.metric == k) & (m30.cfg != "None")).sum()] for k, _ in mets])
bk = np.array([[((m30.metric == k) & (m30.lo95 > .005) & (m30.ref <= 0)).sum(), ((m30.metric == k) & (m30.lo95 > .005)).sum()] for k, _ in mets])
nv, bv = 100 * nk[:, 0] / nk[:, 1], 100 * bk[:, 0] / np.maximum(bk[:, 1], 1)
(nlo, nhi), (blo, bhi) = wilson(nk[:, 0], nk[:, 1]), wilson(bk[:, 0], bk[:, 1])
fig, ax = plt.subplots(figsize=(4.6, 2.6)); x = np.arange(len(mets)); w = .38; ek = dict(lw=.7, ecolor="#333333")
ax.bar(x - w / 2, nv, w, label="Naive (validation improves)", color="#c0504d", yerr=[nv - nlo, nhi - nv], capsize=2, error_kw=ek)
ax.bar(x + w / 2, bv, w, label="BRIDGE", color="#1f4e79", yerr=[bv - blo, bhi - bv], capsize=2, error_kw=ek)
for i, (a, b) in enumerate(zip(nv, bv)):
    ax.text(i - w / 2, nhi[i] + .5, f"{a:.1f}", ha="center", fontsize=6.5); ax.text(i + w / 2, bhi[i] + .5, f"{b:.1f}", ha="center", fontsize=6.5)
ax.axhline(10, color="r", lw=.6, ls="--"); ax.set_xticks(x); ax.set_xticklabels([l for _, l in mets], fontsize=7); ax.set_ylabel("False improvements among accepted (%) $\\downarrow$")
ax.legend(fontsize=7, frameon=False); ax.set_title("12 clinical tasks, 6 sources (confirmatory run)", fontsize=8.5)
save(fig, "fig3_clinical_harm_by_metric")

# Figure 4: sensitivity / specificity change of accepted changes (net benefit 0.2)
s = m30[(m30.metric == "nb20") & (m30.lo95 > .005)]
fig, ax = plt.subplots(figsize=(3.4, 3))
ax.scatter(s.d_spes, s.d_sens, s=5, alpha=.4, color="#1f4e79"); ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
ax.set_xlabel("Δ specificity"); ax.set_ylabel("Δ sensitivity"); ax.set_title(f"Accepted changes, net benefit $p_t$=0.2 (n={len(s)})", fontsize=8)
save(fig, "fig4_sens_spec_accepted")

# Figure 5: training size, BRIDGE vs fixed thresholds (MINI-20/29)
u = pd.concat([RD(f"mini29_ukuran_{i}.csv") for i in range(3)], ignore_index=True)
u20 = RD("mini20_size_a.csv", "mini20_size_b.csv")
u["dpool"] = pd.to_numeric(u.dpool, errors="coerce")  # SKIP rows contain an empty value
U = u20.merge(u[["dataset", "TOTAL", "rep", "dpool"]], on=["dataset", "TOTAL", "rep"]); U = U[U.kelas != "skip"]
fig, axs = plt.subplots(1, 2, figsize=(7, 2.6)); sizes = [1000, 2430, 6000]
for rule, c in (("BRIDGE", "#1f4e79"), (.02, "#e69138"), (.025, "#6aa84f"), (.035, "#999999")):
    bnd, yl = [], []
    for T in sizes:
        x_ = U[U.TOTAL == T]; acc = x_.lo95 > .005 if rule == "BRIDGE" else (x_.cfg != "None") & (x_.dpool > rule)
        a = x_[acc]; bnd.append(100 * up(int((a.ref <= 0).sum()), len(a), len(x_), 1)); yl.append(100 * len(a) / len(x_))
    lb = rule if rule == "BRIDGE" else f"fixed threshold t={rule}"
    axs[0].plot(sizes, bnd, "o-", color=c, label=lb, lw=1.6 if rule == "BRIDGE" else 1); axs[1].plot(sizes, yl, "o-", color=c, lw=1.6 if rule == "BRIDGE" else 1)
axs[0].axhline(10, color="r", lw=.6, ls="--"); axs[0].set_ylabel("Upper bound on risk (%) $\\downarrow$"); axs[1].set_ylabel("Accepted per 100 decisions $\\uparrow$")
for ax in axs:
    ax.set_xlabel("Training rows"); ax.set_xticks(sizes)
axs[0].legend(fontsize=6.5, frameon=False)
save(fig, "fig5_training_size_vs_fixed_threshold")

def _pct(a, h):
    return f"{int(a):,} / {int(h)}" + (f" ({100 * h / a:.1f})" if a else "")
def _num(v):
    return "--" if v != v else f"{v:.2f}"
_t39 = pd.read_csv(HERE / "out" / "analisis_mini39.csv"); _t40 = pd.read_csv(HERE / "out" / "analisis_mini40.csv")
_arm = {"R": "(A) 11 new datasets", "Q": "(B) Synthetic tasks", "M": "(D) 3 multiclass datasets"}
tunrows = []
for d in (_t39[_t39.arm == "R"], _t39[_t39.arm == "Q"], _t40):
    for L in ("HGB", "DT", "SVC", "KNN"):
        nv = d[(d.learner == L) & (d.aturan == "naif")].iloc[0]; br = d[(d.learner == L) & (d.aturan == "BRIDGE")].iloc[0]
        tunrows.append([Raw(_arm[br.arm]), L, _pct(nv.terima, nv.harmful), _pct(br.terima, br.harmful), _num(br.batas), _num(br.LOCO),
                        _num(br.R0), f"{br.yield_:.1f}", str(br.status).startswith("LULUS")])
CAPS["tab12_tuning_learners"] = "Tuning submitted as an update on (A), (B) and three multiclass datasets of (D): BRIDGE lowers the harmful proportion in every cell."
tex(pd.DataFrame(tunrows, columns=["Data", "Learner", "Naive: accepted / harmful (%)", "BRIDGE: accepted / harmful (%)", "Bound (%)",
                                   "LOCO (%)", "R0 (%)", "Yield", "Criterion"]),
    "tab12_tuning_learners",
    "Each learner (HGB, DT, SVC, KNN) is tuned with Optuna on SEARCH, and the tuned model is the update against the same learner with default "
    "settings; one decision per trial (11 new datasets, 100 trials each; synthetic population, tasks T2900-T4299, one trial each) or per trial of "
    "the multiclass run (Fashion-MNIST, USPS, DryBean, 60 trials each; the two remaining datasets were not run, deviation recorded before the "
    "results were seen). Bound: KL-ratio bound on the real data, Proposition 1 bound on the synthetic population; LOCO and R0 are not defined on "
    "the synthetic population. Pre-registered criterion as in Table criterion. Pre-registrations MINI-39 and MINI-40.",
    "tab:tuning", natural=True)
print("tab12 written")


# Tables
def crit_real(m, col, ref="ref"):
    """Pre-registered criterion, real non-clinical runs (PRAREG MINI-20/21/22): KL-ratio bound, LOCO and the
    R0 power check all below 10%, yield >= 5 per 100, and at least 3 datasets with yield >= 5 per 100."""
    acc = m[col] == 1; wrong = m[ref] <= 0; NT = len(m); per = m.groupby("dataset").size()
    n, k = int(acc.sum()), int((acc & wrong).sum())
    r0 = max(up(0, int((acc & (m.dataset != x)).sum()), NT - nx, 1) if (acc & (m.dataset != x)).sum() else 1.0 for x, nx in per.items())
    dsy = sum(100 * (acc & (m.dataset == x)).sum() / nx >= 5 for x, nx in per.items())
    return bool(up(k, n, NT, 1) < .1 and SP.loco(acc, wrong, m.dataset, per, NT) < .1 and r0 < .1 and 100 * n / NT >= 5 and dsy >= 3)

t1 = SP.rule_table(b24, SP.RULES); t1.columns = ["Rule", "Accepted", "Harmful", "% harmful", "Risk bound %", "LOCO %", "Yield/100", "Total gain", "Total loss"]
t1 = t1.drop(columns=["Total loss"]); t1["Rule"] = [str(v)[:1].upper() + str(v)[1:] for v in t1["Rule"]]
t1["Meets criterion"] = [crit_real(b24, r) for r in SP.RULES]
_sm = m27.merge(b24[["dataset", "rep", "bridge"]], on=["dataset", "rep"]); _sm = _sm[(_sm.cfg != "None") & (_sm.dpool_best > 0) & (_sm.dpool_best <= 0.01)]
_small = f" Small validation gains ($0<\\hat\\Delta\\le0.01$): the naive rule accepts {len(_sm)} changes, {int((_sm.dref_deploy <= 0).sum())} of them harmful; BRIDGE accepts {int(_sm.bridge.sum())}."
tex(t1, "tab1_rules_identical_decisions",
    "Setting (A): 11 new datasets, 1,100 decisions; every rule judges the same decisions. Total gain: summed reference gain of the accepted updates." + _small,
    "tab:rules", rc=[yn_color(v) for v in t1["Meets criterion"]], pw={"Rule": 118},
    bold=("% harmful", "Risk bound %", "LOCO %", "Total gain"),
    bold_sel=[not str(v).lower().startswith("naive") for v in t1["Rule"]], bold_note=BOLD_NOTE_NONAIVE)

# Comparator rules on (C) and (D) (MINI-48; analysis_mini48.py)
_r48 = pd.read_csv(HERE / "out" / "analysis_mini48_rules.csv")
_nm48 = {"naive": "Naive (validation improves)", "mcnemar_pool": "McNemar on same POOL", "mcnemar_cf": "McNemar on cross-fit",
         "t5x2": r"$5\times2$~cv paired $t$", "crt_final": "Corrected resampled $t$", "crt": "Corrected $t$ + per-split re-selection",
         "boot0": "BRIDGE without margin", "ho23": "Holdout 2/3 + 1/3", "ho12": "Holdout 1/2 + 1/2", "bridge": "BRIDGE"}
def _cell48(r):
    return Raw(f"{int(r.accepted):,} / {int(r.harmful)} ({r.harm_pct:.1f})")
def _diff48(r):
    if pd.isna(r.bridge_minus_pp):
        return "--"
    s = "$^{*}$" if r.p_holm < .05 else ""
    return Raw(f"${r.bridge_minus_pp:+.1f}$ [{r.lo:+.1f}; {r.hi:+.1f}]{s}")
_rows48 = []
for k in _nm48:
    c, d_ = _r48[(_r48.setting == "C") & (_r48.rule == k)].iloc[0], _r48[(_r48.setting == "D") & (_r48.rule == k)].iloc[0]
    _rows48.append([Raw(_nm48[k]), _cell48(c), f"{c.kl:.2f}", bool(c.meets), _diff48(c), _cell48(d_), f"{d_.kl:.2f}", bool(d_.meets), _diff48(d_)])
CAPS["tab1b_rules_CD"] = "Acceptance rules on identical decisions in (C) and (D): several comparators also meet the criterion; BRIDGE's harmful proportion is lowest or tied, except against the 2/3 holdout in (D), where the difference is not significant."
tex(pd.DataFrame(_rows48, columns=["Rule", "(C) accepted / harmful (%)", "(C) bound %", "(C) criterion", "(C) BRIDGE minus rule, pp",
                                   "(D) accepted / harmful (%)", "(D) bound %", "(D) criterion", "(D) BRIDGE minus rule, pp"]),
    "tab1b_rules_CD",
    "(C): 12 clinical tasks, 2,400 decisions, net benefit at p_t = 0.2, LOCO by source; (D): 39 supporting datasets, 3,700 decisions, weighted-F1. "
    "Every rule judges the same decisions with the same deployed configuration, except the holdouts, which deploy the configuration they select. "
    "Current model HGB with the default candidate space. BRIDGE minus rule: difference in harmful proportion with a 95% cluster-bootstrap interval "
    "(tasks in (C), datasets in (D); B = 10,000); $^{*}$ = significant after the Holm correction over the comparators of a setting. "
    "Criterion: Table~\\ref{tab:criterion}. Script: analysis_mini48.py.",
    "tab:rulesCD", pw={"Rule": 104})
rows = []
for r in SP.RULES:
    a = q25[q25[r] == 1]; M = len(a); S_ = int((a.ref <= 0).sum()); Sc = int((a.kelas != "membaik").sum())
    rows.append([SP.NAMES[r], M, S_, round(100 * SP.prop1(S_, M), 2), round(100 * SP.prop1(Sc, M), 2), round(100 * M / len(q25), 2)])
d2 = pd.DataFrame(rows, columns=["Rule", "Accepted", "Harmful", "Proposition 1 bound %", "Conservative bound %", "Yield/100"]).drop(columns=["Conservative bound %"])
d2["Rule"] = [str(v)[:1].upper() + str(v)[1:] for v in d2["Rule"]]
d2["Rule"] = d2["Rule"].replace({"Corrected t + per-split re-selection": "Corrected t, re-selection"})
d2["Meets criterion"] = (d2["Proposition 1 bound %"] < 10) & (d2["Yield/100"] >= 5)  # criterion: PRAREG MINI-25
d2 = d2.rename(columns={"Proposition 1 bound %": "Bound %", "Meets criterion": "Criterion"})
tex(d2, "tab2_synthetic_population",
    "Setting (B): 1,400 synthetic tasks drawn independently from one population, one decision per task. Bound: Proposition 1. Conservative bounds: Table~\\ref{tab:chronology}.", "tab:synthetic",
    rc=[yn_color(v) for v in d2["Criterion"]], narrow=True,
    bold=("Bound %",),
    bold_sel=[not str(v).lower().startswith("naive") for v in d2["Rule"]], bold_note=BOLD_NOTE_NONAIVE, crit_note=QCRIT_NOTE)
rows = []
for k, lbl in mets:
    x_ = m30[m30.metric == k]; NT = len(x_); per = x_.groupby("sumber").size()
    for nm, acc in (("BRIDGE", x_.lo95 > .005), ("Naive", x_.cfg != "None")):
        a = x_[acc]; n, kk = len(a), int((a.ref <= 0).sum())
        lo = max(up(int((a[a.sumber != q].ref <= 0).sum()), int((a.sumber != q).sum()), NT - per[q], 1) if (a.sumber != q).sum() else 1 for q in per.index)
        tug = sum(100 * (a.dataset == t_).sum() / (x_.dataset == t_).sum() >= 5 for t_ in x_.dataset.unique())
        ok = bool(up(kk, n, NT, 1) < .1 and lo < .1 and 100 * n / NT >= 5 and tug >= 3)
        rows.append([lbl.replace("\n", " ").replace("$p_t$", "p_t"), nm, n, kk, int((a.kelas == "memburuk").sum()), round(100 * up(kk, n, NT, 1), 2), round(100 * lo, 2),
                     round(a.d_sens.mean(), 3) if n else np.nan, round(a.d_spes.mean(), 3) if n else np.nan, ok])
d5 = pd.DataFrame(rows, columns=["Metric", "Rule", "Accepted", "Harmful", "Clearly harmful", "Risk bound %", "LOCO-source %", "Δ sens.", "Δ spec.", "Meets criterion"])
_mn = {"Net benefit p_t=0.2": r"Net benefit, $p_t=0.2$", "Net benefit p_t=0.1": r"Net benefit, $p_t=0.1$", "F2": r"$F_2$"}
c5 = []
for mt, g in d5.groupby("Metric", sort=False):
    br, nv = g[g.Rule == "BRIDGE"].iloc[0], g[g.Rule == "Naive"].iloc[0]
    c5.append([Raw(_mn.get(mt, _esc(mt))), f"{100 * nv.Harmful / nv.Accepted:.1f}", f"{int(br.Accepted):,}", f"{100 * br.Harmful / max(br.Accepted, 1):.1f}",
               _num(br["Risk bound %"]), _num(br["LOCO-source %"])])
assert d5[d5.Rule == "BRIDGE"]["Meets criterion"].all()  # guard: the caption states the criterion is met on every metric
c5 = pd.DataFrame(c5, columns=["Metric", "Naive harmful %", "BRIDGE accepted", "BRIDGE harmful %", "Bound %", "LOCO-source %"])
CAPS["tab5_clinical"] = "Clinical tasks (C) by metric: BRIDGE meets the criterion on every metric."
tex(c5, "tab5_clinical",
    "12 tasks from 6 independent sources, 200 trials per task (2,400 decisions per metric); primary metric: net benefit at $p_t=0.2$. Bound and LOCO-source for BRIDGE; "
    "changes in sensitivity and specificity: Fig.~\\ref{fig:sens_spec}. Criterion: Table~\\ref{tab:criterion}.",
    "tab:clinical", narrow=True,
    hdroverrides={"Naive harmful %": Raw(r"\begin{tabular}[b]{@{}c@{}}Naive\\harmful \%\end{tabular}"),
                  "BRIDGE accepted": Raw(r"\begin{tabular}[b]{@{}c@{}}BRIDGE\\accepted\end{tabular}"),
                  "BRIDGE harmful %": Raw(r"\begin{tabular}[b]{@{}c@{}}BRIDGE\\harmful \%\end{tabular}"),
                  "LOCO-source %": Raw(r"\begin{tabular}[b]{@{}c@{}}LOCO-\\source \%\end{tabular}")})
_t5 = OUT / "tab5_clinical.tex"; _t5.write_text(_t5.read_text().replace(r"fig:sens\_spec", "fig:sens_spec").replace(r"\\ref", r"\ref"))
TASKS = [  # population, outcome, prediction time / predictors, hypothetical action, from health_data*.py
    ("mammography", "Screening mammograms (OpenML 310)", "Malignant calcification", "Image-derived features at reading", "Recall for biopsy"),
    ("actg175", "HIV-1 trial participants (UCI 890)", "Death or AIDS event during follow-up", "Baseline covariates only", "Intensify monitoring"),
    ("nhanes_*", "US adults ≥20 y, NHANES 2017–18", "Self-reported: hypertension, diabetes, asthma, CHD", "Exam & questionnaire, same visit", "Refer for testing"),
    ("brfss_*", "US adults, BRFSS 2015", "Self-reported: stroke, asthma, kidney disease, depression", "Survey responses, same interview", "Refer for screening"),
    ("covid_mx", "Confirmed COVID-19 patients, Mexico", "In-hospital/registered death", "Admission data (comorbidities, triage)", "Escalate care"),
    ("wids_icu", "ICU admissions, GOSSIS (WiDS 2020)", "Hospital death", "First-24 h ICU measurements", "Escalate care"),
]
tex(pd.DataFrame(TASKS, columns=["Task", "Population", "Outcome", "Predictors / timing", "Hypothetical action"]),
    "tab6_clinical_tasks", "Clinical tasks: population, outcome, timing of predictors and hypothetical action. The COVID-19 cohort is the open data released by the Mexican Secretaría de Salud. NHANES outcomes are physician diagnoses as reported by the respondent; the NHANES and BRFSS survey tasks are cross-sectional health-related classification (predictors from the same visit or interview), not prognosis.", "tab:tasks", natural=True,
    hdroverrides={"Predictors / timing": "Predictors / timing", "Hypothetical action": "Hypothetical action"})
print("written:", sorted(p.name for p in OUT.iterdir()))


# Figure 5 (outline numbering): cross-fitted estimate vs deployed truth (MINI-27)
x = m27.merge(b24[["dataset", "rep", "bridge"]], on=["dataset", "rep"]); x = x[x.cfg != "None"]
fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
for ax, col, nm in ((axs[0], "dpool_best", "Naive estimate (full POOL)"), (axs[1], "dcf", "BRIDGE cross-fitted estimate")):
    acc = x.bridge == 1
    ax.scatter(x.loc[~acc, col], x.loc[~acc, "dref_deploy"], s=4, alpha=.35, color="#999999", label="rejected by BRIDGE")
    ax.scatter(x.loc[acc, col], x.loc[acc, "dref_deploy"], s=6, alpha=.6, color="#1f4e79", label="accepted by BRIDGE")
    lim = [min(x[col].min(), x.dref_deploy.min()), max(x[col].max(), x.dref_deploy.max())]
    ax.plot(lim, lim, "k--", lw=.6); ax.axhline(0, color="r", lw=.6)
    ax.set_xlabel(nm); ax.set_title(f"mean bias {(x[col] - x.dref_deploy).mean():+.4f}", fontsize=8)
axs[0].set_ylabel("Δ wF1 on locked reference (deployed)"); axs[1].legend(fontsize=6.5, frameon=False)
save(fig, "fig5b_crossfit_vs_truth")

# Table 1: frozen hyperparameters of BRIDGE
CAPS["tabS13_hyper_basis"] = "Frozen parameters of BRIDGE and of the evaluation protocol: details and the basis of each value."
CAPS["tab1b_hyperparameters"] = "Frozen parameters of BRIDGE and of the evaluation protocol."
tex(pd.DataFrame([
    [Raw("Cross-fit folds $F$"), Raw("$3$"), Raw("fixed during development; see the chronology in Table~\\ref{tab:chronology}; sensitivity in Appendix~\\ref{app:sens}"), Raw(r"set in development; $F\in\{2,3,5\}$ on $\mathcal{Q}$ (Appendix~\ref{app:sens})")],
    [Raw("Bootstrap replicates $B$"), Raw("$20{,}000$"), Raw("multinomial over (y, baseline, candidate) counts"), Raw(r"set in development; Monte Carlo error over $20$ bootstrap seeds, and $N_{\mathrm{boot}}=2{,}000$ (Appendix~\ref{app:sens})")],
    [Raw("Lower quantile"), Raw("$5\\%$"), Raw("lower bootstrap quantile of the metric difference, used as the gate's decision statistic (distinct from the certificate error level $\\delta=0.05$)"), Raw(r"set in development; $q\in\{2.5,5,10\}\%$ on $\mathcal{Q}$ and on (A), (D) (Tables~\ref{tab:sens31}, \ref{tab:qepsreal})")],
    [Raw("Margin"), Raw("$0.005$"), Raw("$\\varepsilon=0.005$; criterion met at the tested margins 0.0025, 0.005, 0.0075 and 0.01 wherever it is met at 0.005 (Table~\\ref{tab:marginsweep})"), Raw(r"set in development; four margins in all settings (Tables~\ref{tab:marginsweep}, \ref{tab:marginpaired}, \ref{tab:qepsreal})")],
    [Raw("HGB base learner"), Raw("learning rate $0.1$; at most $300$ boosting iterations; at most $31$ leaf nodes; at least $20$ samples per leaf; no L2 regularization; early stopping after $10$ iterations without improvement beyond the implementation tolerance"), Raw("implementation settings, fixed before any evaluation; early stopping uses a $10\\%$ internal validation subset of the bag"), Raw(r"library settings, not tuned; tuned HGB tested (Tables~\ref{tab:tuning}, \ref{tab:tuningtests})")],
    [Raw("Bagged base models"), Raw("up to $32$ bag draws, $80\\%$ of SEARCH"), Raw("HGB (baseline), DT, SVC; in the binary protocol a drawn bag is kept only if every class has at least $10$ examples; in the multiclass protocol (MINI-22) each class that occurs must have at least $2$ examples; kept bags are averaged; run labels: Appendix~\\ref{app:prereg}"), Raw(r"set in development; $16/32/64$ bags on $\mathcal{Q}$ (Table~\ref{tab:citedruns})")],
    [Raw("DT and SVC members"), Raw("DT: scikit-learn defaults (Gini, no depth limit); SVC: $C=1$, RBF kernel, $\\gamma=$ scale, Platt probabilities"), Raw("library settings, fixed before any evaluation; members chosen stepwise in development (Table~\\ref{tab:devruns})"), Raw(r"chosen stepwise in development, SVC over KNN (Table~\ref{tab:devruns})")],
    [Raw("Preprocessing"), Raw("numeric: median imputation and standardization; categorical: most-frequent imputation and ordinal encoding ($-1$ for unseen)"), Raw("fitted per bag on SEARCH only"), Raw(r"standard; fitted on SEARCH only (Table~\ref{tab:leakage})")],
    [Raw("Candidate space"), Raw("$173$ configurations (binary default)"), Raw("$18$ threshold variants of HGB (grid $0.05$--$0.95$, step $0.05$, without $0.5$); $57$ HGB+DT mixtures (weights $0.25/0.5/0.75$ $\\times$ $19$ thresholds); $98$ HGB+DT+SVC mixtures ($14$ simplex weights, step $0.25$, $\\times$ thresholds $0.30$--$0.60$); denser spaces in Table~\\ref{tab:kspace}; other protocols use separate prespecified spaces"), Raw(r"set in development; $1{,}825$ and $11{,}249$ configurations (Table~\ref{tab:kspace})")],
    [Raw("Multiclass candidates"), Raw("$14+8K$ configurations for $K$ classes"), Raw("$14$ simplex weights over HGB/DT/SVC (step $0.25$, without pure HGB), argmax; class scales $s\\in\\{0.5,0.7,1.4,2\\}$ on one class at a time, for HGB and for the $0.5$/$0.5$ HGB+DT average; ties: smallest class index, and candidates within $10^{-12}$ take the first in list order"), Raw(r"set in multiclass development (Table~\ref{tab:chronology})")],
    [Raw("Training rows"), Raw("$1{,}000$ / $2{,}430$ / $6{,}000$"), Raw("SEARCH $=25.9\\%$, POOL $=$ remainder; $6{,}000$ is a prespecified training-size variant"), Raw(r"$2{,}430$ = SEARCH $630$ + POOL $1{,}800$ from development; $1{,}000/2{,}430/6{,}000$ rows (Table~\ref{tab:general}); SEARCH $360/630/1{,}000$ on $\mathcal{Q}$ (Table~\ref{tab:citedruns})")],
    [Raw("Locked reference"), Raw("real data: $\\lfloor 0.3N\\rfloor$ capped at $8{,}000$ for $N<40{,}000$, and $\\min(N-20{,}000,\\,100{,}000)$ for $N\\ge40{,}000$; $\\mathcal{Q}$: $100{,}000$ of a $103{,}000$-row batch"), Raw("fixed per-task seed on the real data; never used for training or selection"), Raw(r"precision rule from development; reference standard error and two further reference seeds (Table~\ref{tab:citedruns})")],
    [Raw("Trials and seeds"), Raw("$100$ trials per dataset, $200$ per clinical task, $60$ per multiclass dataset, $1$ per task of $\\mathcal{Q}$; one fresh seed per trial"), Raw("the seed sets the training rows, the SEARCH/POOL split, the bags, the folds and the bootstrap draws; learners use a fixed random state ($42$); the locked reference uses one pre-registered seed per task"), Raw(r"$100$ trials from the power check (LOCO$_0$ $18.2\%$ at $50$ trials, $0.8\%$ at $100$; Table~\ref{tab:devruns}); $200$ per clinical task after LOCO-source failed at $100$ (Table~\ref{tab:chronology}); $60$ per multiclass dataset as pre-registered for the multiclass confirmation ($40$ in development); $1{,}400$ tasks of $\mathcal{Q}$ = the generator seeds not used by the pilots (Appendix~\ref{app:datasets}); reference seed replicated twice (Table~\ref{tab:citedruns})")],
], columns=["Parameter", "Value", "Note", "Basis: where set; where tested"]), "tabS13_hyper_basis", "Basis: where each value was set and where it was tested. Set in development: chosen on the 18 development datasets and the development runs (Table~\\ref{tab:devruns}) and frozen before any confirmatory run (Table~\\ref{tab:chronology}).", "tab:hyperbasis", pw={"Value": 112, "Note": 140, "Basis: where set; where tested": 156}, fixed=True, hdroverrides={"Basis: where set; where tested": Raw(r"Basis (where set; where tested)")})

# Table 4 (main text): frozen parameters, values only; details and basis in Table hyperbasis
tex(pd.DataFrame([
    [Raw(r"Cross-fit folds $F$"), Raw(r"$3$")],
    [Raw(r"Bootstrap replicates $N_{\mathrm{boot}}$"), Raw(r"$20{,}000$")],
    [Raw(r"Lower quantile $q$"), Raw(r"$5\%$")],
    [Raw(r"Margin $\varepsilon$"), Raw(r"$0.005$")],
    [Raw(r"Base learner"), Raw(r"HGB, library settings with early stopping")],
    [Raw(r"Bagged base models"), Raw(r"$32$ bags of $80\%$ of SEARCH")],
    [Raw(r"Ensemble members"), Raw(r"DT and SVC, library settings")],
    [Raw(r"Preprocessing"), Raw(r"fitted on SEARCH only")],
    [Raw(r"Candidate space"), Raw(r"$173$ (binary); $14+8K$ for $K$ classes")],
    [Raw(r"Training rows"), Raw(r"$2{,}430$ (SEARCH $630$, POOL $1{,}800$); $1{,}000$ when fewer rows")],
    [Raw(r"Locked reference"), Raw(r"one pre-registered seed per task; size by a fixed rule")],
    [Raw(r"Trials"), Raw(r"$100$ per dataset, $200$ per clinical task, $60$ per multiclass dataset, $1$ per task of $\mathcal{Q}$; one fresh seed per trial")],
], columns=["Parameter", "Value"]), "tab1b_hyperparameters",
    "Details of every value and its basis (where it was set and where it was tested): Table~\\ref{tab:hyperbasis} in Appendix~\\ref{app:sens}.",
    "tab:hyper", narrow=True)

# Table 4: BRIDGE vs honest holdout (MINI-26)
h = RD("mini26_nyata_m1.csv", "mini26_nyata_m2.csv", "mini26_nyata_m3.csv"); hq = RD("mini26_q_m1.csv", "mini26_q_m2.csv", "mini26_q_m3.csv")
rows = []
for r in ("bridge", "ho23", "ho12"):
    rc = "ref" if r == "bridge" else r + "_ref"
    a = h[h[r] == 1]; aq = hq[hq[r] == 1]
    rows.append([SP.NAMES[r], len(a), int((a[rc] <= 0).sum()), round(100 * up(int((a[rc] <= 0).sum()), len(a), len(h), 1), 2), round(a[rc].sum(), 2),
                 len(aq), int((aq[rc] <= 0).sum()), round(100 * SP.prop1(int((aq[rc] <= 0).sum()), len(aq)), 2), round(aq[rc].sum(), 2)])
b, l23, l12 = h[h.bridge == 1], h[h.ho23 == 1], h[h.ho12 == 1]
bq, q23, q12 = hq[hq.bridge == 1], hq[hq.ho23 == 1], hq[hq.ho12 == 1]
rat_r = f"{len(b) / len(l23):.2f} / {len(b) / len(l12):.2f}"
rat_q = f"{len(bq) / len(q23):.2f} / {len(bq) / len(q12):.2f}"
hrows = []
for k, r in enumerate(rows):
    nm = "BRIDGE" if k == 0 else ("Holdout 2/3 + 1/3" if k == 1 else "Holdout 1/2 + 1/2")
    hrows.append(["(A)", nm, r[1], r[2], r[3], r[4]])
for k, r in enumerate(rows):
    nm = "BRIDGE" if k == 0 else ("Holdout 2/3 + 1/3" if k == 1 else "Holdout 1/2 + 1/2")
    hrows.append(["(B)", nm, r[5], r[6], r[7], r[8]])
_h48 = pd.read_csv(HERE / "out" / "analysis_mini48_rules.csv")
for _S in ("C", "D"):
    for _rule, _nm in (("bridge", "BRIDGE"), ("ho23", "Holdout 2/3 + 1/3"), ("ho12", "Holdout 1/2 + 1/2")):
        _r = _h48[(_h48.setting == _S) & (_h48.rule == _rule)].iloc[0]
        hrows.append([f"({_S})", _nm, int(_r.accepted), int(_r.harmful), round(_r.kl, 2), round(_r.total_gain, 2)])
d4 = pd.DataFrame(hrows, columns=["Setting", "Rule", "Accepted", "Harmful", "Risk bound %", "Total gain"])
d4["Setting"] = [v if k % 3 == 0 else "" for k, v in enumerate(d4["Setting"])]
CAPS["tab4_holdout"] = "BRIDGE against an honest holdout on the same labeled data: more accepted updates at a similar or lower risk bound."
tex(d4, "tab4_holdout",
    f"(A) 11 new datasets; (B) synthetic tasks; (C) clinical tasks; (D) 39 supporting datasets. Holdout a + b: a share of POOL selects, b tests. The holdout keeps part of POOL to test the chosen update; BRIDGE reuses all of POOL through cross-fitting. BRIDGE accepts {rat_r.replace(' / ', ' and ')} times as many updates as the two holdouts in (A), "
    f"{rat_q.replace(' / ', ' and ')} times in (B), 1.09 and 1.21 times in (C), and 1.35 and 1.54 times in (D). Risk bound: KL-ratio bound in (A), (C), and (D); Proposition 1 bound in (B).",
    "tab:holdout", crit_note=MIXCRIT_NOTE, narrow=True)

# Table 5: Generalization
def gen_row(s_, lbl):
    """One generalization setting: counts, KL-ratio bound, LOCO and the pre-registered criterion (crit_real)."""
    s_ = s_.assign(acc=(s_.lo95 > .005).astype(int)); acc = s_.acc == 1; wrong = s_.ref <= 0
    per = s_.groupby("dataset").size(); NT = len(s_); n, k = int(acc.sum()), int((acc & wrong).sum())
    return [lbl, n, k, round(100 * up(k, n, NT, 1), 2), round(100 * SP.loco(acc, wrong, s_.dataset, per, NT), 2),
            round(100 * n / NT, 2), crit_real(s_, "acc")]
rows = []
m21 = RD("mini21_m1.csv", "mini21_m2.csv", "mini21_m3.csv")
for des, lbl in (("union_wF1", "(A) weighted F1"), ("union_bacc", "(A) balanced accuracy"), ("union_macroF1", "(A) macro-F1"),
                 ("fsthr_KNN", "(A) feature selection + threshold (KNN)"), ("fsthr_SVC", "(A) feature selection + threshold (SVC)")):
    rows.append(gen_row(m21[m21.design == des], lbl))
m20 = RD("mini20_size_a.csv", "mini20_size_b.csv"); m20 = m20[m20.kelas != "skip"]
for T, s_ in m20.groupby("TOTAL"):
    rows.append(gen_row(s_, f"(D) {26 if T == 1000 else 27} datasets, {T:,} training rows"))
m22 = RD("mini22_m1.csv", "mini22_m2.csv", "mini22_m3.csv")
rows.append(gen_row(m22, "(D) 5 multiclass datasets"))
d5b = pd.DataFrame(rows, columns=["Setting", "Accepted", "Harmful", "Risk bound %", "LOCO %", "Yield/100", "Meets criterion"])
tex(d5b, "tab5b_generalisation",
    "Generalization of the frozen gate across metrics, policies, training sizes and multiclass tasks: the criterion is met in every setting except feature selection with SVC, where LOCO exceeds 10%.",
    "tab:general", pw={"Setting": 165})
print("extra assets written")

# Appendix table: exact data partition per dataset/task (dataset_splits.py)
sp = pd.read_csv(HERE / "out" / "dataset_splits.csv")
PHCODE = {"confirmation (34)": "(D)", "new binary (11)": "(A)", "new multiclass (5)": "(D) mc", "clinical (12 tasks)": "(C)"}
DM = {  # compact domain and one-line description (about 20 chars) of each public source; loaders in the code repo
    # confirmation (34)
    "miniboone": ("Physics", "MiniBooNE particle ID"), "compas": ("Justice", "recidivism risk"),
    "credit-default-tw": ("Finance", "Taiwan credit default"), "noshow": ("Healthcare", "appointment no-shows"),
    "cardio": ("Medical", "cardiovascular risk"), "heloc": ("Finance", "home equity credit"),
    "lt-vehicle-loan": ("Finance", "vehicle-loan approval"), "codexglue": ("Software", "code defect detection"),
    "kickstarter": ("Crowdfund", "project success"), "bank-churn": ("Banking", "customer churn"),
    "ecom-shipping": ("E-commerce", "shipping outcome"), "employee": ("HR", "employee attrition"),
    "amazon-access": ("HR", "employee access"), "kick": ("Automotive", "used-car auction bad buy"),
    "porto-seguro": ("Insurance", "car claim prediction"), "relevant-images": ("Vision", "image relevance"),
    "newspaper-churn": ("Media", "subscriber churn"), "online-shoppers": ("E-commerce", "purchase intention"),
    "give-some-credit": ("Finance", "credit default risk"), "hmeq": ("Finance", "home-equity default"),
    "avida-hil6": ("Biology", "antibody-antigen binding"), "avida-cov2": ("Biology", "antibody-antigen binding"),
    "acs-drem": ("Social", "public health-coverage survey"), "fake-job": ("Employment", "fake job detection"),
    "sepsis": ("Medical", "sepsis risk (ICU)"), "dota2": ("Gaming", "player skill"),
    "santander": ("Banking", "customer satisfaction"), "home-credit": ("Finance", "loan default risk"),
    "coil2000": ("Insurance", "caravan insurance"), "bnp-paribas": ("Insurance", "insurance claim"),
    "polish-bankruptcy": ("Finance", "company bankruptcy"), "taiwan-bankruptcy": ("Finance", "company bankruptcy"),
    "naticusdroid": ("Cyber", "Android malware"), "hr-analytics": ("HR", "employee attrition"),
    # new binary (11)
    "pc3": ("Software", "software defect prediction (NASA)"), "MIC": ("Medical", "myocardial infarction complications"),
    "Marketing_Campaign": ("Marketing", "campaign response"), "Pumpkin_Seeds": ("Agriculture", "pumpkin-seed variety"),
    "seismic-bumps": ("Mining", "seismic bumps"), "Rice": ("Agriculture", "rice variety"),
    "Census-Income-KDD": ("Social", "adult income census"), "Santander2016": ("Banking", "customer satisfaction"),
    "KDD98": ("Social", "donation response"), "LICD": ("Employment", "labor rights, workplace safety"),
    "Homesite": ("Insurance", "quote conversion"),
    # new multiclass (5)
    "Fashion-MNIST": ("Vision", "fashion articles"), "DryBean": ("Agriculture", "dry-bean variety"),
    "USPS": ("Vision", "handwritten digits"), "mnist_784": ("Vision", "handwritten digits"),
    "Kuzushiji-MNIST": ("Vision", "cursive characters"),
    # clinical (12)
    "mammography": ("Radiology", "mammogram screening"), "actg175": ("Clinical", "HIV-1 trial outcome"),
    "nhanes_hipertensi": ("Survey", "NHANES hypertension"), "nhanes_diabetes": ("Survey", "NHANES diabetes"),
    "nhanes_asma": ("Survey", "NHANES asthma"), "nhanes_pjk": ("Survey", "NHANES coronary disease"),
    "brfss_stroke": ("Survey", "BRFSS stroke"), "brfss_asma": ("Survey", "BRFSS asthma"),
    "brfss_ginjal": ("Survey", "BRFSS kidney disease"), "brfss_depresi": ("Survey", "BRFSS depression"),
    "covid_mx": ("Public Health", "COVID-19 mortality"), "wids_icu": ("ICU", "ICU mortality"),
}
DGROUP = {  # each fine domain label belongs to exactly one of the seven groups used in the text
    "Health": ["Healthcare", "Medical", "Radiology", "Clinical", "Survey", "Public Health", "ICU"],
    "Finance, banking and insurance": ["Finance", "Banking", "Insurance", "Crowdfund"],
    "Agriculture and mining": ["Agriculture", "Mining"],
    "Cyber and software": ["Software", "Cyber"],
    "Commerce, services and HR": ["E-commerce", "Marketing", "HR", "Employment", "Automotive"],
    "Social, legal and media": ["Social", "Justice", "Media", "Gaming"],
    "Science and vision": ["Physics", "Biology", "Vision"]}
lab2grp = {l: g for g, ls in DGROUP.items() for l in ls}
assert all(v[0] in lab2grp for v in DM.values()), [v for v in DM.values() if v[0] not in lab2grp]
_real = sp[sp["group"].isin(["confirmation (34)", "new binary (11)", "new multiclass (5)"])]["dataset"]
_used = {lab2grp[DM[k][0]] for k in _real}; assert len(_used) == 7, _used
GROUP_NOTE = (f" The {len({v[0] for v in DM.values()})} domain labels map to seven groups, all represented among the 50 real datasets: "
              + "; ".join(f"{g} ({', '.join(ls)})" for g, ls in DGROUP.items()) + ".")
sp = sp[["group", "dataset", "source", "N", "features", "minority_pct", "reference", "search", "pool"]].copy()
sp["Phase"] = sp["group"].map(PHCODE)
sp["Split (SEARCH/POOL)"] = (sp["search"].astype(int).astype(str) + "/" + sp["pool"].astype(int).astype(str))
dm = sp["dataset"].map(lambda k: DM.get(k, ("", "")))
sp["Domain"], sp["Description"] = dm.str[0], dm.str[1]
sp = sp[["Phase", "dataset", "Domain", "Description", "source", "N", "features", "minority_pct", "reference", "Split (SEARCH/POOL)"]]
sp.columns = ["Group", "Dataset", "Domain", "Description", "Source", "N", "Features", "Minority %", "Reference", "Split (SEARCH/POOL)"]
# the 18 development datasets (mini05_ensemble.DS: exp40/exp41 datasets with N < 40,000) on which the gate and candidate space were frozen
DEV18 = {"amazon-access", "avida-cov2", "bank-churn", "codexglue", "coil2000", "compas", "credit-default-tw", "ecom-shipping", "employee",
         "fake-job", "heloc", "hmeq", "hr-analytics", "naticusdroid", "newspaper-churn", "online-shoppers", "polish-bankruptcy", "taiwan-bankruptcy"}
assert len(DEV18) == 18 and DEV18 <= set(sp.Dataset)
sp["Dataset"] = [d + " (dev)" if d in DEV18 else d for d in sp.Dataset]
tex(sp, "tabA_dataset_splits",
    "Exact data partition per dataset or task: the locked reference and the per-trial SEARCH/POOL split; the remaining rows form the eligible training pool from which the TRAIN rows of each trial are drawn (rows not drawn are unused in that trial). Group (62 data units = 11 + 12 + 39): (A) = the 11 new binary datasets of setting (A); (C) = the 12 clinical tasks of setting (C); (D) = the 39 supporting datasets, namely the 34 confirmation datasets (including the 18 development datasets, marked (dev), on which the gate and candidate space were frozen) and the 5 new multiclass datasets, marked (D) mc. Setting (B), the synthetic population, is generated rather than drawn from a dataset and is described in this appendix. Colors: (A) yellow, (C) orange, (D) purple, (D) mc teal. Default split uses 2,430 training rows (SEARCH 630 / POOL 1,800); tasks marked 259/741 use 1,000 training rows. Trials per task: 100 ((A) and (D)), 200 ((C)), 60 ((D) mc). Domains and one-line descriptions briefly characterize the public sources; exact definitions live in the code-repository loaders." + GROUP_NOTE,
    "tab:splits", groupcol="Group", grouppal={"(D)": ("F1E9FA",), "(A)": ("FFF6D2",), "(D) mc": ("E2F5F2",), "(C)": ("FFEEDD",)},
    pw={"Dataset": 74, "Domain": 54, "Description": 92, "Source": 46})
print("appendix split table written")


# Ablation (MINI-32) and candidate-space size (MINI-35): run analisis_mini32.py / analisis_mini35.py first
ab = pd.read_csv(HERE / "out" / "analisis_mini32.csv")
ab["Data"] = ab.data.map(lambda s: "Real" if s.startswith("nyata") else "Q")
ab["Evidence"] = ab.sumber.map({"POOL-sama": "same POOL", "cross-fit": "cross-fitted"})
ab["Statistic"] = ab.statistik.map({"titik": "point", "bootstrap": "bootstrap 5% bound"})
ab["Rule"] = ab.nama.fillna("").map({"naif": "naive", "VALMARGIN 0,005": "fixed margin", "boot0": "BRIDGE without margin", "BRIDGE": "BRIDGE", "": ""})
ab["Rule"] = ab["Rule"].map(lambda v: v[:1].upper() + v[1:] if v else v)
_ab48 = pd.read_csv(HERE / "out" / "analysis_mini48_ablation.csv")
abrows = []
for (ev, st, mg), g in ab.groupby(["Evidence", "Statistic", "margin"], sort=False):
    ra, rb = g[g.Data == "Real"].iloc[0], g[g.Data == "Q"].iloc[0]
    _on = lambda b_: Raw(r"$\checkmark$") if b_ else "--"
    _nm = ra.Rule or {("same POOL", True, False): "Same-POOL bootstrap", ("same POOL", True, True): "Same-POOL cell",
                       ("cross-fitted", False, False): "Cross-fit, point", ("cross-fitted", False, True): "Cross-fit, point, margin"}[(ev, st.startswith("bootstrap"), mg > 0)]
    _ev48 = "cross-fit" if ev == "cross-fitted" else ev
    _st48 = "bootstrap" if st.startswith("bootstrap") else "point"
    _cd = [_ab48[(_ab48.setting == S) & (_ab48.evidence == _ev48) & (_ab48.statistic == _st48) & (_ab48.margin == int(mg > 0))].iloc[0] for S in ("C", "D")]
    abrows.append([_nm, _on(ev == "cross-fitted"), _on(st.startswith("bootstrap")), _on(mg > 0),
                   f"{ra.pct_salah:.2f}", f"{ra.batas:.2f}", bool(ra.lulus), f"{rb.pct_salah:.2f}", f"{rb.batas:.2f}", bool(rb.lulus),
                   f"{_cd[0].harm_pct:.2f}", f"{_cd[0].kl:.2f}", bool(_cd[0].meets),
                   f"{_cd[1].harm_pct:.2f}", f"{_cd[1].kl:.2f}", bool(_cd[1].meets)])
t = pd.DataFrame(abrows, columns=["Known as", "Cross-fit", "Bootstrap bound", "Margin", "(A) harmful %", "(A) bound %", "(A) criterion",
                                  "(B) harmful %", "(B) bound %", "(B) criterion", "(C) harmful %", "(C) bound %", "(C) criterion",
                                  "(D) harmful %", "(D) bound %", "(D) criterion"])
CAPS["tab9_ablation_factorial"] = "Component analysis on (A) to (D): the bootstrap bound and the margin lower the harmful proportion most."
tex(t, "tab9_ablation_factorial",
    "Each component switched on ($\\checkmark$) or off (--): cross-fitted evidence instead of the same POOL; the 5% bootstrap lower bound instead of the point estimate; "
    "the margin 0.005 instead of 0. Same-POOL cell: the rule that chooses and measures on the same POOL with the bootstrap bound and the margin. Every cell judges the same deployed configuration; current model HGB with the default candidate space. "
    "Bound: KL-ratio bound in (A), (C), and (D), and Proposition 1 bound in (B). Paired intervals for the same-POOL cell against BRIDGE: Table~\\ref{tab:ci}. "
    "Counts, LOCO, yield and total gain: out/analisis_mini32.csv and out/analysis_mini48_ablation.csv. Criterion: Table~\\ref{tab:criterion}.",
    "tab:ablation", natural=True)
k = pd.read_csv(HERE / "out" / "analisis_mini35.csv")
k51 = pd.read_csv(HERE / "out" / "analysis_mini51.csv")
fig, axs = plt.subplots(2, 2, figsize=(7, 5.4))
for ax, (d, nm) in zip(axs.flat[:2], (("nyata", "(A) 11 new datasets"), ("Q", "(B) Synthetic tasks"))):
    for rule, lbl, c in (("naif", "Naive (point, same POOL)", "#999999"), ("POOL-boot 0,005", "Bootstrap + margin, same POOL", "#e69138"), ("BRIDGE", "BRIDGE (cross-fitted)", "#1f4e79")):
        s = k[(k.data == d) & (k.aturan == rule)].sort_values("nK")
        lo6, hi6 = wilson(s.salah, s.terima)
        ax.errorbar(s.nK, s.pct_salah, yerr=[s.pct_salah - lo6, hi6 - s.pct_salah], marker="o", label=lbl, color=c, lw=1.6 if rule == "BRIDGE" else 1, capsize=2.5, elinewidth=.7)
    ax.axhline(10, color="r", lw=.6, ls="--"); ax.set_xscale("log"); ax.set_title(nm, fontsize=8.5); ax.set_xlabel("Candidate-space size $|\\mathcal{C}|$ (log scale)"); ax.set_ylabel("False improvements (%)")
for ax, (setting, nm) in zip(axs.flat[2:], (("C", "(C) 12 clinical tasks"), ("D", "(D) 34 supporting binary datasets"))):
    for rule, lbl, c in (("naive", "Naive (point, same POOL)", "#999999"), ("same_pool_boot_005", "Bootstrap + margin, same POOL", "#e69138"), ("BRIDGE", "BRIDGE (cross-fitted)", "#1f4e79")):
        s = k51[(k51.setting == setting) & (k51.rule == rule)].sort_values("nK")
        ax.errorbar(s.nK, s.harm_pct, yerr=[s.harm_pct - s.harm_ci_lo, s.harm_ci_hi - s.harm_pct], marker="o", label=lbl, color=c, lw=1.6 if rule == "BRIDGE" else 1, capsize=2.5, elinewidth=.7)
    ax.axhline(10, color="r", lw=.6, ls="--"); ax.set_xscale("log"); ax.set_title(nm, fontsize=8.5); ax.set_xlabel("Candidate-space size $|\\mathcal{C}|$ (log scale)"); ax.set_ylabel("False improvements (%)")
axs.flat[0].legend(fontsize=6.5, frameon=False)
save(fig, "fig6_candidate_space_size")
import os  # noqa: E402
os.environ["BRIDGE_ASSETS_OUT"] = str(OUT)
exec(open(HERE / "make_fig6_row.py").read(), {"__file__": str(HERE / "make_fig6_row.py")})  # full-width one-row layout used in the paper
k2 = k.assign(Rule=k.aturan.map({"naif": "Naive", "POOL-boot 0,005": "Bootstrap + margin, same POOL", "BRIDGE": "BRIDGE"}), Data=k.data.map({"nyata": "(A)", "Q": "(B)"}))
t = k2[["Data", "nK", "Rule", "terima", "salah", "pct_salah", "batas", "LOCO", "yield_", "total_delta", "pct_salah_yield_BRIDGE", "lulus"]].copy()
t.columns = ["Data", "|C|", "Rule", "Accepted", "Harmful", "Harmful %", "Risk bound %", "LOCO %", "Yield/100", "Total gain", "Harmful % at BRIDGE yield", "Meets criterion"]
t51 = k51.assign(Data=k51.setting.map({"C": "(C)", "D": "(D)"}), Rule=k51.rule.map({"naive": "Naive", "same_pool_boot_005": "Bootstrap + margin, same POOL", "BRIDGE": "BRIDGE"}))
t51 = t51[["Data", "nK", "Rule", "accepted", "harmful", "harm_pct", "kl", "loco", "yield_", "total_gain", "harm_at_bridge_yield", "meets"]]
t51.columns = t.columns
t = pd.concat([t, t51], ignore_index=True)
tex(t.round(2), "tab10_candidate_space",
    "Effect of candidate-space size $|\\mathcal{C}|$ in settings (A)--(D) (same HGB, DT and SVC members; the default HGB-based candidate space is refined) on the naive rule, the same-POOL cell (bootstrap lower bound and margin without cross-fitting), and BRIDGE. In this table, setting (D) covers the 34 binary supporting datasets; the multiclass datasets are not part of this analysis. "
    "Grids (experiments/mini35_ruang_kandidat.py): 1,825 = 46 HGB thresholds (0.05--0.95, step 0.02) + 414 HGB--DT pairs (weights 0.1--0.9, step 0.1, $\\times$ 46 thresholds) + 1,365 three-model simplex points (weight step 0.1, 65 weight vectors without pure HGB, $\\times$ 21 thresholds 0.30--0.70, step 0.02); "
    "11,249 = 90 HGB thresholds (step 0.01, without 0.50) + 1,729 pairs (weights 0.05--0.95, step 0.05, $\\times$ 91 thresholds) + 9,430 simplex points (weight step 0.05, 230 weight vectors, $\\times$ 41 thresholds, step 0.01).",
    "tab:kspace", rc=[yn_color(v) for v in t["Meets criterion"]],
    hdroverrides={"K": r"$|\mathcal{C}|$"}, pw={"Rule": 98}, crit_note=MIXCRIT_NOTE)

# Base-learner dependence (MINI-33/33b/34)
b = pd.concat([pd.read_csv(HERE / "out" / "analisis_mini34.csv"), pd.read_csv(HERE / "out" / "analisis_mini36.csv")], ignore_index=True)
NAMES = {"HGB": "HGB (scikit-learn default, early stopping)", "XGBeq": "XGBoost, regularized (4 settings changed)", "X0": "XGBoost, project default (300 rounds)",
         "X_ES": "+ early stopping only", "X_MCW5": "+ min_child_weight 5 only", "X_LEAF31": "+ 31-leaf trees only", "X_L2_0": "+ reg_lambda 0 only",
         "X_N30": "30 rounds, no early stopping", "X_N100": "100 rounds, no early stopping",
         "MLP_ES": "MLP (64,32), early stopping", "MLP_noES": "MLP (64,32), 300 epochs, no early stopping"}
t = b.assign(cfg=b.kondisi.map(NAMES))[["cfg", "terima", "salah", "batas", "konservatif", "yield_", "total_delta", "naif_pct_salah", "med_skill_pool", "lulus"]]
t.columns = ["Base learner", "Accepted", "Harmful", "Proposition 1 bound %", "Conservative %", "Yield/100", "Total gain", "Naive harmful %", "Median POOL log-loss skill", "Meets criterion"]
tfull = t.copy()
CAPS["tabS14_base_learner_full"] = "Base learners on the synthetic tasks (B), every tested setting with the bound, the conservative bound and the log-loss skill."
tex(tfull.round(3), "tabS14_base_learner_full",
    "Base-learner dependence on 1,400 fresh synthetic tasks: one setting of the project-default XGBoost changed at a time, plus a multilayer perceptron with and without early stopping (pre-registered prediction test). POOL log-loss skill = 1 - log-loss / log-loss of a prevalence-only forecast, computable without reference data.",
    "tab:basefull", rc=[yn_color(v) for v in t["Meets criterion"]], pw={"Base learner": 140}, crit_note=QCRIT_NOTE)
_keep = ["HGB (scikit-learn default, early stopping)", "XGBoost, project default (300 rounds)", "+ early stopping only", "30 rounds, no early stopping",
         "100 rounds, no early stopping", "XGBoost, regularized (4 settings changed)", "MLP (64,32), early stopping", "MLP (64,32), 300 epochs, no early stopping"]
_short = {"HGB (scikit-learn default, early stopping)": "HGB, early stopping (paper)", "XGBoost, project default (300 rounds)": "XGBoost, 300 rounds",
          "+ early stopping only": "XGBoost, early stopping", "30 rounds, no early stopping": "XGBoost, 30 rounds", "100 rounds, no early stopping": "XGBoost, 100 rounds",
          "XGBoost, regularized (4 settings changed)": "XGBoost, regularized", "MLP (64,32), early stopping": "MLP, early stopping",
          "MLP (64,32), 300 epochs, no early stopping": "MLP, 300 epochs"}
tc = t.set_index("Base learner").loc[_keep].reset_index()
tc = pd.DataFrame({"Base learner": tc["Base learner"].map(_short), "Naive harmful %": tc["Naive harmful %"].map(lambda v: f"{v:.1f}"),
                   "BRIDGE: accepted / harmful (%)": [_pct(a_, h_) for a_, h_ in zip(tc.Accepted, tc.Harmful)],
                   "Proposition 1 bound %": tc["Proposition 1 bound %"].map(lambda v: f"{v:.2f}"), "Yield/100": tc["Yield/100"].map(lambda v: f"{v:.1f}"),
                   "Meets criterion": tc["Meets criterion"]})
tex(tc, "tab11_base_learner",
    "1,400 fresh synthetic tasks (B); one base learner replaces HGB everywhere, the gate and candidate space unchanged. Every tested setting, the conservative "
    "bound and the log-loss skill: Table~\\ref{tab:basefull}.",
    "tab:base", rc=[yn_color(v) for v in tc["Meets criterion"]], natural=True, crit_note=QCRIT_NOTE)
au = pd.concat([pd.read_csv(HERE / "out" / f"mini33b_m{i}.csv") for i in (1, 2)])
t = au.groupby("model").agg(AUC=("auc", "median"), LL=("logloss", "median"), EX=("extreme_pct", "mean"), WF=("wf1", "median"), R=("rounds", "median"), S=("fit_s_per_bag", "median")).loc[["HGB", "XGB", "XGBeq"]].reset_index()
t["model"] = t.model.map({"HGB": "HGB", "XGB": "XGBoost, project default", "XGBeq": "XGBoost, regularized"})
t.columns = ["Base learner", "AUC", "Log-loss", "Probabilities <0.01 or >0.99 %", "Weighted F1 at 0.5", "Boosting rounds used", "Fit time per bag (s)"]
tex(t.round(4), "tabS2_base_learner_audit", "Audit of the base learners on the locked reference data (median over 1,400 tasks; mean for the share of probabilities below 0.01 or above 0.99; descriptive).", "tab:audit", crit_note="")
ida = pd.read_csv(HERE / "out" / "audit_identitas.csv").astype(object)
ida = ida.map(lambda v: "" if v != v else (int(v) if isinstance(v, float) else v))
ida.columns = ["Source", "Identity key", "Rows", "Unique keys", "Duplicates", "Note"]
# one statement per line: each ';'-separated note becomes its own row (source and counts only on the first);
# long "not available" keys are abbreviated here and explained in the footnote
ida["Identity key"] = ida["Identity key"].replace({"(not available in the UCI copy)": "none in UCI copy",
                                                   "(not available; SEQNO is state-specific)": "SEQNO (state-level)",
                                                   "(not available)": "none"})
_rows = []
for _, r in ida.iterrows():
    for k, part in enumerate(str(r["Note"]).split("; ")):
        _rows.append(list(r[:5]) + [part] if k == 0 else ["", "", "", "", "", part])
ida = pd.DataFrame(_rows, columns=ida.columns).astype(object)
tex(ida, "tabS3_identity_audit", "Patient-identity audit of the clinical sources (de-duplication and filtering applied in the loaders before any split). Participant identifier unavailable in the UCI copy of ACTG 175 and patient identifier unavailable in the OpenML mammography data; the BRFSS record number SEQNO is unique only within a state.", "tab:identity", natural=True)
print("tab9, tab10, tab11, tabS2, tabS3, fig6 written")

# Table 8: complexity, rebuilt from out/complexity_timing.csv (measured by complexity_timing.py)
ct = HERE / "out" / "complexity_timing.csv"
if ct.exists():
    d = pd.read_csv(ct)
    g = d.groupby("dataset", sort=False).agg(n_pool=("n_pool", "first"), features=("n_features", "first"), fit=("fit_s", "mean"), naive=("naive_select_s", "mean"),
                                             cf=("crossfit_select_s", "mean"), boot=("bootstrap_s", "mean"), gate=("bridge_gate_s", "mean"),
                                             inf_b=("infer_baseline_ms_per_row", "mean"), inf_e=("infer_ensemble_ms_per_row", "mean")).reset_index()
    g["gate % of fit"] = 100 * g.gate / g.fit
    g.columns = ["Dataset", "POOL rows", "Features", "Fit bags on SEARCH (s)", "Naive selection (s)", "Cross-fitted selection (s)", "Bootstrap (s)", "BRIDGE gate total (s)",
                 "Inference, baseline (ms/row)", "Inference, 3-model ensemble (ms/row)", "Gate / fit (%)"]
    tex(g.round(4), "tab8_complexity", "Wall-clock cost (mean of 3 trials, one CPU core, Intel Core i9-13900K): fitting the bagged base models on SEARCH (32 bags each of HGB, DT and SVC), naive selection, the BRIDGE gate (cross-fitted selection and bootstrap) and inference latency of the deployed predictor (batch of 2,000 rows; baseline = 32 HGB bags, ensemble = 96 bagged models). Inference cost equals that of the deployed predictor, independent of BRIDGE. Gate total: cross-fitted selection plus bootstrap. The percentage is computed from unrounded times.", "tab:complexity", crit_note="")

# Supplement: per-dataset results on the 11 new datasets (MINI-24; accepted / harmful per rule)
pdm = RD("mini24_m1.csv", "mini24_m2.csv", "mini24_m3.csv")
PD_RULES = [("naif", "Naive"), ("t5x2", "5x2cv"), ("crt_final", "Corrected t"), ("crt", "Corrected t + re-selection"), ("boot0", "BRIDGE w/o margin"), ("bridge", "BRIDGE")]
for c, _ in PD_RULES: pdm[c] = pd.to_numeric(pdm[c], errors="coerce").fillna(0).astype(int)
pdrows = []
for ds, g in pdm.groupby("dataset", sort=True):
    row = {"Dataset": ds, "Trials": len(g)}
    for c, lab in PD_RULES:
        a = g[g[c] == 1]; row[lab] = f"{len(a)}/{int((a.ref <= 0).sum())}"
    pdrows.append(row)
tot = {"Dataset": "Total", "Trials": len(pdm)}
for c, lab in PD_RULES:
    a = pdm[pdm[c] == 1]; tot[lab] = f"{len(a)}/{int((a.ref <= 0).sum())}"
pdrows.append(tot)
CAPS["tabS4_per_dataset"] = "Per-dataset results on the 11 new datasets."
tex(pd.DataFrame(pdrows), "tabS4_per_dataset",
    "Per-dataset results on the 11 new datasets (100 trials each). Cells: accepted / harmful updates per rule; harmful = gain $\\le 0$ on the locked reference (Definition~\\ref{def:risk}); 0/0 = no acceptance (harmful proportion undefined). Totals reproduce Table~\\ref{tab:rules}.",
    "tab:perdataset")
print("tabS4 written")

# Appendix table: clinical results by source and with each source removed (analysis_clinical_by_source.py)
cs = pd.read_csv(HERE / "out" / "analysis_clinical_by_source.csv")
SRCN = {"actg175": "ACTG 175", "brfss": "BRFSS (4 tasks)", "covid_mx": "COVID-19 Mexico", "mammography": "Mammography",
        "nhanes": "NHANES (4 tasks)", "wids": "WiDS ICU"}
csrows = []
for met, mlab in (("Net benefit p_t=0.2", "Net benefit, $p_t=0.2$ (primary)"), ("Balanced accuracy", "Balanced accuracy (co-primary)")):
    c_ = cs[cs.metric == met]
    keys = [(f"source={k}", v) for k, v in SRCN.items()] + [("all sources", "All sources")] + [(f"without {k}", f"Without {v.split(' (')[0]}") for k, v in SRCN.items()]
    for key, lbl in keys:
        b_ = c_[(c_.rule == "BRIDGE") & (c_.subset == key)].iloc[0]; n_ = c_[(c_.rule == "Naive") & (c_.subset == key)].iloc[0]
        single = key.startswith("source=")  # one source: counts only (a bound on a handful of acceptances is uninformative)
        csrows.append([mlab, lbl, int(b_.trials), int(b_.accepted), int(b_.harmful), int(b_.reference_uncertain),
                       np.nan if single else round(b_.bound, 2), np.nan if single else round(b_.conservative_bound, 2),
                       np.nan if single else round(b_.loco, 2), int(n_.accepted), int(n_.harmful)])
CAPS["tabS5_clinical_by_source"] = ("Clinical results by source and with each source removed: on the primary metric BRIDGE has zero harmful acceptances "
                                    "in every source and subset; on balanced accuracy all 8 harmful acceptances come from NHANES.")
tex(pd.DataFrame(csrows, columns=["Metric", "Source or subset", "Trials", "BRIDGE accepted", "BRIDGE harmful", "BRIDGE reference-uncertain", "BRIDGE bound %",
                                  "BRIDGE conservative %", "BRIDGE LOCO %", "Naive accepted", "Naive harmful"]),
    "tabS5_clinical_by_source",
    "Confirmatory clinical run, 200 trials per task. Harmful: gain $\\le 0$ on the locked reference; reference-uncertain: the Bayesian-bootstrap interval of the reference gain contains $0$. "
    "Bound: KL-ratio bound; conservative: every accepted update that is not clearly improving counted as harmful; LOCO: leave-one-source-out. "
    "Single-source rows give counts only.",
    "tab:clinsource", groupcol="Metric")
_t5 = OUT / "tabS5_clinical_by_source.tex"; _s5 = _t5.read_text().replace(r"$p\_t=0.2$", r"$p_t=0.2$").replace(r"\textbackslash{}", "\\").replace(r"\$", "$")
_seen = set(); _out = []
for _ln in _s5.split("\n"):  # print each metric label only on the first row of its group
    for _ml in ("Net benefit, $p_t=0.2$ (primary)", "Balanced accuracy (co-primary)"):
        if _ln.startswith(_ml + " &"):
            if _ml in _seen:
                _ln = _ln.replace(_ml, "", 1)
            _seen.add(_ml)
    _out.append(_ln)
_t5.write_text("\n".join(_out))
print("tabS5 written")

# Appendix table: margin sensitivity from stored lower bounds (analysis_margin_sweep.py)
ms = pd.read_csv(HERE / "out" / "analysis_margin_sweep.csv")
msrows = []
for st, g in ms.groupby("setting", sort=False):
    row = [st.replace("p_t=0.2", "$p_t=0.2$")]
    for _, r in g.sort_values("eps").iterrows():
        lo = "--" if pd.isna(r.loco) else f"{r.loco:.2f}"
        row.append(f"{int(r.accepted)}/{int(r.harmful)}: {r.bound:.2f} / {lo} {'$\\checkmark$' if r.meets else '$\\times$'}")
    msrows.append(row)
CAPS["tabS6_margin_sweep"] = ("Margin sensitivity at the four tested margins (0.0025, 0.005, 0.0075, 0.01): the criterion is met at all four in each "
                              "setting where it is met at the default 0.005; feature selection with SVC fails at all four (LOCO above 10%).")
tex(pd.DataFrame(msrows, columns=["Setting", "eps = 0.0025", "eps = 0.005 (default)", "eps = 0.0075", "eps = 0.01"]),
    "tabS6_margin_sweep",
    "Cells: accepted/harmful: risk bound % / LOCO % and the pre-registered criterion of the setting, recomputed from the stored 5% bootstrap lower bound of every trial "
    "(the deployed configuration does not depend on the margin). Risk bound: KL-ratio bound (real data) or Proposition 1 (population Q); LOCO is not defined on Q. Trials, yield, the R0 power check and the number of units with yield of at least 5 per 100 are in out/analysis_margin_sweep.csv (script analysis_margin_sweep.py).",
    "tab:marginsweep", natural=True)
_t6 = OUT / "tabS6_margin_sweep.tex"; _t6.write_text(_t6.read_text().replace(r"\textbackslash{}", "\\").replace(r"\$", "$").replace("eps =", r"$\varepsilon=$").replace(r"$p\_t=0.2$", r"$p_t=0.2$"))
print("tabS6 written")

# Appendix table: paired cluster-bootstrap differences with raw and Holm p (analysis_paired_ci.py)
pc = pd.read_csv(HERE / "out" / "paired_ci.csv")
PRES = 2 / 10000  # resolution of the two-sided bootstrap p with B = 10,000


def _p(v, holm=False):
    if pd.isna(v):
        return "--"
    if v == 0:
        return f"<{6 * PRES:.4f} DAGGER" if holm else f"<{PRES:.4f} DAGGER"
    return f"{v:.4f}" if v < .01 else f"{v:.3f}" if v < .1 else f"{v:.2f}"


def _d(r):
    return f"{r.diff_pp:+.1f} [{r.ci95_lo:+.1f}; {r.ci95_hi:+.1f}]".replace("+0.0]", "0.0]")


CMP = [("naive", "Naive"), ("McNemar on same POOL", "McNemar on same POOL"), ("5x2cv paired t", "5x2cv paired t"),
       ("corrected resampled t", "Corrected resampled t"), ("BRIDGE without margin", "BRIDGE without margin"),
       ("corrected t + per-split reselection", "Corrected t + per-split re-selection")]
ops = pc[pc.analysis == "operating point"]
cirows = []
for key, lbl in CMP:
    a = ops[(ops.data == "real") & (ops.comparator == key)].iloc[0]; b = ops[(ops.data == "Q") & (ops.comparator == key)].iloc[0]
    cirows.append([lbl, _d(a), _p(a.p_boot), _p(a.p_holm, True), _d(b), _p(b.p_boot), _p(b.p_holm, True)])
_sp = pc[pc.comparator == "same-POOL bootstrap + margin"]
for an, lbl in (("ablation cell, operating point", "Same-POOL cell, own operating point"), ("ablation cell, matched yield", "Same-POOL cell, matched yield")):
    a = _sp[(_sp.analysis == an) & (_sp.data == "real")].iloc[0]; b = _sp[(_sp.analysis == an) & (_sp.data == "Q")].iloc[0]
    cirows.append([lbl, _d(a), _p(a.p_boot), "--", _d(b), _p(b.p_boot), "--"])
fm = pc[pc.analysis.str.startswith("fixed margin") & pc.analysis.str.endswith("harmful pp")]
_fm = lambda t, data: fm[(fm.data == data) & (fm.analysis.str.contains(f"t={t},"))]
for t, real, lbl in ((0.025, "11 new datasets", "Fixed margin t=0.025 (post hoc), 11 new datasets"),
                     (0.025, "1000 training rows", "Fixed margin t=0.025 (post hoc), 1,000 training rows"),
                     (0.02, None, "Fixed margin t=0.02 (post hoc)")):
    rr = _fm(t, real) if real else None; rq = _fm(t, "Q") if t == 0.02 or real == "11 new datasets" else None
    cells = lambda r: [_d(r.iloc[0]), _p(r.iloc[0].p_boot), "--"] if r is not None and len(r) else ["--"] * 3
    cirows.append([lbl] + cells(rr) + cells(rq))
CAPS["tabS7_paired_ci"] = "Difference in harmful proportion, BRIDGE minus comparator (percentage points; negative = BRIDGE safer)."
tex(pd.DataFrame(cirows, columns=["Comparator", "Real: difference (95% CI)", "Real: p", "Real: Holm p",
                                  "Q: difference (95% CI)", "Q: p", "Q: Holm p"]),
    "tabS7_paired_ci",
    "Paired cluster bootstrap at each rule's own operating point: in every resample both rules are evaluated on the same resampled decisions; "
    "clusters are the 11 datasets (real data) or the 1,400 tasks (Q); B = 10,000, seed 20261012; percentile 95% intervals. "
    "Two-sided p = 2 x min(share of resampled differences >= 0, share <= 0), resolution 2/B = 0.0002. DAGGER No resampled difference fell on the opposite side of zero (0 of 10,000): the empirical p is below its resolution, which is not an upper confidence bound on the true tail probability (likewise for the Holm-adjusted value); "
    "resamples in which a rule accepts nothing are left out for that rule. Holm adjustment over the six comparators; "
    "each test is of H0: the two harmful proportions are equal, against H1: the two proportions differ. The same-POOL cell rows (at each rule's own operating point, and at matched yield, i.e., the same number of top-ranked accepted updates) and the fixed-margin rows are post-hoc, exploratory "
    "comparisons with nominal intervals, outside the Holm family. Script: analysis_paired_ci.py.",
    "tab:ci", natural=True)
_t7 = OUT / "tabS7_paired_ci.tex"
_s7 = re.sub(r"([+-]\d+\.\d) \[([+-]?\d+\.\d); ([+-]?\d+\.\d)\]", r"$\1\\ [\2;\\,\3]$", _t7.read_text())
_t7.write_text(_s7.replace(" x min", r" $\times$ min").replace("Q:", r"$\mathcal{Q}$:").replace("(Q)", r"($\mathcal{Q}$)")
               .replace("t=0.025", "$t=0.025$").replace("t=0.02 ", "$t=0.02$ ").replace("<0.0", "$<$0.0").replace("$<$0.0002)", "<0.0002)")
               .replace(">= 0", r"$\geq 0$").replace("<= 0", r"$\leq 0$").replace(" DAGGER", r"$^{\dagger}$").replace("DAGGER ", r"$^{\dagger}$\,"))
print("tabS7 written")

# Appendix table: random-selection control and gain-loss trade-off (analysis_selectivity.py; post hoc)
sl = pd.read_csv(HERE / "out" / "analysis_selectivity.csv")


def _pv(v):
    return "--" if pd.isna(v) else (f"<10^{{{int(np.floor(np.log10(v))) + 1}}}" if v < 1e-4 else f"{v:.2f}")


def _ci(r, c, nd=0):
    return f"{r[c]:.{nd}f} [{r[c + '_lo']:.{nd}f}; {r[c + '_hi']:.{nd}f}]"


slrows = []
for _, r in sl.iterrows():
    slrows.append([{"11 new datasets": "(A)", "Synthetic population Q": "(B)", "Clinical, net benefit": "(C)", "39 supporting datasets": "(D)"}[r.setting], f"{int(r.decisions):,}",
                   f"{int(r.naive_accepted):,} ({100 * r.naive_harmful / r.naive_accepted:.1f})",
                   f"{int(r.bridge_accepted):,} ({100 * r.bridge_harmful / r.bridge_accepted:.1f})",
                   f"{r.random_expected_harmful:.1f} vs {int(r.bridge_harmful)} (p {_pv(r.random_p)})",
                   "--" if pd.isna(r.random_within_unit_expected) else f"{r.random_within_unit_expected:.1f} vs {int(r.bridge_harmful)} (p {_pv(r.random_within_unit_p)})",
                   _ci(r, "gain_kept"), _ci(r, "harmful_removed"), _ci(r, "loss_removed"),
                   f"{r.gain_per_acc_bridge:.4f} / {r.gain_per_acc_naive:.4f}"])
CAPS["tabS8_selectivity"] = ("Is BRIDGE's low harmful proportion only a consequence of accepting fewer updates? Random-selection control and "
                             "gain-loss trade-off against the naive rule (post hoc, exploratory).")
tex(pd.DataFrame(slrows, columns=["Setting", "Decisions", "Naive accepted (% harmful)", "BRIDGE accepted (% harmful)",
                                  "Random draw of BRIDGE's count: harmful expected vs BRIDGE", "Same, within each dataset or task",
                                  "Naive net gain kept by BRIDGE %", "Naive harmful updates removed %", "Naive loss removed %",
                                  "Gain per accepted update, BRIDGE / naive"]),
    "tabS8_selectivity",
    "Post hoc, exploratory reanalysis of the stored runs: (A) 11 new datasets; (B) synthetic population Q; (C) clinical tasks under net benefit at p_t = 0.2; (D) 39 supporting datasets. Every setting: HGB with the default candidate space. "
    "In every setting the updates BRIDGE accepts are a subset of those the naive rule accepts. Random draw: BRIDGE's number of accepted updates drawn "
    "at random from the naive rule's accepted updates (exact hypergeometric p of a harmful count at most BRIDGE's, drawing without replacement and conditional on the observed accepted updates; it compares BRIDGE with a random choice of the same size, not a probability about future tasks); within each dataset or task: the "
    "same with BRIDGE's own count in each dataset (clinical: task), so that only the choice among trials of the same dataset is tested; on Q every "
    "task has one trial. Net gain kept: BRIDGE's summed reference gain over the naive rule's; loss: summed negative reference gain. "
    "Brackets: 95% cluster-bootstrap percentile intervals over datasets (real), sources (clinical) or tasks (Q), B = 10,000. Script: analysis_selectivity.py.",
    "tab:selectivity", natural=True)
_t8 = OUT / "tabS8_selectivity.tex"
_s8 = re.sub(r"<10\^\{(-?\d+)\}", r"$<10^{\1}$", _t8.read_text())
_t8.write_text(_s8.replace("QQ", r"$\mathcal{Q}$").replace(" Q;", r" $\mathcal{Q}$;").replace("(Q)", r"($\mathcal{Q}$)").replace("p\\_t = 0.2", r"$p_t=0.2$"))
print("tabS8 written")

# Appendix table: development runs cited in the text (analysis_development_runs.py)
_dv = pd.read_csv(HERE / "out" / "analysis_development_runs.csv")
_dvq = {"MINI-02": "34 (D); 100 trials", "MINI-04": "34 (D); 100 trials", "MINI-09": "18 dev.; 100 trials", "MINI-12": "18 dev.; 100 trials",
        "MINI-14": "27 (D); 40 trials per size", "MINI-16": "18 dev.; 100 trials", "MINI-19": "8 wide (dev.)"}
dvrows = []; _dvn = []; _lt = iter("abcdefghij")
for _, r in _dv.iterrows():
    if pd.isna(r.trials):
        _dvn.append(f"MINI-12 mean reference weighted-F1: {r.mean_default:.4f} (default HGB) vs {r.mean_tuned:.4f} (tuned), mean gain {r.mean_tuned - r.mean_default:+.4f}"); continue
    if False:
        res = f"{r.mean_default:.4f} (default) vs {r.mean_tuned:.4f} (tuned); mean gain {r.mean_tuned - r.mean_default:+.4f}"; acc = bd = lc = yl = "--"
    else:
        acc = f"{int(r.accepted):,} / {int(r.harmful)}"; bd = f"{r.bound_pct:.2f}"; lc = f"{r.loco_pct:.2f}"; yl = f"{r.yield_per100:.1f}"
        res = (f"mean gain {r.mean_gain:+.4f}; clearly worse {r.clearly_worse_pct:.1f}%"
               if not pd.isna(r.mean_gain) else (f"power check R0 {r.r0_pct:.2f}%" if r.run in ("MINI-02", "MINI-19") else "--"))
    _item = r["item"]
    if res != "--":
        _l = next(_lt); _item = _item + f"$^{{{_l}}}$"; _dvn.append(f"{_l}: {res}")
    dvrows.append([r.run, Raw(_item), _dvq[r.run] if r.run in _dvq else "", f"{int(r.trials):,}", acc, bd, lc, yl])
CAPS["tabS9_development_runs"] = "Development runs cited in the text: the numbers quoted from the exploratory stage, recomputed from the stored outputs."
tex(pd.DataFrame(dvrows, columns=["Run", "Comparison", "Data", "Decisions", "Accepted / harmful", "Bound %", "LOCO %", "Yield/100"]),
    "tabS9_development_runs",
    "Development-stage runs (before the confirmatory runs; none uses the 11 new datasets). Data: datasets; trials per dataset (dev. = the 18 development datasets; "
    "(D) = the 34 confirmation datasets of the supporting group). MINI-12 and MINI-14 tune HGB with Optuna (TPE, 25 search iterations) on the 630-row SEARCH split; MINI-14 "
    "repeats this at 630, 2,000 and 5,000 tuning rows. Superscripts: further quantities of the row (mean reference weighted-F1 gain of the tuned learner and the share of trials whose "
    "95% reference interval lies below zero; power check). " + "; ".join(_dvn) + ". Accepted: lower bootstrap quantile above the margin 0.005; "
    "harmful: reference gain at most 0. Bound: KL-ratio bound (b = 1); LOCO: leave-one-dataset-out. Clearly worse: the 95% reference interval lies "
    "below zero. Power check R0: the LOCO bound recomputed with zero harmful updates; with 50 trials per dataset it exceeds 10% even when no "
    "decision is harmful (MINI-19), and 100 or more trials per dataset bring it below 10% (MINI-02; MINI-19 extended by MINI-19b). "
    "Script: analysis_development_runs.py.",
    "tab:devruns", pw={"Comparison": 150})
print("tabS9 written")

# Appendix table: confirmatory and sensitivity runs cited in the text (analysis_cited_runs.py)
_cr2 = pd.read_csv(HERE / "out" / "analysis_cited_runs.csv")
crows = []
for _, r in _cr2.iterrows():
    it = r["item"]
    if r.run == "MINI-28":
        data = "12 clinical tasks, 6 sources; 100 trials per task"
        res = f"{int(r.accepted):,} / {int(r.harmful)} ({r.harmful_pct:.1f}%); bound {r.bound_pct:.2f}%; LOCO-source {r.loco_pct:.2f}%"
    elif r.run == "MINI-31":
        data = "Q, 1,400 tasks"
        res = f"{int(r.accepted)} / {int(r.harmful)} ({r.harmful_pct:.1f}%); Proposition 1 bound {r.bound_pct:.2f}%; yield {r.yield_per100:.1f}"
    elif "Spearman" in it:
        data = f"Q, 1,400 tasks; {int(r.configurations)} configurations (Table base)"
        res = f"rho = {r.rho:.2f}, p = {r.p:.3f}"
        it = it.replace(" (MINI-34, MINI-36)", "").replace(" (MINI-34)", "")
    elif r.run == "MINI-33b":
        data = "Q, 1,400 tasks; locked reference"
        res = f"{r.share_pct:.1f}% of tasks"
    elif "skipped" in it:
        data = f"training-size study; {int(r.trials):,} trials"
        res = r.skipped
    elif "skill" in it:
        data = "Q, 1,400 tasks; 9 configurations"
        res = f"{int(r.accepted):,} / {int(r.harmful)} ({r.harmful_pct:.2f}%)"
    elif "bags" in it:
        data = "Q, 1,400 tasks"
        res = f"{int(r.accepted)} / {int(r.harmful)}; bound {r.bound_pct:.2f}%; yield {r.yield_per100:.1f}"
    elif "Monte Carlo" in it:
        data = "Q, 1,400 tasks" if it.endswith("Q") else "11 new datasets, 10 trials each"
        res = (f"median SD {r.median_sd:.5f}; one alternative seed changes the decision in {r.flip_per_rerun_pct:.1f}% of re-runs; "
               f"at least one of 20 seeds changes it in {int(r.trials_any_flip)} of {int(r.trials):,} trials ({r.trials_any_flip_pct:.1f}%)")
    elif "reference SE" in it:
        data = f"{int(r.datasets)} of the 11 new datasets"
        res = f"median SE {r.se_min:.4f}-{r.se_max:.4f} (references {int(r.n_ref_min):,}-{int(r.n_ref_max):,} rows)"
    else:
        data = "11 new datasets; 100 trials each"
        res = f"{int(r.accepted):,} / {int(r.harmful)} ({r.harmful_pct:.1f}%); bound {r.bound_pct:.2f}%; LOCO {r.loco_pct:.2f}%; yield {r.yield_per100:.1f}"
        it = it.replace("asli ", "original ").replace(": naif", ": naive")
    crows.append([r.run, it.replace(" (Q)", "").replace(" (nine configurations)", ""), data, res])
crows.sort(key=lambda x: (int(re.sub(r"\D", "", x[0][5:7])), x[0]))  # by run number; stable within a run
CAPS["tabS12_cited_runs"] = "Confirmatory and sensitivity runs cited in the text whose numbers appear in no other table, recomputed from the stored outputs."
tex(pd.DataFrame(crows, columns=["Run", "Check", "Data", "Result"]),
    "tabS12_cited_runs",
    "Cells: accepted / harmful updates (harmful share), KL-ratio bound (b = 1), leave-one-out bound (LOCO; by dataset, or by source for the clinical "
    "tasks) and yield per 100 decisions, as applicable. MINI-28: the clinical stage with weighted-F1 as the primary endpoint, before the net-benefit "
    "stage (MINI-30). MINI-31: XGBoost with the project defaults (300 rounds, no early stopping) during development. MINI-33b: descriptive audit of the base learners on the locked reference. MINI-17/MINI-20: trials skipped because a bag had fewer than 10 examples of a class. MINI-34: accepted updates pooled over the nine pre-registered HGB and XGBoost configurations, split by the sign of the base "
    "learner's log-loss skill on POOL. MINI-37: bag count and SEARCH size varied one at a time; Monte Carlo error: standard deviation of the 5% "
    "bootstrap quantile over 20 bootstrap seeds per trial; reference SE: per-dataset median bootstrap standard error of the reference gain. MINI-38: "
    "the outer split repeated with two fresh reference seeds. Script: analysis_cited_runs.py.",
    "tab:citedruns", pw={"Check": 130, "Result": 170})
_t12 = OUT / "tabS12_cited_runs.tex"
_t12.write_text(_t12.read_text().replace("Q, 1,400", r"$\mathcal{Q}$, 1,400").replace("error, Q", r"error, $\mathcal{Q}$")
                .replace(">= 40,000", "at least 40,000").replace("< 40,000", "below 40,000"))
print("tabS12 written")

# Appendix tables: MINI-39 characteristics (analysis_mini39_characteristics.py; post hoc, exploratory)
_cq = pd.read_csv(HERE / "out" / "mini39_char_Q.csv")
cqrows = [[r.characteristic, r.level, f"{int(r.tasks):,}", f"{r.clear_pct:.0f}", f"{r.win_HGB_pct:.0f}", f"{r.win_DT_pct:.0f}",
           f"{r.win_SVC_pct:.0f}", f"{r.win_KNN_pct:.0f}", f"{int(r.naive_acc):,} / {int(r.naive_harm)} ({100 * r.naive_harm / r.naive_acc:.1f})",
           f"{int(r.bridge_acc)} / {int(r.bridge_harm)} ({100 * r.bridge_harm / max(r.bridge_acc, 1):.1f})"] for r in _cq.itertuples()]
CAPS["tabS10_mini39_char_Q"] = "Tuning updates on the synthetic population by task characteristic: which tuned learner wins, and how the naive rule and BRIDGE fare (post hoc, exploratory)."
tex(pd.DataFrame(cqrows, columns=["Characteristic", "Level", "Tasks", "Clear winner %", "HGB wins %", "DT wins %", "SVC wins %", "KNN wins %",
                                  "Naive: accepted / harmful (%)", "BRIDGE: accepted / harmful (%)"]),
    "tabS10_mini39_char_Q",
    "MINI-39, tasks T2900-T4299 of the synthetic population, whose generator parameters are known (mechanism; signal strength s in tertiles; "
    "minority parameter; number of features). Clear winner: the best tuned learner's mean reference weighted-F1 exceeds the second best by at least the "
    "margin 0.005; win shares are among tasks with a clear winner. Accepted and harmful counts are pooled over the four learners (one decision per "
    "learner and task). The groupings were chosen after the results were seen. Script: analysis_mini39_characteristics.py.",
    "tab:mini39charQ", natural=True)
_cr = pd.read_csv(HERE / "out" / "mini39_char_R.csv")
crrows = [[r.dataset.replace("_", " "), f"{int(r.rows):,}", f"{int(r.features)}", f"{r.minority_pct:.1f}", f"{r.default_HGB_wF1:.3f}",
           r.tuned_winner, r.BRIDGE_HGB, r.BRIDGE_DT, r.BRIDGE_SVC, r.BRIDGE_KNN, r.naive_all] for r in _cr.itertuples()]
CAPS["tabS11_mini39_char_R"] = "Tuning updates on the 11 new datasets by dataset characteristic (post hoc, exploratory)."
tex(pd.DataFrame(crrows, columns=["Dataset", "Rows", "Features", "Minority %", "Default HGB weighted-F1", "Tuned winner",
                                  "BRIDGE HGB", "BRIDGE DT", "BRIDGE SVC", "BRIDGE KNN", "Naive, all learners"]),
    "tabS11_mini39_char_R",
    "MINI-39, 100 trials per dataset and learner, sorted by the default HGB reference weighted-F1 (a difficulty proxy). Tuned winner: the learner "
    "whose tuned model has the highest mean reference weighted-F1, if it leads the second by at least 0.005; otherwise tie. Cells: accepted / "
    "harmful tuning updates (BRIDGE per learner; naive pooled over the four learners). With 11 datasets the table is descriptive. Script: "
    "analysis_mini39_characteristics.py.",
    "tab:mini39charR", natural=True)
print("tabS10/S11 written")

# Summary table: main results in the three scenarios (analysis_selectivity.py, paired_ci.csv)
from verify_exp41 import kl_ratio_up as _up
_sl = pd.read_csv(HERE / "out" / "analysis_selectivity.csv"); _pc = pd.read_csv(HERE / "out" / "paired_ci.csv")
_m24 = SP.rd("mini24_m1.csv", "mini24_m2.csv", "mini24_m3.csv")
_m30 = SP.rd("mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"); _m30 = _m30[~_m30.kelas.isin(["error", "skip"])]; _m30 = _m30[_m30.metric == "nb20"]
_m43 = pd.concat([pd.read_csv(HERE / "out" / f"mini43_m{i}.csv") for i in (1, 2, 3)]).rename(columns={"lower_bound": "lo95", "gain_ref": "ref"})


def _kl(m):
    a = m[m.lo95 > .005]; return 100 * _up(int((a.ref <= 0).sum()), len(a), len(m), 1)


_bounds = {"11 new datasets": ("KL-ratio", _kl(_m24)), "Synthetic population Q": ("Proposition 1", 100 * SP.prop1(8, 475)),
           "Clinical, net benefit": ("KL-ratio", _kl(_m30)), "39 supporting datasets": ("KL-ratio", _kl(_m43))}
_pcd = {"11 new datasets": _pc[(_pc.analysis == "operating point") & (_pc.data == "real") & (_pc.comparator == "naive")].iloc[0],
        "Synthetic population Q": _pc[(_pc.analysis == "operating point") & (_pc.data == "Q") & (_pc.comparator == "naive")].iloc[0]}
def _harm_ci(m, nv, br, unit, seed=20261024, B=10_000):
    """Pooled % harmful of the naive rule and BRIDGE with 95% cluster-bootstrap percentile intervals."""
    rng = np.random.default_rng(seed); h = (m.ref <= 0).to_numpy(); u = m[unit].to_numpy()
    idx = [np.flatnonzero(u == k) for k in pd.unique(u)]; out = {}
    for nm, a in (("naive", nv), ("bridge", br)):
        f = lambda s_: 100 * h[s_][a[s_]].mean() if a[s_].any() else np.nan
        bs = [f(np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])) for _ in range(B)]
        out[nm] = (f(np.arange(len(m))), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5))
    return out
_q25 = SP.rd("mini25_m1.csv", "mini25_m2.csv", "mini25_m3.csv"); _q25 = _q25.assign(task=np.arange(len(_q25)))
_m43n = pd.concat([pd.read_csv(HERE / "out" / f"mini43_m{i}.csv") for i in (1, 2, 3)]).reset_index(drop=True)
_m43n = _m43n.assign(ref=_m43n.gain_ref)
_HCI = {"11 new datasets": _harm_ci(_m24.reset_index(drop=True), (_m24.cfg != "None").to_numpy(), (_m24.lo95 > .005).to_numpy(), "dataset"),
        "Synthetic population Q": _harm_ci(_q25, (_q25.cfg != "None").to_numpy(), (_q25.lo95 > .005).to_numpy(), "task"),
        "Clinical, net benefit": _harm_ci(_m30.reset_index(drop=True), (_m30.cfg != "None").to_numpy(), (_m30.lo95 > .005).to_numpy(), "dataset"),
        "39 supporting datasets": _harm_ci(_m43n, (_m43n.naive_accept == 1).to_numpy(), (_m43n.accept == 1).to_numpy(), "dataset")}
_hc = lambda v: f"{v[0]:.1f}" if v[2] - v[1] < 1e-9 else f"{v[0]:.1f} [{v[1]:.1f}; {v[2]:.1f}]"
smrows = []
for _, r in _sl.iterrows():
    if r.setting in _pcd:
        q = _pcd[r.setting]; dif = f"{-q.diff_pp:.1f} [{-q.ci95_hi:.1f}; {-q.ci95_lo:.1f}]"
    else:
        dif = f"{-r.diff_pp:.1f} [{-r.diff_hi:.1f}; {-r.diff_lo:.1f}]"
    bn, bv = _bounds[r.setting]
    smrows.append([{"11 new datasets": "(A) 11 new datasets", "Synthetic population Q": "(B) Synthetic tasks", "Clinical, net benefit": "(C) Clinical tasks, net benefit",
                    "39 supporting datasets": "(D) 39 supporting datasets"}[r.setting], f"{int(r.decisions):,}",
                   f"{int(r.naive_accepted):,}", _hc(_HCI[r.setting]["naive"]), f"{int(r.bridge_accepted):,}", _hc(_HCI[r.setting]["bridge"]), dif,
                   f"{bv:.2f}", f"{r.gain_kept:.0f} [{r.gain_kept_lo:.0f}; {r.gain_kept_hi:.0f}]"])
_se = pd.concat([pd.read_csv(HERE / "out" / "settingE_laptop.csv"),
                 pd.read_csv(HERE / "out" / "settingE_part2_laptop.csv")], ignore_index=True)
_se = _se.assign(dataset=_se.data_id.astype(str), ref=_se.gain_ref)
_seh = _harm_ci(_se, _se.naive_accept.astype(bool).to_numpy(), _se.accept.astype(bool).to_numpy(), "dataset")
_ser = pd.read_csv(HERE / "out" / "analysis_settingE.csv").query("group == 'E_all'").iloc[0]
smrows.append(["(E) OpenML collection, 14 datasets", f"{int(_ser.decisions):,}", f"{int(_ser.naive_accepted):,}",
               _hc(_seh["naive"]), f"{int(_ser.bridge_accepted):,}", _hc(_seh["bridge"]),
               f"{-_ser.bridge_minus_naive_pp:.1f} [{-_ser.bmn_hi:.1f}; {-_ser.bmn_lo:.1f}]", f"{_ser.bridge_kl:.2f}",
               f"{_ser.gain_retained_pct:.0f} [{_ser.gain_retained_lo:.0f}; {_ser.gain_retained_hi:.0f}]"])
CAPS["tab0_summary"] = "Main results in detail, settings (A) to (D) and the OpenML collection (E): on the same decisions, BRIDGE lowers the harmful proportion with every interval above zero."
tex(pd.DataFrame(smrows, columns=["Setting", "Decisions", "Naive: accepted", "Naive % harmful [95% CI]", "BRIDGE: accepted", "BRIDGE % harmful [95% CI]",
                                  "Reduction in % harmful, pp [95% CI]", "BRIDGE 95% upper bound %",
                                  "Naive net gain kept by BRIDGE % [95% CI]"]),
    "tab0_summary",
    "Every setting: current model HGB with library settings; candidates from the default space built from HGB, DT and SVC ($173$ binary, $14+8K$ multiclass; Section~\\ref{sec:method-partition}); other learners and ensemble members: Table~\\ref{tab:scope}. % harmful: pooled over all decisions of a setting, one fresh seed per decision; 95% cluster-bootstrap percentile interval over datasets, tasks or clinical tasks "
    "(10,000 resamples); (C) BRIDGE: no harmful acceptance, so the upper bound is the informative quantity. "
    "Reduction: naive minus BRIDGE, cluster-bootstrap interval over datasets, tasks or clinical tasks (TABCI); over the 6 clinical sources the "
    "interval is CLINSRC. Upper bound of BRIDGE: KL-ratio bound UHAT (eq. Uhat) on (A), (C), (D), and (E); Proposition 1 bound U on (B). Setting (E) combines 11 binary and 3 multiclass datasets; its subtype checks are reported separately in the scope table. Net gain kept and the random-selection control: TABSEL.",
    "tab:summary", natural=True,
    hdroverrides={"Naive: accepted": Raw(r"\begin{tabular}[b]{@{}c@{}}Naive:\\accepted\end{tabular}"),
                  "Naive % harmful [95% CI]": Raw(r"\begin{tabular}[b]{@{}c@{}}Naive \% harmful\\{[}95\% CI{]}\end{tabular}"),
                  "BRIDGE: accepted": Raw(r"\begin{tabular}[b]{@{}c@{}}BRIDGE:\\accepted\end{tabular}"),
                  "BRIDGE % harmful [95% CI]": Raw(r"\begin{tabular}[b]{@{}c@{}}BRIDGE \% harmful\\{[}95\% CI{]}\end{tabular}"),
                  "Reduction in % harmful, pp [95% CI]": Raw(r"\begin{tabular}[b]{@{}c@{}}Reduction, pp\\{[}95\% CI{]}\end{tabular}"),
                  "BRIDGE 95% upper bound %": Raw(r"\begin{tabular}[b]{@{}c@{}}Upper bound\\$\hat U$ or $U$ (\%)\end{tabular}"),
                  "Naive net gain kept by BRIDGE % [95% CI]": Raw(r"\begin{tabular}[b]{@{}c@{}}Net gain kept \%\\{[}95\% CI{]}\end{tabular}")})
_t0 = OUT / "tab0_summary.tex"
_s0 = re.sub(r"<10\^\{(-?\d+)\}", r"$<10^{\1}$", _t0.read_text())
_s0 = re.sub(r"(\d+\.\d) \[(\d+\.\d); (\d+\.\d)\]", r"$\1\\ [\2;\\,\3]$", _s0)
_cr = _sl[_sl.setting == "Clinical, net benefit"].iloc[0]
_s0 = _s0.replace("TABCI", r"Table~\ref{tab:ci}").replace("TABSEL", r"Table~\ref{tab:selectivity}").replace(r"\\[95\% CI]", r"\\{[}95\% CI{]}")
_s0 = _s0.replace("CLINSRC", f"{abs(_cr.diff_hi_by_sumber):.1f} to {abs(_cr.diff_lo_by_sumber):.1f} pp")
_s0 = _s0.replace("UHAT (eq. Uhat)", r"$\hat U$ \eqref{eq:Uhat}").replace("Proposition 1 bound U on", r"Proposition 1 bound $U$ \eqref{eq:U} on")
_t0.write_text(_s0.replace("QQ", r"$\mathcal{Q}$").replace(" Q,", r" $\mathcal{Q}$,").replace("(Q", r"($\mathcal{Q}$").replace("on Q.", r"on $\mathcal{Q}$."))
print("tab0 written")

# Graphical abstract: how BRIDGE decides, and harmful proportions in the three scenarios
from matplotlib.patches import FancyBboxPatch
gfig = plt.figure(figsize=(7.2, 2.6))
axl = gfig.add_axes([0.0, 0.0, 0.47, 1.0]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
_steps = [("Candidate update\n(from the validation search)", "#DBEFF8"),
          ("Gain measured on validation folds\nnot used to choose the update", "#DBEFF8"),
          ("Bootstrap resampling gives a\npessimistic estimate of the gain", "#DBEFF8"),
          ("Accept only if the pessimistic\ngain exceeds a minimum improvement", "#D7F0D2")]
for k, (txt, col) in enumerate(_steps):
    y = 0.84 - k * 0.24
    axl.add_patch(FancyBboxPatch((0.06, y - 0.085), 0.88, 0.17, boxstyle="round,pad=0.01", fc=col, ec="#1f4e79", lw=0.8))
    axl.text(0.5, y, txt, ha="center", va="center", fontsize=7.2)
    if k < len(_steps) - 1:
        axl.annotate("", xy=(0.5, y - 0.105), xytext=(0.5, y - 0.085), arrowprops=dict(arrowstyle="-|>", color="#1f4e79", lw=0.8))
axl.text(0.5, 0.995, "BRIDGE: a fixed, model-agnostic acceptance gate", ha="center", va="top", fontsize=7.6, weight="bold")
axr = gfig.add_axes([0.56, 0.2, 0.42, 0.68])
_lab = ["(A) 11 new\ndatasets", "(B) Synthetic\ntasks", "(C) Clinical\ntasks", "(D) Supporting\ndatasets"]
_nv = [100 * r.naive_harmful / r.naive_accepted for _, r in _sl.iterrows()]
_br = [100 * r.bridge_harmful / r.bridge_accepted for _, r in _sl.iterrows()]
_x = np.arange(4); _w = 0.36
b1 = axr.bar(_x - _w / 2, _nv, _w, color="#cc4125", label="Naive acceptance")
b2 = axr.bar(_x + _w / 2, _br, _w, color="#1f4e79", label="BRIDGE")
for bars in (b1, b2):
    for b in bars:
        axr.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=6.8)
axr.set_xticks(_x); axr.set_xticklabels(_lab, fontsize=7); axr.set_ylabel("False improvements (%)", fontsize=7.2)
axr.set_ylim(0, 29); axr.tick_params(axis="y", labelsize=6.8); axr.legend(fontsize=6.6, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
axr.spines[["top", "right"]].set_visible(False)
axr.set_title("Same decisions, same locked reference", fontsize=7.6)
save(gfig, "graphical_abstract")
print("graphical abstract written")

_t10 = OUT / "tab10_candidate_space.tex"; _t10.write_text(_t10.read_text().replace(" & |C| & ", r" & $|\mathcal{C}|$ & ").replace("Data & |C| &", r"Data & $|\mathcal{C}|$ &"))
_t2 = OUT / "tab2_synthetic_population.tex"; _t2.write_text(_t2.read_text().replace("task population Q (", r"task population $\mathcal{Q}$ ("))
for _nm in ("tabS5_clinical_by_source", "tabS6_margin_sweep", "tabS7_paired_ci", "tabS8_selectivity", "tabS9_development_runs", "tabS12_cited_runs", "tabS10_mini39_char_Q", "tabS11_mini39_char_R", "tabS4_per_dataset", "tabS3_identity_audit", "tabS2_base_learner_audit"):
    _f = OUT / f"{_nm}.tex"; _x = _f.read_text()  # slightly tighter rows in the appendix tables (keeps the paper at 25 pages)
    if r"\renewcommand{\arraystretch}" not in _x:
        _f.write_text(_x.replace(r"\scriptsize", r"\scriptsize\renewcommand{\arraystretch}{0.9}", 1))

# Tuning as an update across learners (MINI-39, binary; MINI-40, multiclass, 3 datasets)


# Imbalanced classes (MINI-41)
_t41 = pd.read_csv(HERE / "out" / "analisis_mini41.csv")
imbrows = []
for m in ("wF1", "bacc", "macroF1"):
    for lv in (0.2, 0.1, 0.05):
        d = _t41[(_t41.metric == m) & (_t41.level == lv)]
        nv = d[d.aturan == "naif"].iloc[0]; br = d[d.aturan == "BRIDGE"].iloc[0]
        imbrows.append([{"wF1": "weighted-F1", "bacc": "balanced accuracy", "macroF1": "macro-F1"}[m], f"{100 * lv:.0f}%",
                        _pct(nv.terima, nv.harmful), _pct(br.terima, br.harmful), _num(br.batas), _num(br.LOCO), _num(br.R0),
                        f"{br.yield_:.1f}", str(br.status).startswith("LULUS")])
CAPS["tab13_imbalance"] = "Imbalanced classes on five datasets of (A): with balanced accuracy and macro-F1 BRIDGE meets the criterion down to 5% prevalence."
tex(pd.DataFrame(imbrows, columns=["Metric", "Minority prevalence", "Naive: accepted / harmful (%)", "BRIDGE: accepted / harmful (%)",
                                   "Bound (%)", "LOCO (%)", "R0 (%)", "Yield", "Criterion"]),
    "tab13_imbalance",
    "Five large binary datasets of (A) (Census-Income-KDD, Homesite, KDD98, LICD, Santander2016), the only ones whose locked "
    "reference keeps at least 800 minority cases at every level. Per dataset and level the rows are subsampled once, with a fixed seed, to the stated "
    "minority prevalence before the locked outer split, so the reference has the same prevalence; about 360, 180 and 90 minority cases fall in POOL "
    "at 20%, 10% and 5%. 100 trials per dataset and level (500 decisions per cell); the trial code, frozen gate and candidate space are unchanged. "
    "Pre-registered criterion as in Table criterion. Pre-registration MINI-41.",
    "tab:imbalance", natural=True)
print("tab13 written")
for _n in ("tab12_tuning_learners", "tab13_imbalance", "tabS13_hyper_basis"):
    _p = OUT / f"{_n}.tex"; _p.write_text(_p.read_text().replace("as in Table criterion", r"as in Table~\ref{tab:criterion}"))

# Headline table for the introduction: the core contrast only, in positive form
HL = lambda *l: Raw(r"\begin{tabular}[b]{@{}c@{}}" + r"\\".join(l) + r"\end{tabular}")
hlrows = []
_hlname = {"11 new datasets": "(A) 11 new datasets", "Synthetic population Q": "(B) Synthetic tasks",
           "Clinical, net benefit": "(C) Clinical tasks", "39 supporting datasets": "(D) Supporting datasets"}
for _, r in _sl.iterrows():
    hlrows.append([_hlname[r.setting], Raw(f"{100 * r.naive_harmful / r.naive_accepted:.1f}"),
                   Raw(f"{100 * r.bridge_harmful / r.bridge_accepted:.1f}"), Raw(f"{r.gain_kept:.0f}"),
                   Raw(f"{r.gain_per_acc_bridge / r.gain_per_acc_naive:.1f}$\\times$")])
CAPS["tab0_headline"] = "Main result: naive acceptance against BRIDGE on the same decisions."
tex(pd.DataFrame(hlrows, columns=["Setting", "Naive: harmful %", "BRIDGE: harmful %", "Net gain kept %", "Gain per accepted update"]),
    "tab0_headline", "Harmful %: pooled over all decisions of a setting (one fresh seed per decision). Gain per update: BRIDGE relative to the naive rule. 95% intervals, bounds and definitions: TABSUM. Every setting: current model HGB with library settings; candidates from the default space built from HGB, DT and SVC ($173$ binary, $14+8K$ multiclass; Section~\\ref{sec:method-partition}); other learners and ensemble members: Table~\\ref{tab:scope}.", "tab:headline", natural=True, crit_note="",
    hdroverrides={"Naive: harmful %": HL("Naive", r"harmful \% $\downarrow$"),
                  "BRIDGE: harmful %": HL("BRIDGE", r"harmful \% $\downarrow$"),
                  "Net gain kept %": HL("Net gain", r"kept \% $\uparrow$"),
                  "Gain per accepted update": HL("Gain per", r"update $\uparrow$")})
_th = OUT / "tab0_headline.tex"
_th.write_text(_th.read_text().replace("TABSUM", r"Table~\ref{tab:summary} (Section~\ref{sec:exp-summary})")
               .replace("smaller/larger is better}", "smaller/larger is better.}"))
print("tab0_headline written")

# Tuning tests (MINI-44): tuned baseline and per-task tuned gate against the paper's fixed design
import analysis_mini44 as A44  # noqa: E402
_d44 = A44.load(); tturows = []
_h1 = pd.read_csv(HERE / "out" / "analysis_mini44_h1.csv").set_index("setting"); _h2 = pd.read_csv(HERE / "out" / "analysis_mini44_h2.csv").set_index("setting")
for S, m in _d44.groupby("S"):
    m = m.reset_index(drop=True); u = m.dataset.to_numpy(); fr, hr = m.f_gain_ref.to_numpy(), m.h_gain_ref.to_numpy()
    fa = (m.has & (m.f_lb05 > .005)).to_numpy(); fn = (m.f_gain_pool > 0).to_numpy()
    arms = [("Paper design", fn, fr, fa, fr, "--", "--"),
            ("Tuned HGB baseline", (m.h_naive == 1).to_numpy(), hr, (m.h_accept == 1).to_numpy(), hr,
             f"{_h1.loc[S].baseline_gain_from_tuning_wf1_pts:+.2f} [{_h1.loc[S].bg_lo:+.2f}; {_h1.loc[S].bg_hi:+.2f}]", "--"),
            ("Gate tuned per task", fn, fr, (m.t_accept == 1).to_numpy(), fr, "--",
             f"{_h2.loc[S].netgain_diff_per100:+.3f} [{_h2.loc[S].ng_lo:+.3f}; {_h2.loc[S].ng_hi:+.3f}]")]
    for nm, nv, nr, ba, br_, bg, ng in arms:
        c = A44.crit(ba, br_ <= 0, u)
        tturows.append([{"A": "(A)", "D": "(D)"}[S], Raw(nm), _pct(int(nv.sum()), int((nv & (nr <= 0)).sum())),
                        _pct(c["accepted"], c["harmful"]), _num(c["kl"]), _num(c["loco"]), _num(c["r0"]), c["meets"], bg, ng])
CAPS["tab14_tuning_tests"] = ("Tuning tests on the real data: a tuned baseline and a gate tuned per task against the paper's fixed design, "
                              "on the same decisions.")
tex(pd.DataFrame(tturows, columns=["Setting", "Design", "Naive: accepted / harmful (%)", "BRIDGE: accepted / harmful (%)", "Bound (%)",
                                   "LOCO (%)", "R0 (%)", "Criterion", "Baseline change, F1 points [95% CI]",
                                   "Net gain vs paper gate, per 100 [95% CI]"]),
    "tab14_tuning_tests",
    "Paper design: fixed HGB and frozen gate. Same trials as the paper's runs (same seeds, splits and bags; the paper-design rows reproduce the stored decisions exactly in all 4,800 trials). "
    "Tuned HGB: Optuna TPE, 25 iterations, 3-fold cross-validation inside SEARCH only, the HGB space and budget of the tuning runs; the tuned HGB "
    "replaces the fixed one as current model and member. Gate tuned per task: POOL split into two thirds and one third; the gate runs on the two "
    "thirds for every $q\\in\\{2.5,5,10\\}%$ and $\\varepsilon\\in\\{0.0025,0.005,0.0075,0.01\\}$, the chosen candidate is scored on the third, "
    "and the best pair (ties: the default) is applied on the full POOL. Brackets: cluster bootstrap over datasets, B = 10,000. "
    "Pre-registered criterion as in Table criterion. Pre-registration MINI-44.",
    "tab:tuningtests", natural=True)

# Quantile level and margin on the real data (MINI-44)
_qe = pd.read_csv(HERE / "out" / "analysis_mini44_qeps.csv"); qerows = []
for S in ("A", "D"):
    for q in (0.025, 0.05, 0.1):
        row = [{"A": "(A)", "D": "(D)"}[S], f"{100 * q:g}%"]
        for e in (0.0025, 0.005, 0.0075, 0.01):
            r = _qe[(_qe.setting == S) & np.isclose(_qe.q, q) & np.isclose(_qe.eps, e)].iloc[0]
            mark = "default" if pd.isna(r.p_tost) else ("eq." if r.equivalent else "n.e.")
            row.append(f"{int(r.accepted):,}/{int(r.harmful)}; {r.kl:.2f}; {mark}")
        qerows.append(row)
CAPS["tab15_qeps_real"] = ("Quantile level and margin on the real data: every one of the 24 cells meets the pre-registered "
                           "criterion.")
tex(pd.DataFrame(qerows, columns=["Setting", "q", "eps 0.0025", "eps 0.005", "eps 0.0075", "eps 0.01"]),
    "tab15_qeps_real",
    "Cells: accepted / harmful updates; KL-ratio bound (%); equivalence of the harmful proportion with the default ($q=5%$, "
    "$\\varepsilon=0.005$) by a paired TOST with margin $\\pm2$ percentage points and Holm correction over the 11 variants of a setting "
    "(eq.: equivalent; n.e.: not shown equivalent, which with 11 datasets reflects limited power, not a difference). All cells also meet LOCO, "
    "R0 and yield. Decisions at every $(q,\\varepsilon)$ are taken from the same cross-fit counts and bootstrap draws as the default. "
    "Pre-registration MINI-44.",
    "tab:qepsreal", natural=True,
    hdroverrides={"q": Raw("$q$"), "eps 0.0025": Raw("$\\varepsilon=0.0025$"), "eps 0.005": Raw("$\\varepsilon=0.005$"),
                  "eps 0.0075": Raw("$\\varepsilon=0.0075$"), "eps 0.01": Raw("$\\varepsilon=0.01$")})

# Margin: paired equivalence with the default in the four settings (analysis_margin_paired.py)
_mp = pd.read_csv(HERE / "out" / "analysis_margin_paired.csv"); mprows = []
for _, r in _mp.iterrows():
    mprows.append([r.setting.split(" ")[0], f"{r.eps:g}", f"{int(r.acc_variant):,} / {int(r.harm_variant)}",
                   f"{r.diff_pp:+.2f} [{r.ci95_lo:+.2f}; {r.ci95_hi:+.2f}]", ("$<10^{-4}$" if r.p_tost_holm < 1e-4 else f"{r.p_tost_holm:.3f}"),
                   bool(r.equivalent), f"{r.yield_ratio:.2f}", f"{r.gain_ratio:.2f}"])
CAPS["tab16_margin_paired"] = "Margin against the default 0.005 on the same decisions: the harmful proportion is equivalent at every tested margin."
tex(pd.DataFrame(mprows, columns=["Setting", "Margin", "Accepted / harmful", "Harmful % vs default, pp [95% CI]", "TOST p (Holm)",
                                  "Equivalent (±2 pp)", "Accepted, ratio to default", "Net gain, ratio to default"]),
    "tab16_margin_paired",
    "The decision at margin $\\varepsilon$ is the stored 5% lower bound compared with $\\varepsilon$, so all margins are judged on the same trials. "
    "Default accepted / harmful: (A) 284 / 3, (B) 475 / 8, (C) 1,199 / 0, (D) 1,113 / 6. Brackets: cluster bootstrap over datasets, tasks or "
    "clinical tasks, B = 10,000; TOST with equivalence margin $\\pm2$ percentage points, Holm over the 12 comparisons; specification committed "
    "before the computation. In (C) no accepted update is harmful at any margin.",
    "tab:marginpaired", natural=True, hdroverrides={"Equivalent (±2 pp)": Raw(r"\begin{tabular}[b]{@{}c@{}}Equivalent\\($\pm2$ pp)\end{tabular}")})
for _n in ("tab14_tuning_tests", "tab15_qeps_real", "tab16_margin_paired"):
    _p = OUT / f"{_n}.tex"; _p.write_text(_p.read_text().replace("as in Table criterion", r"as in Table~\ref{tab:criterion}")
                                          .replace(r"\\[95\% CI]", r"\\{[}95\% CI{]}"))
print("tab14-16 written")

# Member families (MINI-45) and base learners on real data (MINI-46)
_ARMNAME = {"default": "DT and SVC (paper)", "LR-NB": "Logistic regression and naive Bayes", "MLP-KNN": "MLP and KNN",
            "all": "All tested families (711 binary configurations)", "XGB0": "XGBoost, 300 rounds", "XGBES": "XGBoost, early stopping", "MLPES": "MLP, early stopping"}
for _run, _name, _lbl, _cap, _note, _col in (
        ("mini45", "tabS15_member_families", "tab:members",
         "Other model families as ensemble members on (A) and (D): the gate meets the criterion when families are exchanged or combined.",
         "The two extra members are either replaced by another pair (MINI-45) or all seven tested member families are combined in a 711-configuration binary space (MINI-50); the current model (HGB), gate and every seed remain unchanged. No member is tuned.",
         "Ensemble members"),
        ("mini46", "tabS16_base_learners_real", "tab:basereal",
         "Base learners on the real data (A) and (D): with early stopping, XGBoost and MLP meet the criterion; XGBoost without iteration control fails on (A).",
         "The base learner (current model and member of every candidate) replaced; DT, SVC, candidate space, gate and seeds unchanged; settings as on the "
         "synthetic tasks (Table~\\ref{tab:base}). No learner is tuned. Pre-registration MINI-46.",
         "Base learner")):
    _a = pd.read_csv(HERE / "out" / f"analysis_{_run}.csv"); _rows = []
    if _run == "mini45":
        _m50 = pd.read_csv(HERE / "out" / "analysis_mini50.csv").assign(arm="all")
        _a = pd.concat([_a, _m50], ignore_index=True, sort=False)
    for _, r in _a.iterrows():
        _rows.append([{"A": "(A)", "D": "(D)"}[r.setting], ("HGB (paper)" if (_run == "mini46" and r.arm == "default") else _ARMNAME[r.arm]),
                      f"{r.naive_harm_pct:.1f}", _pct(r.bridge_accepted, r.bridge_harmful), _num(r.bridge_kl), _num(r.bridge_loco), _num(r.bridge_r0),
                      bool(r.bridge_meets)])
    CAPS[_name] = _cap
    tex(pd.DataFrame(_rows, columns=["Setting", _col, "Naive harmful %", "BRIDGE: accepted / harmful (%)", "Bound (%)", "LOCO (%)", "R0 (%)", "Criterion"]),
        _name, _note + " Criterion: Table criterion.", _lbl, rc=[yn_color(r[-1]) for r in _rows], natural=True)
    _pp = OUT / f"{_name}.tex"; _pp.write_text(_pp.read_text().replace("Criterion: Table criterion.", r"Criterion: Table~\ref{tab:criterion}."))
print("tabS15/S16 written")

# Absolute final-model quality on the real data (MINI-49)
_q49 = pd.read_csv(HERE / "out" / "analysis_mini49_overall.csv")
_q49rows = [[f"({r.setting})", r.learner, f"{r.baseline_mean:.3f}", f"{r.bridge_final_mean:.3f}",
             f"{r.bridge_minus_hgb_pts:+.3f} [{r.ci_lo:+.3f}; {r.ci_hi:+.3f}]",
             f"{int(r.datasets_better_than_hgb)} / {int(r.datasets_equal_hgb)} / {int(r.datasets_worse_than_hgb)}"]
            for r in _q49.itertuples()]
CAPS["tabS18_absolute_quality"] = "Absolute weighted-F1 of the final models after BRIDGE for four base learners on (A) and (D)."
tex(pd.DataFrame(_q49rows, columns=["Setting", "Base learner", "Baseline mean", "Final mean after BRIDGE",
                                    "Final minus HGB, F1 points [95% CI]", "Datasets better / equal / worse than HGB"]),
    "tabS18_absolute_quality", "The first 20 trials of every dataset; paired cluster-bootstrap intervals over datasets, B=10,000. MINI-49.",
    "tab:absolutequality", natural=True)
_p18 = OUT / "tabS18_absolute_quality.tex"
_p18.write_text(_p18.read_text().replace(r"\\[95\% CI]", r"\\{[}95\% CI{]}"))
print("tabS18 written")

# Clinical consequences per 1,000 observations of the updates BRIDGE accepts (net benefit p_t = 0.2; analysis_review4.py, part 1)
_cc = SP.rd("mini30_m1.csv", "mini30_m2.csv", "mini30_m3.csv"); _cc = _cc[(_cc.metric == "nb20") & (_cc.lo95 > .005)].copy()
_c = np.array([list(map(int, x.split())) for x in _cc.counts_ref]).reshape(-1, 2, 2, 2)
_b, _d, _n, _w = _c.sum(axis=3), _c.sum(axis=2), _c.sum(axis=(1, 2, 3)), .2 / .8
_cc["tp"] = 1000 * (_d[:, 1, 1] - _b[:, 1, 1]) / _n; _cc["fp"] = 1000 * (_d[:, 0, 1] - _b[:, 0, 1]) / _n
_cc["fb"] = 1000 * _b[:, :, 1].sum(1) / _n; _cc["fa"] = 1000 * _d[:, :, 1].sum(1) / _n
_cc["nbb"] = _b[:, 1, 1] / _n - _b[:, 0, 1] / _n * _w; _cc["nba"] = _d[:, 1, 1] / _n - _d[:, 0, 1] / _n * _w
_pi = _d[:, 1, :].sum(1) / _n; _cc["nbt"] = _pi - (1 - _pi) * _w
_crow = lambda nm, g: [nm, f"{len(g):,}", f"{g.fb.mean():.0f} $\\to$ {g.fa.mean():.0f}", f"{g.tp.mean():.1f}", f"{g.fp.mean():.1f}",
                       f"{g.nbb.mean():.4f}", f"{g.nba.mean():.4f}", f"{g.nbt.mean():.4f}"]
_TASKEN = {"mammography": "Mammography", "actg175": "ACTG 175 (HIV-1)", "nhanes_hipertensi": "NHANES hypertension", "nhanes_diabetes": "NHANES diabetes",
           "nhanes_asma": "NHANES asthma", "nhanes_pjk": "NHANES coronary heart disease", "brfss_stroke": "BRFSS stroke", "brfss_asma": "BRFSS asthma",
           "brfss_ginjal": "BRFSS kidney disease", "brfss_depresi": "BRFSS depression", "covid_mx": "COVID-19 Mexico", "wids_icu": "ICU (WiDS 2020)"}
_crows = [_crow(_TASKEN.get(k, k), g) for k, g in _cc.groupby("dataset", sort=False)] + [_crow("All accepted updates", _cc)]
CAPS["tabS17_clinical_consequences"] = "Clinical consequences of the updates BRIDGE accepts under net benefit at $p_t=0.2$, per 1,000 observations of the locked reference."
tex(pd.DataFrame([[Raw(x) if "$" in str(x) else x for x in r] for r in _crows],
                 columns=["Task", "Accepted updates", "Flagged per 1,000, before $\\to$ after", "Extra true positives per 1,000",
                          "Extra false positives per 1,000", "Net benefit before", "Net benefit after", "Net benefit, treat all"]),
    "tabS17_clinical_consequences",
    "Means over the accepted updates of each task (MINI-30, 200 trials per task). Flagged: observations predicted positive. Net benefit as in (10); "
    "treat all: every observation flagged. Exploratory; script analysis_review4.py.",
    "tab:clinconseq", natural=True,
    hdroverrides={"Flagged per 1,000, before $\\to$ after": Raw(r"\begin{tabular}[b]{@{}c@{}}Flagged per 1,000\\before $\to$ after\end{tabular}")})
print("tabS17 written")

# Grouping columns: a value repeated from the row above is left blank
def _blank_repeats(name, cols):
    pth = OUT / f"{name}.tex"; lines = pth.read_text().split("\n"); inb = False; prev = {}
    for k, ln in enumerate(lines):
        if ln.startswith(r"\midrule"):
            inb = True; continue
        if ln.startswith(r"\bottomrule"):
            break
        if not inb or "&" not in ln:
            continue
        cells = re.split(r"(?<!\\)&", ln)
        for c in cols:
            v = cells[c].strip()
            if v and prev.get(c) == v:
                cells[c] = " "
            elif v:
                prev[c] = v
                for c2 in cols:
                    if c2 > c:
                        prev.pop(c2, None)
        lines[k] = "&".join(cells)
    pth.write_text("\n".join(lines))
for _n, _c in (("tabS15_member_families", (0,)), ("tabS16_base_learners_real", (0,)), ("tab10_candidate_space", (0, 1)), ("tab12_tuning_learners", (0,)), ("tab13_imbalance", (0,)), ("tab14_tuning_tests", (0,)),
               ("tab15_qeps_real", (0,)), ("tab16_margin_paired", (0,)), ("tabS10_mini39_char_Q", (0,)), ("tabS12_cited_runs", (0, 2)),
               ("tabS9_development_runs", (0, 2))):
    _blank_repeats(_n, _c)
print("repeats blanked")

# Letter namespace (ruler 26.9a): no single letter both a symbol and a setting/run/hypothesis label
for _p in sorted(OUT.glob("*.tex")):
    _s0 = _s = _p.read_text()
    _s = re.sub(r"\$R_0\$", r"LOCO$_0$", _s); _s = re.sub(r"(?<![A-Za-z$\\])R0(?![0-9A-Za-z])", r"LOCO$_0$", _s)
    _s = _s.replace("B = 10,000", "10,000 resamples").replace("2/B = 0.0002", "2/10,000 = 0.0002")
    _s = re.sub(r"reference seed R([12])\b", r"reference seed \1", _s)
    _s = _s.replace("each test is of H0: the two harmful proportions are equal, against H1: the two proportions differ",
                    "each test has the null hypothesis of equal harmful proportions against the alternative that they differ")
    _parts = re.split(r"((?<![^\\]\\)\$)", _s); _inm = False  # a $ preceded by exactly one backslash is a literal dollar; \\$ is a line break then math
    for _k, _x in enumerate(_parts):
        if _x == "$":
            _inm = not _inm
        elif _inm:
            _parts[_k] = re.sub(r"(?<![\\A-Za-z{])B(?![A-Za-z])", r"N_{\\mathrm{boot}}", _x)
    _s = "".join(_parts)
    if _s != _s0:
        _p.write_text(_s)
print("namespace pass done")
