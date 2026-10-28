"""Plain-language result card shared by the command line and the Python API.

The card has four parts: the two models side by side, the two evidence checks
with the value each needs, a short explanation, and whether the decision lies
inside the conditions tested in the paper. It only formats numbers that the
gate has already computed.
"""
import textwrap

from .core import MARGIN, QUANTILE, REPLICATES

METRIC_LABEL = {"wF1": "weighted F1", "acc": "accuracy", "bacc": "balanced accuracy", "macroF1": "macro F1",
                "nb10": "net benefit at 0.1", "nb20": "net benefit at 0.2", "F2": "F2"}
WIDTH = 78
CLINICAL_METRICS = ("nb10", "nb20", "F2")       # tested on the retrospective clinical tasks only
UNCHECKABLE = ("the judging rows were not used to train, tune or choose any model",
               "the candidates were fixed before the judging rows were examined",
               "the judging rows come from the same population as the data the model will see")


TESTED_MULTI = ("wF1", "bacc", "macroF1")       # metrics tested with more than two classes (MINI-54)


def metric_condition(metric, K):
    """The condition on the metric that is not met, or None."""
    if K == 2 and metric in CLINICAL_METRICS:
        return f"metric {metric}: tested only on the 12 retrospective clinical tasks"
    if (K == 2 and metric not in ("wF1", "bacc", "macroF1")) or (K > 2 and metric not in TESTED_MULTI):
        return (f"metric {metric}: the tested metrics are wF1, bacc and macroF1 (two classes) and "
                f"{', '.join(TESTED_MULTI)} (more classes)")
    return None


def pool_conditions(n, K, metric, smallest_class):
    """Tested conditions that can be checked from the judging rows alone; returns those not met."""
    from .protocol import CONFIRMED_K, SMALL_SEARCH, SMALL_TOTAL
    c, min_pool = [], SMALL_TOTAL - SMALL_SEARCH
    if n < min_pool:
        c.append(f"{n} judging rows: the tested runs judged on at least {min_pool} rows")
    if K > 2 and not CONFIRMED_K[0] <= K <= CONFIRMED_K[1]:
        c.append(f"{K} classes: the multiclass confirmation covers {CONFIRMED_K[0]}-{CONFIRMED_K[1]} classes")
    if smallest_class < 30:
        c.append(f"smallest class has {smallest_class} judging rows: only large improvements can be accepted")
    mc = metric_condition(metric, K)
    return c + ([mc] if mc else [])


def _cut(text, w):
    text = str(text)
    return text if len(text) <= w else text[:w - 1] + "~"


def card(accept, n, metric, current, candidate, score_current, score_candidate, right_current, right_candidate,
         gain, lower_bound, n_tried, margin=MARGIN, reference=None, closest=None, conditions=(), uncheckable=()):
    """Return the result card as text.

    candidate is None when no candidate scored higher than the current model; closest = (name, score, right)
    then shows the candidate that came nearest.
    reference: optional (rows, gain, low, high, verdict) from a check on rows the decision never used.
    conditions: tested conditions that this decision does not meet (empty: all met).
    uncheckable: conditions of use that the program cannot check, listed as reminders.
    """
    m = METRIC_LABEL.get(metric, metric); line = "=" * WIDTH; rule = "  " + "-" * (WIDTH - 2)
    out = [line, f"  RESULT: {'KEEP the current model' if not accept else 'INSTALL the candidate' if candidate == 'candidate' else 'INSTALL the candidate ' + _cut(candidate, 40)}", line]

    # 1. the two models side by side
    head, name = "Best candidate", candidate
    if candidate is None and closest is not None:
        head, (name, score_candidate, right_candidate) = "Closest candidate", closest
    out.append(f"  {'':26s}{'Current model':22s}{head}")
    out.append(f"  {'Model':26s}{_cut(current, 21):22s}{_cut(name if name is not None else '-', WIDTH - 50)}")
    sc = f"{score_candidate:.3f}" if score_candidate is not None else "-"
    out.append(f"  {_cut('Score (' + m + ')', 25):26s}{score_current:<22.3f}{sc}")
    rc = f"{right_candidate} of {n}" if right_candidate is not None else "-"
    out.append(f"  {'Rows predicted correctly':26s}{f'{right_current} of {n}':22s}{rc}")
    out.append(rule)

    # 2. the evidence checks
    out.append(f"  {'Evidence check (both must pass to install)':45s}{'value':>9s}{'needed':>12s}{'pass?':>8s}")
    ok1 = gain is not None and gain > 0
    g = f"{gain:+.4f}" if gain is not None else "-"
    out.append(f"  {'1. Candidate scores higher on the judging rows':45s}{g:>9s}{'above 0':>12s}{'yes' if ok1 else 'no':>8s}")
    ok2 = lower_bound > margin and candidate is not None
    q = f"worst {QUANTILE:.0%}"
    lb = f"{lower_bound:+.4f}" if candidate is not None else "-"
    out.append(f"  {f'2. Still higher in an unlucky case ({q})':45s}{lb:>9s}{f'> {margin:+.4f}':>12s}"
               f"{'yes' if ok2 else 'no' if candidate is None else 'NO':>8s}")
    out.append(rule)

    # 3. plain words
    if candidate is None:
        who = "The candidate did not" if n_tried == 1 else f"None of the {n_tried} candidates"
        why = f"{who} score higher than the current model on the {n} judging rows, so there is nothing to install."
    elif accept:
        why = (f"The candidate scores higher, and the gain stays above the required minimum of {margin:+.4f} even in an "
               f"unlucky case. A gain this firm is very unlikely to be luck, so the candidate can replace the current model.")
    else:
        luck = (f"{n_tried} candidates were tried, and when many are tried some score higher by luck alone. "
                if n_tried > 1 else "A small difference on these rows can be luck. ")
        why = (f"The candidate scores higher on these rows, but {luck}In an unlucky case the gain falls to "
               f"{lower_bound:+.4f}, below the required {margin:+.4f}. The higher score is therefore not proof of a "
               f"real improvement, and the current model stays.")
    out.append("  In plain words:")
    out += ["    " + s for s in textwrap.wrap(why, WIDTH - 4)]
    if reference is not None:
        rows, gr, lo, hi, verdict = reference
        out.append(rule)
        txt = (f"Check on {rows} extra rows that the decision never used: {m} gain {gr:+.4f} "
               f"(95% interval {lo:+.4f} to {hi:+.4f}), so the change is {verdict}.")
        out += ["  " + s for s in textwrap.wrap(txt, WIDTH - 2)]
    out.append(rule)
    if conditions:
        out.append(f"  TESTED CONDITIONS: NOT MET ({len(conditions)})")
        for c in conditions:
            out += textwrap.wrap(c, WIDTH - 6, initial_indent="    - ", subsequent_indent="      ")
        out += textwrap.wrap("The gate still decides, but the results of the paper (at most 1.7% false improvements "
                             "among accepted updates) do not cover this decision. Treat it as a first screen.",
                             WIDTH - 4, initial_indent="    ", subsequent_indent="    ")
    else:
        out += textwrap.wrap("TESTED CONDITIONS: MET. The decision lies inside the conditions tested in the paper, "
                             "where at most 1.7% of the accepted updates were false improvements.",
                             WIDTH - 4, initial_indent="  ", subsequent_indent="    ")
    if uncheckable:
        out.append("  The program cannot check these; make sure that:")
        for c in uncheckable:
            out += textwrap.wrap(c, WIDTH - 6, initial_indent="    - ", subsequent_indent="      ")
    out.append(line)
    note = (f"Judging rows: labeled rows that no model saw in training. Score: {m}, higher is better. Unlucky case: "
            f"the lowest {QUANTILE:.0%} of {REPLICATES:,} resamples of the judging rows.")
    out += ["  " + s for s in textwrap.wrap(note, WIDTH - 2)]
    return "\n".join(out)
