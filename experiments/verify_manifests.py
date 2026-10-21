"""Check the run manifests in out/*_MANIFEST*.sha256 against the files in this release.

Each line of a manifest is resolved in one of four ways:
  current  - the tree file has the recorded SHA-256;
  frozen   - the tree file was edited after the run. The exact run-time version is kept as
             frozen/<sha12>_<name> and has the recorded SHA-256. Python files are also compared
             as programs (abstract syntax tree without docstrings; comments are not part of it):
             'code identical' means that only comments, docstrings or the version string changed,
             and 'messages only' means that, in addition, only the text of string literals changed;
  withheld - the run-time version is withheld because its comments or text hold internal working
             notes; frozen/WITHHELD.tsv lists its SHA-256. For Python files the table also gives
             the SHA-256 of the program structure of the run-time version (computed with Python 3.12),
             which is compared with the current file. Plan and report documents are available to the
             editor and reviewers on request;
  missing  - no file in the release has the recorded SHA-256 (see the README, provenance notes).
Usage: python verify_manifests.py [-v]   (exit code 1 if an entry is missing outside the documented exceptions)"""
import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen"
DOCUMENTED_MISSING = {("mini17", "mini17_trainsize.py"), ("mini19", "mini19_fs_lebar.py"),
                      *((run, f) for run in ("settingE2_amendment1", "settingE2_tabular")
                        for f in ("screen_e2_external.py", "screen_e2_openml_custom.py"))}
WITHHELD = {r[0]: r for r in (line.split("\t") for line in (FROZEN / "WITHHELD.tsv").read_text().splitlines()[1:])}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def program(src, blank_strings=False):
    """Program structure without docstrings; the version string counts as metadata."""
    t = ast.parse(src)
    for n in ast.walk(t):
        for field in ("body", "orelse", "finalbody"):
            b = getattr(n, field, None)
            if isinstance(b, list):
                keep = [s for s in b if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant)
                                             and isinstance(s.value.value, str))
                        and not (isinstance(s, ast.Assign) and any(getattr(x, "id", "") == "__version__" for x in s.targets))]
                setattr(n, field, keep or [ast.Pass()])
    if blank_strings:
        for n in ast.walk(t):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                n.value = ""
    return ast.dump(t)


def resolve(name):
    for base in (HERE, HERE.parent, HERE / "out"):
        p = (base / name).resolve()
        if p.is_file():
            return p
    return None


def main():
    verbose = "-v" in sys.argv
    bad = 0
    for m in sorted((HERE / "out").glob("*_MANIFEST*.sha256")):
        run = m.name.split("_MANIFEST")[0]
        tally = {"current": 0, "frozen, code identical": 0, "frozen, messages only": 0, "frozen, code differs": 0,
                 "frozen": 0,
                 "withheld, code identical": 0, "withheld, code differs": 0, "withheld": 0, "missing": 0}
        notes = []
        for line in m.read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            h, name = parts[0], parts[-1]
            cur = resolve(name)
            if cur is not None and sha(cur) == h:
                tally["current"] += 1
                continue
            fz = FROZEN / f"{h[:12]}_{Path(name).name}"
            if fz.is_file() and sha(fz) == h:
                if name.endswith(".py") and cur is not None:
                    same = program(fz.read_text()) == program(cur.read_text())
                    if same:
                        key = "frozen, code identical"
                    elif program(fz.read_text(), True) == program(cur.read_text(), True):
                        key = "frozen, messages only"
                    else:
                        key = "frozen, code differs"
                else:
                    key = "frozen"
                tally[key] += 1
                if verbose or key in ("frozen, messages only", "frozen, code differs"):
                    notes.append(f"    {key}: {name} -> frozen/{fz.name}")
                continue
            if h in WITHHELD:
                fp = WITHHELD[h][3]
                if fp != "-" and cur is not None:
                    same = hashlib.sha256(program(cur.read_text()).encode()).hexdigest() == fp
                    key = "withheld, code identical" if same else "withheld, code differs"
                else:
                    key = "withheld"
                tally[key] += 1
                if verbose or key == "withheld, code differs":
                    notes.append(f"    {key}: {name}")
                continue
            tally["missing"] += 1
            documented = (run, Path(name).name) in DOCUMENTED_MISSING
            bad += not documented
            notes.append(f"    missing{' (documented)' if documented else ''}: {name}")
        print(f"{run:>16}: " + ", ".join(f"{k} {v}" for k, v in tally.items() if v))
        for s in notes:
            print(s)
    print("all entries accounted for" if not bad else f"{bad} undocumented missing entries")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
