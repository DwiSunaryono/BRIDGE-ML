"""Frozen runner for PRAREG_SETTING_E2.md.

Usage: python setting_e2.py settingE2_daftar_beku.csv OUT.csv NPROC [I/N]
"""
import csv, json, os, subprocess, sys, tempfile, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tabel_c as TC
import screen_e2_external as EXT
import screen_e2_openml_custom as CUSTOM

OFF = 1_045_000_000
SPLIT = Path.home() / "settingE2_split"
if "BRIDGE_PKG_SRC" in os.environ:
    PKG = Path(os.environ["BRIDGE_PKG_SRC"])
else:
    candidates = [HERE.parent.parent / "pkg" / "src", HERE.parent.parent.parent / "BRIDGE_repo" / "pkg" / "src"]
    PKG = next((path for path in candidates if path.is_dir()), candidates[0])
ENV = {**os.environ, "PYTHONPATH": str(PKG), "OMP_NUM_THREADS": "1"}
CUSTOM_IDS = set(CUSTOM.SPECS)
COLS = ["pos", "source_id", "name", "rep", "seed", "status", "K", "rows", "total", "bags", "validated", "chosen",
        "accept", "naive_accept", "lower_bound", "gain_pool", "gain_ref", "ref_lo", "ref_hi", "ref_class", "ref_rows", "seconds"]


def load_source(source_id):
    kind, number = source_id.split("-", 1); number = int(number)
    if kind == "UCI":
        return EXT.load_uci(number)
    if kind == "OpenML" and number in CUSTOM_IDS:
        return CUSTOM.load(number)
    if kind == "OpenML":
        return TC.load(number)
    raise ValueError(f"unsupported source {source_id}")


def split_paths(source_id):
    SPLIT.mkdir(exist_ok=True)
    stem = source_id.replace("-", "_")
    dev, ref = SPLIT / f"{stem}_dev.pkl", SPLIT / f"{stem}_ref.pkl"
    if not (dev.exists() and ref.exists()):
        from sklearn.model_selection import train_test_split
        X, y, _ = load_source(source_id)
        Xd, Xr, yd, yr = train_test_split(X, y, test_size=TC.ref_size(len(y)), stratify=y,
                                           random_state=TC.LOCKED_REF_SEED)
        for path, features, target in ((dev, Xd, yd), (ref, Xr, yr)):
            frame = features.reset_index(drop=True).copy(); frame["__target__"] = np.asarray(target)
            tmp = path.with_suffix(".tmp"); frame.to_pickle(tmp); tmp.rename(path)
    return dev, ref


def run_one(job):
    pos, source_id, name, rep = job; seed = OFF + 1000 * pos + rep; start = time.time()
    try:
        dev, ref = split_paths(source_id)
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-m", "bridge_gate", "--dataset", str(dev), "--reference", str(ref),
                "--target", "__target__", "--seed", str(seed), "--metric", "wF1", "--no-progress", "--out", f"{tmp}/res"],
                capture_output=True, text=True, env=ENV)
            if proc.returncode:
                raise RuntimeError((proc.stderr or proc.stdout).strip().splitlines()[-1][:200])
            result = json.load(open(f"{tmp}/res.json"))
        return dict(pos=pos, source_id=source_id, name=name, rep=rep, seed=seed, status="ok", K=result["K"],
            rows=result["rows"], total=result["total"], bags=result["bags"], validated=result["validated"],
            chosen=result["chosen_config"], accept=result["accept"], naive_accept=result["naive_accept"],
            lower_bound=repr(result["lower_bound"]), gain_pool=repr(result["gain_pool_chosen"]),
            gain_ref=repr(result["gain_ref"]), ref_lo=repr(result["ref_interval"][0]), ref_hi=repr(result["ref_interval"][1]),
            ref_class=result["ref_class"], ref_rows=result["reference_rows"], seconds=round(time.time()-start, 1))
    except Exception as exc:
        return dict(pos=pos, source_id=source_id, name=name, rep=rep, seed=seed,
                    status="ERROR " + str(exc)[:200], seconds=round(time.time()-start, 1))


if __name__ == "__main__":
    rows = list(csv.DictReader(open(sys.argv[1])))
    out, nproc = Path(sys.argv[2]), int(sys.argv[3])
    jobs = [(p, row["source_id"], row["nama"], rep) for p, row in enumerate(rows)
            for rep in range(100 if int(row["K"]) == 2 else 60)]
    if len(sys.argv) > 4:
        shard, count = map(int, sys.argv[4].split("/")); jobs = [job for job in jobs if job[0] % count == shard - 1]
    done = {(row["source_id"], int(row["rep"])) for row in csv.DictReader(open(out)) if row["status"] == "ok"} if out.exists() else set()
    todo = [job for job in jobs if (job[1], job[3]) not in done]
    print(f"{len(jobs)} trials in this part, {len(jobs)-len(todo)} already done", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as handle, Pool(nproc) as pool:
        writer = csv.DictWriter(handle, fieldnames=COLS, restval="")
        if new: writer.writeheader()
        for index, result in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            writer.writerow(result); handle.flush()
            if index % 20 == 0 or result["status"] != "ok":
                print(f"{index}/{len(todo)} {result['source_id']} rep {result['rep']}: {result['status']} {result['seconds']} s", flush=True)
    print("done", flush=True)
