"""Basic checks of the bridge package. Run: python -m pytest pkg/tests, or python pkg/tests/test_basic.py."""
import contextlib
import io
import warnings
import numpy as np
import pandas as pd
import pytest
import bridge_gate as bridge
from bridge_gate import cli, protocol

warnings.filterwarnings("ignore")


def test_counts_binary_layout():
    c = bridge.counts([0, 1, 1], [0, 0, 1], [1, 1, 1])
    assert c.shape == (8,) and c[4 * 0 + 2 * 0 + 1] == 1 and c[4 * 1 + 2 * 0 + 1] == 1 and c[4 * 1 + 2 * 1 + 1] == 1


def test_identical_models_never_accepted():
    rng = np.random.default_rng(0); y = rng.integers(0, 3, 500); p = y.copy(); p[:50] = (p[:50] + 1) % 3
    d = bridge.gate(y, p, p)
    assert d.gain_pool == 0 and not d.accept


def test_clear_improvement_accepted():
    rng = np.random.default_rng(1); y = rng.integers(0, 2, 2000)
    old = y.copy(); old[:400] = 1 - old[:400]; new = y.copy(); new[:100] = 1 - new[:100]
    d = bridge.gate(y, old, new)
    assert d.accept and d.K == 2


def test_select_picks_better_candidate():
    rng = np.random.default_rng(2); y = rng.integers(0, 2, 1800)
    old = y.copy(); old[:300] = 1 - old[:300]
    good = y.copy(); good[:60] = 1 - good[:60]; bad = y.copy(); bad[:500] = 1 - bad[:500]
    d = bridge.gate_select(y, old, {"bad": bad, "good": good})
    assert d.chosen == "good" and d.accept


def test_protocol_candidates_and_labels():
    p = np.array([0.8, 0.3, 0.6])
    P = {"HGB": p, "DT": p, "SVC": p}
    cfg = ({"HGB": 1}, 0.5)
    assert list(protocol.labels(P, cfg, 2)) == [1, 0, 1]
    assert len(protocol.candidates(2)) == 173
    assert len(protocol.candidates(3)) > 0


def test_protocol_run_iris():
    from sklearn.datasets import load_iris
    d = load_iris(as_frame=True)
    r = protocol.run(d.data, d.target, seed=1)
    assert r.K == 3 and r.total == len(d.data) and 0 <= r.lower_bound <= 1


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(["--help"])
    assert exc.value.code == 0


if __name__ == "__main__":
    for f in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        f(); print("OK", f.__name__)
