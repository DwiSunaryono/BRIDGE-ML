"""BRIDGE: an acceptance gate for post-validation model updates.

The gate answers one question: should a candidate model replace the current
model? It uses labeled data that neither model saw during training (POOL) and
the predicted labels of both models on those rows. The number of classes is
read from the labels; binary and multiclass tasks use the same rule.

    >>> import bridge_gate as bridge
    >>> d = bridge.gate(y_pool, pred_current, pred_candidate)             # one candidate fixed in advance
    >>> d = bridge.gate_select(y_pool, pred_current, {"a": pa, "b": pb})  # several candidates, cross-fitted
    >>> d.accept, d.lower_bound, d.warnings

Frozen settings: margin 0.005, lower bootstrap quantile 5%, 20,000 bootstrap
replicates, 3 cross-fitting folds.

The full reference procedure (training draw, 32 bags of HGB/DT/SVC, candidate
space, gate) is bridge_gate.protocol.run. The bridge-gate command runs it by
default.
"""
from .core import (MARGIN, QUANTILE, REPLICATES, FOLDS, Decision, counts, delta, score, lower_bound, gate, gate_select,
                   check, split)
from . import protocol

__all__ = ["MARGIN", "QUANTILE", "REPLICATES", "FOLDS", "Decision", "counts", "delta", "score", "lower_bound", "gate",
           "gate_select", "check", "split", "protocol"]
__version__ = "1.1.1"
