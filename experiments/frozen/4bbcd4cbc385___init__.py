"""BRIDGE: a frozen acceptance gate for post-validation model updates.

The gate answers one question: should the candidate model replace the current model? It uses only labeled data that
the models were not trained on (POOL) and the two models' predicted labels on it. The number of classes is read from
the labels; binary and multiclass tasks use the same rule.

    >>> import bridge_gate as bridge
    >>> d = bridge.gate(y_pool, pred_current, pred_candidate)          # one prespecified candidate (e.g. a tuned model)
    >>> d = bridge.gate_select(y_pool, pred_current, {"a": pa, "b": pb})  # choose among several candidates (cross-fitted)
    >>> d.accept, d.lower_bound, d.warnings

Frozen settings (chosen during development and never changed afterwards): margin 0.005, lower bootstrap quantile 5%,
20,000 bootstrap replicates, 3 cross-fitting folds.

The whole validated procedure of the paper (training draw, 32 bags of HGB/DT/SVC, candidate space, gate) is
bridge_gate.protocol.run; the bridge-gate program runs it by default.
"""
from .core import (MARGIN, QUANTILE, REPLICATES, FOLDS, Decision, counts, delta, lower_bound, gate, gate_select,
                   check, split)
from . import protocol

__all__ = ["MARGIN", "QUANTILE", "REPLICATES", "FOLDS", "Decision", "counts", "delta", "lower_bound", "gate",
           "gate_select", "check", "split", "protocol"]
__version__ = "0.8.1"
