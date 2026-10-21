"""Example: a small three-class task (Iris, 150 rows), like a fruit-grading model built from a small labeled set.

Should a new model replace the current one? Train both on SEARCH only, predict
POOL, and let BRIDGE decide. Install first: pip install ./pkg
Run: python examples/example_small_three_classes.py"""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import bridge_gate as bridge

X, y = load_iris(return_X_y=True)
search, pool = bridge.split(y, seed=1)                       # fit and tune on SEARCH; judge on POOL only
ok, messages = bridge.check(y[search], y[pool])
print("Requirements met:", ok)
for m in messages:
    print("  -", m)

current = DecisionTreeClassifier(random_state=0).fit(X[search], y[search])
candidate = LogisticRegression(max_iter=1000).fit(X[search], y[search])
d = bridge.gate(y[pool], current.predict(X[pool]), candidate.predict(X[pool]), seed=1)
print(f"\nClasses: {d.K}; POOL rows: {d.n}")
print(f"Gain on POOL (what the naive rule looks at): {d.gain_pool:+.4f} -> naive rule would {'install' if d.naive_accept else 'keep the current model'}")
print(f"BRIDGE lower bound: {d.lower_bound:+.4f} (must exceed {bridge.MARGIN}) -> {'INSTALL the candidate' if d.accept else 'KEEP the current model'}")
for w in d.warnings:
    print("  note:", w)
