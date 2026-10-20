"""Stand-ins for the expensive models, instrumented so that "zero repeated model execution" is *measured*.

* Encoder  : fixed random two-layer tanh network (raw features -> unit-norm semantic embedding)
* SentimentModel : multinomial logistic regression trained on embeddings
Every call increments a row counter; the rollback engine must leave both counters untouched.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression


class Encoder:
    def __init__(self, d_in: int, d_hidden: int = 256, d_out: int = 128, seed: int = 0):
        rng = np.random.default_rng(10_000 + seed)
        self.W1 = (rng.normal(size=(d_in, d_hidden)) / np.sqrt(d_in)).astype(np.float32)
        self.W2 = (rng.normal(size=(d_hidden, d_out)) / np.sqrt(d_hidden)).astype(np.float32)
        self.calls = 0
        self.rows = 0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.rows += len(X)
        H = np.tanh(X.astype(np.float32) @ self.W1)
        Z = H @ self.W2
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        return Z


class SentimentModel:
    def __init__(self, C: float = 1.0):
        self.clf = LogisticRegression(C=C, max_iter=1000)
        self.calls = 0
        self.rows = 0

    def fit(self, E: np.ndarray, y: np.ndarray) -> "SentimentModel":
        self.clf.fit(E, y)
        return self

    def predict_proba(self, E: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.rows += len(E)
        return self.clf.predict_proba(E).astype(np.float64)


class ModelBundle:
    """Encoder + classifier with a combined forward-pass ledger."""

    def __init__(self, encoder: Encoder, clf: SentimentModel):
        self.encoder, self.clf = encoder, clf

    @property
    def forward_passes(self) -> dict:
        return {"encoder_rows": self.encoder.rows, "classifier_rows": self.clf.rows}

    def snapshot(self) -> tuple:
        return (self.encoder.rows, self.clf.rows)
