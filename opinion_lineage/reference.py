"""Independent, dense reference implementations (different code path from the selective engine).

* dense_replay        : full re-summation over the *whole* corpus using the same revision-propagation rule
* full_reprocessing   : the conventional pipeline: re-encode + re-classify every record, then dense re-summation
"""
from __future__ import annotations
import numpy as np
from .config import Config
from .lineage import LineageGraph
from .weighting import compute_weights, propagation_confidence
from .rollback import (RevisionEvent, ROOT_INDEPENDENCE, EDGE_CONFIDENCE, RECORD_SYNTHETIC, RECORD_COORDINATION)


def dense_replay(g: LineageGraph, P, bucket, snap: dict, events, cfg: Config, B: int | None = None):
    conf, p_ind, p_syn, p_coord, W = (snap[k].copy() for k in ("conf", "p_ind", "p_syn", "p_coord", "W"))
    for ev in events:
        v = ev.record
        if ev.kind == ROOT_INDEPENDENCE: p_ind[v] = ev.value
        elif ev.kind == EDGE_CONFIDENCE: conf[v] = ev.value
        elif ev.kind == RECORD_SYNTHETIC: p_syn[v] = ev.value
        elif ev.kind == RECORD_COORDINATION: p_coord[v] = ev.value
        W_tilde = compute_weights(g, conf, p_ind, p_syn, p_coord, cfg)
        q = propagation_confidence(g, conf, v)            # 0 outside D(v)
        W = W + q * (W_tilde - W)
    A = (W[:, None] * P).sum(axis=0)
    B = B or int(bucket.max()) + 1
    A_b = np.zeros((B, P.shape[1])); np.add.at(A_b, bucket, W[:, None] * P)
    Z_b = np.bincount(bucket, weights=W, minlength=B)
    return dict(W=W, A=A, Z=float(W.sum()), A_b=A_b, Z_b=Z_b)


def full_reprocessing(models, X_raw, g: LineageGraph, bucket, snap: dict, events, cfg: Config):
    """Conventional approach: every record goes through the representation model and sentiment classifier again."""
    E = models.encoder(X_raw)
    P = models.clf.predict_proba(E)
    out = dense_replay(g, P, bucket, snap, events, cfg)
    out["P"] = P
    return out
