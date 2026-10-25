"""Versioned Sentiment & Provenance Repository.

Per record a compact Sentiment Contribution Witness (columnar arrays): reference to the persisted sentiment
probability vector P_i, parent / edge confidence, provenance scores, denormalised root independence, the path factor
prod(1-alpha*c_e), the *applied* weight W_i, the day bucket, and the model/threshold version.
The aggregate is stored as per-bucket pairs (A_b, Z_b) (+ running totals).  Embeddings are NOT held here: the
representation store is only ever referenced, never re-computed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .config import Config
from .lineage import LineageGraph
from .weighting import compute_weights


@dataclass
class AuditEntry:
    version: int
    event: object
    affected_ids: np.ndarray
    w_prior: np.ndarray
    w_new: np.ndarray
    dA: np.ndarray
    dZ: float
    model_version: str


class OpinionStore:
    def __init__(self, g: LineageGraph, P: np.ndarray, p_ind, p_syn, p_coord, bucket, cfg: Config):
        self.cfg, self.g = cfg, g
        self.n, self.C = P.shape
        self.P = P                                    # persisted sentiment vectors (read-only reference)
        self.parent, self.child_ptr, self.child_idx = g.parent, g.child_ptr, g.child_idx
        self.root = g.root
        self.conf = g.conf.copy()
        self.p_ind = np.asarray(p_ind, float).copy()
        self.p_syn = np.asarray(p_syn, float).copy()
        self.p_coord = np.asarray(p_coord, float).copy()
        self.bucket = np.asarray(bucket, int)
        W, pp, rp = compute_weights(g, self.conf, self.p_ind, self.p_syn, self.p_coord, cfg, parts=True)
        self.W, self.pp, self.root_pind = W, pp, rp
        self.B = int(self.bucket.max()) + 1
        self.A_b = np.zeros((self.B, self.C)); self.Z_b = np.zeros(self.B)
        np.add.at(self.A_b, self.bucket, W[:, None] * P)
        np.add.at(self.Z_b, self.bucket, W)
        self.A, self.Z = self.A_b.sum(0), float(self.Z_b.sum())
        self.version = 0
        self.log: list[AuditEntry] = []

    # -- reporting -----------------------------------------------------------------------------------------------
    def organic(self):
        return None if self.Z <= 0 else self.A / self.Z

    def organic_by_bucket(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.A_b / self.Z_b[:, None]

    def snapshot_arrays(self) -> dict:
        return dict(conf=self.conf.copy(), p_ind=self.p_ind.copy(), p_syn=self.p_syn.copy(),
                    p_coord=self.p_coord.copy(), W=self.W.copy())

    def aggregate_at_version(self, k: int):
        """Reconstruct (A, Z) as published at version k by undoing the logged deltas."""
        A, Z = self.A.copy(), self.Z
        for e in reversed(self.log):
            if e.version > k:
                A -= e.dA; Z -= e.dZ
        return A, Z
