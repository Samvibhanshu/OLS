"""Selective Dependency Rollback Engine (patent Step 8).

For a revision event on record v:
  1. D(v) = v + transitive descendants under primary-parent edges (BFS over a CSR child index)
  2. for u in D(v) ONLY: read the witness, q_{v,u} = prod of edge confidences on path(v,u), q_{v,v}=1
  3. W~_u = counterfactual weight under the fully revised provenance state (same weight function / version)
  4. W'_u = W_u + q_{v,u} (W~_u - W_u)
  5. dA_c = sum_u (W'_u - W_u) P_u(c),  dZ = sum_u (W'_u - W_u);  A' = A + dA,  Z' = Z + dZ
Cost O(|D(v)|).  Neither the encoder nor the sentiment classifier is invoked; P is accessed by reference only.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .store import OpinionStore, AuditEntry

ROOT_INDEPENDENCE = "ROOT_INDEPENDENCE"      # revised independence P_ind of a lineage root
EDGE_CONFIDENCE = "EDGE_CONFIDENCE"          # revised derivation confidence of the primary edge into v
RECORD_SYNTHETIC = "RECORD_SYNTHETIC"        # revised synthetic-content probability of v
RECORD_COORDINATION = "RECORD_COORDINATION"  # revised coordination evidence of v


@dataclass(frozen=True)
class RevisionEvent:
    kind: str
    record: int
    value: float


@dataclass
class RollbackReport:
    event: RevisionEvent
    records_touched: int
    dA: np.ndarray
    dZ: float
    seconds: float
    version: int


class RollbackEngine:
    def __init__(self, store: OpinionStore):
        self.s = store

    def descendants(self, v: int):
        """BFS order of D(v) with q_{v,u} computed on the way down (parents precede children)."""
        s = self.s
        nodes, q, par_pos = [v], [1.0], [-1]
        k = 0
        while k < len(nodes):
            u = nodes[k]
            lo, hi = s.child_ptr[u], s.child_ptr[u + 1]
            if hi > lo:
                for kid in s.child_idx[lo:hi].tolist():
                    nodes.append(kid); q.append(q[k] * s.conf[kid]); par_pos.append(k)
            k += 1
        return np.array(nodes, np.int64), np.array(q), np.array(par_pos, np.int64)

    def apply(self, ev: RevisionEvent) -> RollbackReport:
        s, cfg = self.s, self.s.cfg
        t0 = time.perf_counter()
        v = ev.record
        # revise the stored provenance state of v
        if ev.kind == ROOT_INDEPENDENCE:
            if s.parent[v] >= 0:
                raise ValueError("ROOT_INDEPENDENCE revises a lineage root")
            s.p_ind[v] = ev.value
        elif ev.kind == EDGE_CONFIDENCE:
            if s.parent[v] < 0:
                raise ValueError("EDGE_CONFIDENCE needs a non-root record")
            s.conf[v] = ev.value
        elif ev.kind == RECORD_SYNTHETIC:
            s.p_syn[v] = ev.value
        elif ev.kind == RECORD_COORDINATION:
            s.p_coord[v] = ev.value
        else:
            raise ValueError(ev.kind)

        ids, q, par_pos = self.descendants(v)
        # refresh denormalised root independence / path factors inside D(v) only
        if ev.kind == ROOT_INDEPENDENCE:
            s.root_pind[ids] = ev.value
        elif ev.kind == EDGE_CONFIDENCE:
            pp = np.empty(len(ids))
            pp[0] = s.pp[s.parent[v]] * (1.0 - cfg.alpha * s.conf[v])      # single boundary read: parent's path factor
            fac = 1.0 - cfg.alpha * s.conf[ids]
            for k in range(1, len(ids)):                                   # parents precede children in BFS order
                pp[k] = pp[par_pos[k]] * fac[k]
            s.pp[ids] = pp
        # counterfactual (fully revised) weights for u in D(v), same weight function & model version
        W_tilde = s.root_pind[ids] * s.pp[ids] * (1.0 - cfg.beta * s.p_syn[ids]) * (1.0 - cfg.gamma * s.p_coord[ids])
        W_old = s.W[ids]
        W_new = W_old + q * (W_tilde - W_old)
        dW = W_new - W_old
        contrib = dW[:, None] * s.P[ids]
        dA, dZ = contrib.sum(axis=0), float(dW.sum())
        b = s.bucket[ids]
        np.add.at(s.A_b, b, contrib); np.add.at(s.Z_b, b, dW)
        s.A += dA; s.Z += dZ
        s.W[ids] = W_new
        s.version += 1
        s.log.append(AuditEntry(s.version, ev, ids, W_old, W_new, dA, dZ, cfg.model_version))
        return RollbackReport(ev, len(ids), dA, dZ, time.perf_counter() - t0, s.version)

    def apply_all(self, events):
        return [self.apply(e) for e in events]
