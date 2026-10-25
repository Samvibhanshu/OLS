"""Opinion-Lineage Graph construction (patent Step "Edge-Insertion Rule").

Candidate parents j of record i:  t_j < t_i  and (M_ij = 1  or  S_ij >= tau1).
Typed edges / confidences:
  1. explicit repost/quote metadata   -> REPOST_OF / QUOTES, c ~ explicit_conf
  2. S_ij >= tau2                      -> PARAPHRASES  (AI_DERIVATIVE_OF if P_syn,i > tau3), c increasing in S_ij
  3. coordination score C_ij > tau4    -> COORDINATED_WITH, c = C_ij
  4. tau1 <= S_ij < tau2, nothing else -> UNCERTAIN_DEPENDENCY with a reduced c
The highest-confidence candidate is the primary parent (operative derivation path).
The graph is built incrementally in time-ordered blocks (G_t = G_{t-1} U dG_t); blocks are matrix products, so
tens of thousands of records are handled without ANN indices.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import (Config, NONE, REPOST_OF, QUOTES, PARAPHRASES, AI_DERIVATIVE_OF, COORDINATED_WITH,
                     UNCERTAIN_DEPENDENCY, EDGE_NAMES)


@dataclass
class LineageGraph:
    parent: np.ndarray        # primary parent (-1 = independent root)
    kind: np.ndarray          # edge type of primary edge
    conf: np.ndarray          # derivation confidence c_e of primary edge (0 for roots)
    sim: np.ndarray           # similarity to primary parent
    p_coord: np.ndarray       # coordination evidence per record
    max_sim_prev: np.ndarray  # max similarity to any earlier record (used by the dedup baseline)
    n_cand: np.ndarray        # number of qualifying candidate parents (secondary edges kept for audit)

    def __post_init__(self):
        self._build_structure()

    @property
    def n(self) -> int:
        return len(self.parent)

    def _build_structure(self):
        n = self.n
        par = self.parent
        depth = np.zeros(n, np.int32)
        for i in range(n):                      # parents always precede children
            p = par[i]
            if p >= 0:
                depth[i] = depth[p] + 1
        self.depth = depth
        order = np.argsort(depth, kind="stable")
        bounds = np.searchsorted(depth[order], np.arange(depth.max() + 2))
        self.levels = [order[bounds[d]:bounds[d + 1]] for d in range(depth.max() + 1)]
        # children in CSR form
        has = par >= 0
        kids = np.flatnonzero(has)
        srt = np.argsort(par[kids], kind="stable")
        self.child_idx = kids[srt].astype(np.int64)
        counts = np.bincount(par[has], minlength=n)
        self.child_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        root = np.arange(n)
        for lv in self.levels[1:]:
            root[lv] = root[par[lv]]
        self.root = root

    def lineage_sizes(self) -> np.ndarray:
        """size of the subtree rooted at every node (including itself)"""
        size = np.ones(self.n, np.int64)
        for lv in reversed(self.levels[1:]):
            np.add.at(size, self.parent[lv], size[lv])
        return size

    def to_networkx(self, nodes=None):
        import networkx as nx
        G = nx.DiGraph()
        sel = range(self.n) if nodes is None else nodes
        sel = set(int(x) for x in sel)
        for i in sel:
            G.add_node(i)
        for i in sel:
            p = int(self.parent[i])
            if p >= 0 and p in sel:
                G.add_edge(p, i, kind=EDGE_NAMES[int(self.kind[i])], confidence=float(self.conf[i]))
        return G


def build_lineage(E: np.ndarray, t: np.ndarray, source_cluster: np.ndarray, meta_parent: np.ndarray,
                  meta_type: np.ndarray, p_syn: np.ndarray, cfg: Config, block: int = 1024) -> LineageGraph:
    n = len(E)
    assert np.all(np.diff(t) >= 0), "records must be time ordered"
    E = np.ascontiguousarray(E, dtype=np.float32)
    parent = -np.ones(n, np.int64); kind = np.zeros(n, np.int8); conf = np.zeros(n)
    sim_out = np.zeros(n); pcoord = np.zeros(n); maxsim = np.zeros(n); ncand = np.zeros(n, np.int32)

    for a in range(0, n, block):
        b = min(a + block, n)
        S = E[a:b] @ E[:b].T                                   # (blk, b)
        rows = np.arange(a, b)
        # causal mask: only j < i
        tri = np.arange(b)[None, :] >= rows[:, None]
        S[tri] = -1.0
        if a > 0 or b > a + 1:
            maxsim[a:b] = np.where(rows > 0, S.max(axis=1), 0.0)
        ri, cj = np.nonzero(S >= cfg.tau1)
        ri_g = ri + a
        sim = S[ri, cj].astype(np.float64)
        # explicit metadata candidates (may be below tau1 because the visible text differs from the parent)
        mm = np.flatnonzero(meta_parent[a:b] >= 0)
        if len(mm):
            gi = mm + a; gj = meta_parent[gi]
            have = set(zip(ri_g.tolist(), cj.tolist()))
            add = [(i, j) for i, j in zip(gi, gj) if (int(i), int(j)) not in have]
            if add:
                ai = np.array([x[0] for x in add]); aj = np.array([x[1] for x in add])
                ri_g = np.concatenate([ri_g, ai]); cj = np.concatenate([cj, aj])
                sim = np.concatenate([sim, np.einsum("ij,ij->i", E[ai], E[aj]).astype(np.float64)])
        if len(ri_g) == 0:
            continue
        gi, gj = ri_g, cj
        dt = np.maximum(t[gi] - t[gj], 0.0)
        same = (source_cluster[gi] == source_cluster[gj])
        Cij = sim * np.exp(-dt / cfg.coord_tau_days) * np.where(same, 1.0, cfg.coord_cross_cluster)
        elig = sim >= cfg.tau1
        # rule 4: provisional edge with reduced confidence
        c4 = np.where(elig, cfg.uncertain_max * np.clip((sim - cfg.tau1) / (1 - cfg.tau1), 0, 1), 0.0)
        # rule 2: strong semantic derivation
        c2 = np.where(sim >= cfg.tau2,
                      cfg.para_lo + (cfg.para_hi - cfg.para_lo) * np.clip((sim - cfg.tau2) / (1 - cfg.tau2), 0, 1), 0.0)
        ai = (sim >= cfg.tau2) & (p_syn[gi] > cfg.tau3)
        c2 = np.where(ai, np.minimum(0.99, c2 + 0.1 * p_syn[gi]), c2)
        # rule 3: coordination
        c3 = np.where(elig & (Cij > cfg.tau4), Cij, 0.0)
        # rule 1: explicit repost / quote metadata
        mex = (meta_parent[gi] == gj)
        c1 = np.where(mex, cfg.explicit_conf, 0.0)

        stack = np.vstack([c4, c2, c3, c1])                    # type priority resolved by confidence
        best = stack.argmax(axis=0); cbest = stack.max(axis=0)
        etype = np.select(
            [best == 3, best == 2, best == 1, best == 0],
            [np.where(meta_type[gi] == 2, QUOTES, REPOST_OF), COORDINATED_WITH,
             np.where(ai, AI_DERIVATIVE_OF, PARAPHRASES), UNCERTAIN_DEPENDENCY], NONE)
        ok = cbest > 0
        gi, gj, sim, cbest, etype, Cij = gi[ok], gj[ok], sim[ok], cbest[ok], etype[ok], Cij[ok]
        if len(gi) == 0:
            continue
        # coordination evidence per record: strongest C_ij among semantically eligible candidates
        np.maximum.at(pcoord, gi, np.where(sim >= cfg.tau1, np.clip(Cij, 0, 1), 0.0))
        np.add.at(ncand, gi, 1)
        order = np.lexsort((-cbest, gi))                       # per record: highest confidence first
        gi_s = gi[order]
        first = np.concatenate([[True], gi_s[1:] != gi_s[:-1]])
        sel = order[first]
        parent[gi[sel]] = gj[sel]; kind[gi[sel]] = etype[sel]; conf[gi[sel]] = cbest[sel]; sim_out[gi[sel]] = sim[sel]
    return LineageGraph(parent=parent, kind=kind, conf=conf, sim=sim_out, p_coord=pcoord,
                        max_sim_prev=maxsim, n_cand=ncand)


def oracle_lineage(true_parent: np.ndarray, true_kind: np.ndarray, p_coord: np.ndarray | None = None,
                   cfg: Config = Config(), rng: np.random.Generator | None = None) -> LineageGraph:
    """Ground-truth parents with kind-dependent confidences (used for the oracle upper bound and for very large
    scaling corpora where O(N^2) similarity search would be wasteful)."""
    rng = rng or np.random.default_rng(0)
    n = len(true_parent)
    conf = np.zeros(n)
    m = true_parent >= 0
    base = {1: 0.98, 2: 0.97, 3: 0.88, 4: 0.92, 5: 0.85}
    for k, v in base.items():
        s = m & (true_kind == k)
        conf[s] = np.clip(rng.normal(v, 0.04, s.sum()), 0.5, 0.99)
    kmap = {1: REPOST_OF, 2: QUOTES, 3: PARAPHRASES, 4: AI_DERIVATIVE_OF, 5: COORDINATED_WITH}
    kind = np.zeros(n, np.int8)
    for k, v in kmap.items():
        kind[m & (true_kind == k)] = v
    pc = np.zeros(n) if p_coord is None else p_coord
    return LineageGraph(parent=true_parent.astype(np.int64), kind=kind, conf=conf, sim=np.zeros(n),
                        p_coord=pc, max_sim_prev=np.zeros(n), n_cand=m.astype(np.int32))
