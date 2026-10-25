"""Ways of turning retrieved candidates into (a) an aggregate answer and (b) the evidence shown to the reader/LLM.

Every strategy sees the *same* candidate list (top-K by relevance).  Baselines use only text-visible information;
the lineage strategy additionally uses the Opinion-Lineage Graph weights W (maintained by the OpinionStore).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Ctx:
    ids: np.ndarray          # candidate record ids, ranked by relevance
    rel: np.ndarray          # relevance scores
    P: np.ndarray            # persisted sentiment distributions (N, C)
    sim: callable            # sim(ids) -> dense cosine matrix among ids
    W: np.ndarray            # live contribution weights (N,)
    root: np.ndarray         # lineage root per record
    m: int = 8               # evidence shown (the "context window")
    pind: np.ndarray = None  # provenance-layer independence per record (live)


@dataclass
class Result:
    name: str
    S: np.ndarray            # answer sentiment distribution
    S_raw: np.ndarray        # unweighted mean over all candidates (what a count would say)
    evidence: np.ndarray     # records the statistic is computed from
    weights: np.ndarray      # weight of each evidence record in the statistic
    cited: np.ndarray        # records shown as quotable evidence (<= m)
    n_lineages: int = 0
    info: dict = field(default_factory=dict)

    @property
    def effective_n(self) -> float:
        return float(self.weights.sum())


def _mean(P, ids, w=None):
    w = np.ones(len(ids)) if w is None else w
    return (w[:, None] * P[ids]).sum(0) / w.sum()


def naive_topm(c: Ctx) -> Result:
    ids = c.ids[:c.m]
    return Result("Naive RAG (top-m)", _mean(c.P, ids), c.P[c.ids].mean(0), ids, np.ones(len(ids)), ids)


def naive_all(c: Ctx) -> Result:
    return Result("Naive RAG (all K)", _mean(c.P, c.ids), c.P[c.ids].mean(0), c.ids, np.ones(len(c.ids)), c.ids[:c.m])


def mmr(c: Ctx, lam: float = 0.5) -> Result:
    S = c.sim(c.ids); r = (c.rel - c.rel.min()) / (np.ptp(c.rel) + 1e-12)
    sel = [0]; cand = list(range(1, len(c.ids)))
    while len(sel) < min(c.m, len(c.ids)):
        best = max(cand, key=lambda j: lam * r[j] - (1 - lam) * S[j, sel].max())
        sel.append(best); cand.remove(best)
    ids = c.ids[sel]
    return Result("MMR diversification (m)", _mean(c.P, ids), c.P[c.ids].mean(0), ids, np.ones(len(ids)), ids)


def text_dedup(c: Ctx, tau: float = 0.8, name: str | None = None) -> Result:
    S = c.sim(c.ids); leaders: list[int] = []
    for i in range(len(c.ids)):
        if not leaders or S[i, leaders].max() < tau:
            leaders.append(i)
    ids = c.ids[leaders]
    return Result(name or f"Text near-duplicate collapse (tau={tau:g})", _mean(c.P, ids), c.P[c.ids].mean(0), ids,
                  np.ones(len(ids)), ids[:c.m], n_lineages=len(ids), info=dict(tau=tau))


def dedup_provenance(c: Ctx, tau: float = 0.8) -> Result:
    """Strong non-lineage baseline: collapse text near-duplicates, then weight each survivor by its provenance-layer independence
    score P_ind (the same signal the lineage method starts from).  Isolates what the lineage *graph* adds."""
    S = c.sim(c.ids); leaders: list[int] = []
    for i in range(len(c.ids)):
        if not leaders or S[i, leaders].max() < tau:
            leaders.append(i)
    ids = c.ids[leaders]; w = c.pind[ids]
    if w.sum() <= 1e-12:
        return Result("Text dedup + provenance score", np.full(c.P.shape[1], np.nan), c.P[c.ids].mean(0), ids, w, ids[:0], len(ids), dict(abstain=True))
    order = np.argsort(-(c.rel[leaders] * w), kind="stable")
    cited = ids[order][:c.m]; cited = cited[c.pind[cited] > 0.05]
    return Result("Text dedup + provenance score", _mean(c.P, ids, w), c.P[c.ids].mean(0), ids, w, cited, len(ids))


def lineage_aware(c: Ctx, name: str = "Lineage-aware RAG (ours)", rerank: bool = True, weighted: bool = True,
                  min_mass: float = 1.0) -> Result:
    """S = sum_i W_i P_i / sum_i W_i over the candidates; evidence = one representative per lineage, ranked by
    relevance x weight (so ten reposts of one post occupy one slot, and a retracted source drops out).
    If the candidates carry less than `min_mass` effective independent opinions the answer abstains."""
    ids = c.ids; w = c.W[ids] if weighted else np.ones(len(ids))
    roots = c.root[ids]
    uniq, inv = np.unique(roots, return_inverse=True)
    best = {}
    for pos, g in enumerate(inv):                       # representative = highest-weight member of each lineage
        if g not in best or w[pos] > w[best[g]]:
            best[g] = pos
    reps = np.array(sorted(best.values()))
    score = c.rel[reps] * (w[reps] if rerank else 1.0)
    cited = ids[reps[np.argsort(-score, kind="stable")][:c.m]]
    cited = cited[c.W[cited] > 0.05]                    # never present a (nearly) retracted post as evidence
    if w.sum() < min_mass:                               # fewer than `min_mass` independent opinions left: abstain from a number
        return Result(name, np.full(c.P.shape[1], np.nan), c.P[ids].mean(0), ids, w, cited[:0], len(uniq), dict(abstain=True))
    lineage_mass = np.bincount(inv, weights=w)
    return Result(name, _mean(c.P, ids, w), c.P[ids].mean(0), ids, w, cited, len(uniq),
                  dict(top_lineage_share=float((np.bincount(inv).max()) / len(ids)),
                       lineage_mass=lineage_mass, group_of=inv, reps=ids[reps]))


STRATEGIES = {
    "naive_topm": naive_topm, "naive_all": naive_all, "mmr": mmr, "dedup": text_dedup, "dedup_prov": dedup_provenance,
    "lineage": lineage_aware,
}
