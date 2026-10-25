"""Competing aggregation strategies (all return a weight vector over records)."""
from __future__ import annotations
import numpy as np
from .pipeline import Analysis
from . import weighting as w


def methods(a: Analysis, dedup_thr: float = 0.90, syn_thr: float = 0.5, oracle: Analysis | None = None,
            tuned_dedup: float | None = None) -> dict:
    c, g, cfg = a.corpus, a.graph, a.cfg
    n = c.n
    out = {
        "Raw aggregation": np.ones(n),
        "Deduplication (tau=0.90)": (g.max_sim_prev < dedup_thr).astype(float),
        "Synthetic filtering": (c.p_syn <= syn_thr).astype(float),
        "Message-level weighting": w.weights_message_level(c.p_syn),
        "Tree-normalised (1/size)": w.weights_tree_normalised(g),
        "Proposed (inferred lineage)": a.W,
    }
    if tuned_dedup is not None:
        out["Deduplication (tuned tau)"] = (g.max_sim_prev < tuned_dedup).astype(float)
    if oracle is not None:
        out["Proposed (oracle lineage)"] = oracle.W
    return out


def tune_dedup_threshold(a: Analysis, grid=np.arange(0.70, 0.99, 0.02)) -> float:
    """Give the dedup baseline its best possible threshold (selected against ground truth => optimistic baseline)."""
    from . import aggregate as agg
    truth = a.corpus.truth_distribution()
    errs = [agg.tvd(truth, agg.organic(a.P, (a.graph.max_sim_prev < t).astype(float))) for t in grid]
    return float(grid[int(np.argmin(errs))])
