"""LineageRAG: retrieval-augmented answering about public opinion with lineage-aware weighting and in-place correction."""
from __future__ import annotations
import time
import numpy as np
from ..config import Config
from ..pipeline import analyse, Analysis
from ..store import OpinionStore
from ..rollback import RollbackEngine, RevisionEvent
from ..reference import full_reprocessing
from .textgen import render_texts
from .retrieval import TextIndex
from . import strategies as st
from .generate import Answer, ExtractiveGenerator


class LineageRAG:
    def __init__(self, corpus, models, cfg: Config = Config(), K: int = 300, m: int = 8, text_seed: int = 0, text_disguise: float = 0.0,
                 analysis: Analysis | None = None, generator=None):
        self.corpus, self.models, self.cfg, self.K, self.m = corpus, models, cfg, K, m
        self.texts = render_texts(corpus, text_seed, text_disguise)
        self.index = TextIndex(self.texts)
        self.analysis = analysis or analyse(corpus, models, cfg)
        a = self.analysis
        self.store = OpinionStore(a.graph, a.P, corpus.p_ind, corpus.p_syn, a.graph.p_coord, corpus.bucket, cfg)
        self.engine = RollbackEngine(self.store)
        self.generator = generator or ExtractiveGenerator()
        self._snap0 = self.store.snapshot_arrays()          # state at index time (for reference recomputation)
        self.events: list[RevisionEvent] = []

    # -- retrieval ---------------------------------------------------------------------------------------------------
    def retrieve(self, query: str, K: int | None = None):
        return self.index.search(query, K or self.K)

    def context(self, ids, rel, W=None, m=None) -> st.Ctx:
        s = self.store
        return st.Ctx(ids=ids, rel=rel, P=s.P, sim=self.index.pairwise, W=s.W if W is None else W, root=s.root, m=m or self.m, pind=s.p_ind)

    # -- answering ---------------------------------------------------------------------------------------------------
    def answer_from(self, query: str, res: st.Result, lineage: bool, generate: bool = True) -> Answer:
        s = self.store
        copies = {}
        if lineage and "group_of" in res.info:                  # copies/derivatives per cited lineage inside this result
            g = res.info["group_of"]; ids = res.evidence
            cnt = np.bincount(g)
            pos = {int(i): k for k, i in enumerate(ids)}
            copies = {int(i): int(cnt[g[pos[int(i)]]]) - 1 for i in res.cited}
        rel = dict(zip(res.evidence.tolist(), range(len(res.evidence))))
        cited = [dict(id=int(i), text=self.texts[int(i)], weight=float(s.W[int(i)]), copies=copies.get(int(i), 0),
                      relevance=float(1.0)) for i in res.cited]
        flags = []
        if lineage and not res.info.get("abstain"):
            share = res.info.get("top_lineage_share", 0)
            if share > 0.25:
                flags.append(f"{100 * share:.0f}% of the retrieved posts descend from a single lineage: possible coordinated or viral push.")
            sci = 0.5 * float(np.abs(res.S - res.S_raw).sum())
            if sci > 0.10:
                flags.append(f"the sentiment shifts by {100 * sci:.0f} points once amplification is removed (contamination index {sci:.2f}).")
        a = Answer(query, res.name, res.S, res.S_raw, res.evidence, res.weights.copy(), cited, res.n_lineages, s.version, flags)
        if generate:
            a.text = self.generator(a, lineage)
        return a

    def ask(self, query: str, strategy: str = "lineage", K: int | None = None, m: int | None = None, generate: bool = True,
            **kw) -> Answer:
        ids, rel = self.retrieve(query, K)
        res = st.STRATEGIES[strategy](self.context(ids, rel, m=m), **kw)
        return self.answer_from(query, res, lineage=(strategy == "lineage"), generate=generate)

    # -- correction --------------------------------------------------------------------------------------------------
    def retract(self, events):
        """Apply revision events in place: weights of the affected lineage change; nothing is re-embedded or re-indexed."""
        events = [events] if isinstance(events, RevisionEvent) else list(events)
        reps = self.engine.apply_all(events); self.events += events
        return reps

    def is_stale(self, a: Answer) -> bool:
        return a.version != self.store.version

    def invalidated(self, answers) -> list:
        """Precise cache invalidation: the cached answers whose evidence contains a record whose weight changed since the answer
        was computed (read from the audit log).  Answers not returned are provably unaffected and need no regeneration."""
        out = []
        for a in answers:
            touched = [e.affected_ids for e in self.store.log if e.version > a.version]
            if touched and np.isin(a.evidence, np.concatenate(touched)).any():
                out.append(a)
        return out

    def reference_rebuild(self):
        """What a conventional RAG stack must do after the same news: re-encode and re-classify every post, then rebuild the
        text index; the weights are then recomputed densely.  Returns timing and the resulting weights."""
        c = self.corpus
        t0 = time.perf_counter()
        before = self.models.forward_passes if hasattr(self.models, "forward_passes") else None
        out = full_reprocessing(self.models, c.X, self.analysis.graph, c.bucket, self._snap0, self.events, self.cfg)
        idx = TextIndex(self.texts)
        return dict(W=out["W"], P=out["P"], seconds=time.perf_counter() - t0, index_seconds=idx.build_seconds)
