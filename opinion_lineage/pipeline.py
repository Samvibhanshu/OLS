"""End-to-end pipeline: raw records -> embeddings -> sentiment -> lineage graph -> weights -> organic sentiment."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import Config
from .models import Encoder, SentimentModel, ModelBundle
from .simulate import SimConfig, Corpus, generate_corpus
from .lineage import build_lineage, oracle_lineage, LineageGraph
from .weighting import compute_weights
from . import aggregate as agg


def train_models(sim_template: SimConfig | None = None, n_train: int = 8000, seed: int = 123) -> ModelBundle:
    sc = sim_template or SimConfig()
    tr = generate_corpus(SimConfig(**{**sc.__dict__, "n": n_train, "contamination": 0.0, "seed": seed,
                                      "sleeper_root_share": 0.0}))
    enc = Encoder(sc.d_raw)
    clf = SentimentModel().fit(enc(tr.X), tr.label)
    enc.calls = enc.rows = clf.calls = clf.rows = 0          # ledger starts at zero after training
    return ModelBundle(enc, clf)


@dataclass
class Analysis:
    corpus: Corpus
    E: np.ndarray
    P: np.ndarray
    graph: LineageGraph
    W: np.ndarray
    cfg: Config

    @property
    def S_obs(self): return agg.observed(self.P)
    @property
    def S_org(self): return agg.organic(self.P, self.W)
    @property
    def SCI(self): return agg.sci(self.S_obs, self.S_org)
    def error(self, W=None):
        S = self.S_org if W is None else agg.organic(self.P, W)
        return agg.tvd(self.corpus.truth_distribution(), S)


def analyse(corpus: Corpus, models: ModelBundle, cfg: Config = Config(), oracle: bool = False) -> Analysis:
    E = models.encoder(corpus.X)
    P = models.clf.predict_proba(E)
    if oracle:
        g = oracle_lineage(corpus.true_parent, corpus.kind, cfg=cfg, rng=np.random.default_rng(corpus.cfg.seed))
        g.p_coord = np.zeros(corpus.n)
    else:
        g = build_lineage(E, corpus.t, corpus.source_cluster, corpus.meta_parent, corpus.meta_type,
                          corpus.p_syn, cfg)
    W = compute_weights(g, g.conf, corpus.p_ind, corpus.p_syn, g.p_coord, cfg)
    return Analysis(corpus, E, P, g, W, cfg)


def lineage_quality(a: Analysis) -> dict:
    """Root-identification accuracy, derivative F1, parent accuracy, edge-type confusion (vs ground truth)."""
    c, g = a.corpus, a.graph
    true_der = c.true_parent >= 0; pred_der = g.parent >= 0
    tp = (true_der & pred_der).sum(); fp = (~true_der & pred_der).sum(); fn = (true_der & ~pred_der).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    root_acc = float(((~true_der) == (~pred_der)).mean())
    both = true_der & pred_der
    par_acc = float((g.parent[both] == c.true_parent[both]).mean()) if both.any() else float("nan")
    # same-lineage accuracy: predicted root equals true root (parent may differ inside a lineage)
    tr = np.arange(c.n)
    for lv in g.levels[:1]: pass
    true_root = tr.copy()
    for i in range(c.n):
        p = c.true_parent[i]
        if p >= 0: true_root[i] = true_root[p]
    lin_acc = float((g.root[true_der] == true_root[true_der]).mean()) if true_der.any() else float("nan")
    return dict(root_acc=root_acc, derivative_precision=float(prec), derivative_recall=float(rec),
                derivative_f1=float(f1), exact_parent_acc=par_acc, correct_lineage_acc=lin_acc)
