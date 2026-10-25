"""Controlled synthetic social-media stream with a known ground truth.

Record types: organic independent roots, organic cascades (repost / quote / paraphrase, preferential
attachment), coordinated campaigns (seed + bot copies, AI-generated paraphrases, time bursts) and optional
"sleeper" roots (independent-looking records from a compromised source that the provenance layer initially trusts).
Ground-truth organic sentiment = label distribution of the *organic* independent roots.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .config import (K_INDEPENDENT, K_REPOST, K_QUOTE, K_PARAPHRASE, K_AI, K_COORD)


@dataclass
class SimConfig:
    n: int = 6000
    contamination: float = 0.3          # fraction of records that are derivative / amplified
    d_raw: int = 128
    n_topics: int = 8
    n_classes: int = 3
    class_probs: tuple = (0.30, 0.30, 0.40)   # organic sentiment truth (neg, neu, pos)
    target_class: int = 0                     # class pushed by the campaigns
    campaign_share: float = 0.7               # share of derivatives produced by coordinated campaigns
    avg_campaign_size: float = 40.0
    explicit_meta_rate: float = 0.7           # P(platform metadata exposes the parent of a repost/quote)
    hard_negative_rate: float = 0.08          # organic roots sharing a "talking point" (sim up to ~0.8)
    sleeper_root_share: float = 0.0           # share of roots from a compromised feed (initially trusted)
    days: float = 30.0
    a_topic: float = 0.50
    a_sent: float = 0.08
    a_noise: float = 0.62
    seed: int = 0


@dataclass
class Corpus:
    X: np.ndarray                 # raw features (N,d)
    t: np.ndarray                 # timestamp (days), non-decreasing
    label: np.ndarray             # true sentiment class
    kind: np.ndarray              # ground-truth kind
    true_parent: np.ndarray       # -1 for roots
    meta_parent: np.ndarray       # platform-visible parent id or -1
    meta_type: np.ndarray         # 0 none, 1 repost, 2 quote
    source_cluster: np.ndarray    # account-cluster id (shared by coordinated accounts)
    p_syn: np.ndarray             # synthetic-content detector score
    p_ind: np.ndarray             # provenance-layer independence estimate
    organic_root: np.ndarray      # bool: truly independent human opinion
    campaign: np.ndarray          # campaign id or -1  (-2 = sleeper)
    bucket: np.ndarray            # day bucket
    cfg: SimConfig = field(repr=False, default=None)
    topic: np.ndarray = None      # ground-truth topic id (used by the RAG layer's text renderer and evaluation only)

    @property
    def n(self) -> int:
        return len(self.t)

    def truth_distribution(self) -> np.ndarray:
        m = self.organic_root
        return np.bincount(self.label[m], minlength=self.cfg.n_classes) / m.sum()

    def observed_labels_distribution(self) -> np.ndarray:
        return np.bincount(self.label, minlength=self.cfg.n_classes) / self.n


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def generate_corpus(cfg: SimConfig) -> Corpus:
    rng = np.random.default_rng(cfg.seed)
    n, C, d = cfg.n, cfg.n_classes, cfg.d_raw
    # fixed concept space (depends on a global seed so train/test corpora share the same semantics)
    crng = np.random.default_rng(777)
    topics = _unit(crng.normal(size=(cfg.n_topics, d)))
    sent_dirs = _unit(crng.normal(size=(C, d)))
    talking_pts = _unit(crng.normal(size=(cfg.n_topics, 4, d)))

    D = int(round(cfg.contamination * n))
    R = n - D
    n_camp = int(round(cfg.campaign_share * D))
    K = max(1, int(round(n_camp / cfg.avg_campaign_size))) if n_camp > 0 else 0
    n_sleep = int(round(cfg.sleeper_root_share * R))
    R_org = R - K - n_sleep
    assert R_org > 10, "too few organic roots"

    X = np.zeros((n, d), np.float32)
    t = np.zeros(n); label = np.zeros(n, int); kind = np.zeros(n, int)
    parent = -np.ones(n, int); meta_parent = -np.ones(n, int); meta_type = np.zeros(n, int)
    cluster = np.arange(n) + 10_000; camp = -np.ones(n, int)
    organic = np.zeros(n, bool); ptopic = np.zeros(n, int)
    is_bot = np.zeros(n, bool); is_syn = np.zeros(n, bool)

    def make_root(i, c, topic, hard=False):
        v = cfg.a_topic * topics[topic] + cfg.a_sent * sent_dirs[c] + cfg.a_noise * _unit(rng.normal(size=d))
        if hard:
            v = v + 0.50 * talking_pts[topic, rng.integers(4)]
        X[i] = _unit(v); label[i] = c; ptopic[i] = topic

    # --- organic roots
    for i in range(R_org):
        c = rng.choice(C, p=cfg.class_probs)
        make_root(i, c, rng.integers(cfg.n_topics), hard=rng.random() < cfg.hard_negative_rate)
        t[i] = rng.uniform(0, cfg.days); organic[i] = True
    # --- campaign seeds
    seeds = list(range(R_org, R_org + K))
    for k, i in enumerate(seeds):
        make_root(i, cfg.target_class, rng.integers(cfg.n_topics))
        t[i] = rng.uniform(0.1 * cfg.days, 0.8 * cfg.days); camp[i] = k
        is_bot[i] = True; is_syn[i] = rng.random() < 0.5
        cluster[i] = 50_000 + k
    # --- sleeper roots (compromised feed: look independent, push the target class)
    sleepers = list(range(R_org + K, R))
    for i in sleepers:
        c = cfg.target_class if rng.random() < 0.85 else rng.integers(C)
        make_root(i, c, rng.integers(cfg.n_topics), hard=rng.random() < cfg.hard_negative_rate)
        t[i] = rng.uniform(0, cfg.days); camp[i] = -2; is_bot[i] = True
        cluster[i] = 90_000 + rng.integers(5)

    # --- derivatives (Polya-urn preferential attachment)
    org_pool = list(range(R_org))
    org_pool += [i for i in range(R_org) if label[i] == cfg.target_class and rng.random() < 0.3]   # mild negativity bias
    org_pool += sleepers + sleepers      # sleeper posts look organic and get reposted / quoted by real users
    camp_pool = {k: [s] for k, s in enumerate(seeds)}
    n_org_deriv = D - n_camp
    kinds_org = rng.choice([K_REPOST, K_QUOTE, K_PARAPHRASE], size=n_org_deriv, p=[0.5, 0.2, 0.3])
    kinds_cmp = rng.choice([K_COORD, K_AI, K_REPOST], size=n_camp, p=[0.5, 0.35, 0.15]) if n_camp else []
    sim_par = {K_REPOST: (0.99, 0.006), K_QUOTE: (0.93, 0.03), K_PARAPHRASE: (0.88, 0.05),
               K_AI: (0.86, 0.06), K_COORD: (0.88, 0.05)}
    idx = R
    plan = [(k, False) for k in kinds_org] + [(k, True) for k in kinds_cmp]
    rng.shuffle(plan)
    for kd, from_campaign in plan:
        if from_campaign:
            kc = int(rng.integers(len(seeds))); pool = camp_pool[kc]
            p = pool[rng.integers(len(pool))]
            camp[idx] = kc; cluster[idx] = 50_000 + kc if rng.random() < 0.8 else idx + 10_000
            is_bot[idx] = True; is_syn[idx] = (kd == K_AI) or rng.random() < 0.25
            delay = rng.exponential(0.03)
            lab = cfg.target_class
        else:
            p = org_pool[rng.integers(len(org_pool))]
            delay = rng.exponential(0.30)
            lab = label[p] if rng.random() < 0.97 else rng.integers(C)
        m, s = sim_par[kd]
        sim = float(np.clip(rng.normal(m, s), 0.55, 0.999))
        noise = _unit(rng.normal(size=d))
        X[idx] = _unit(sim * X[p] + np.sqrt(1 - sim ** 2) * noise)
        t[idx] = min(t[p] + delay, cfg.days * 1.3); label[idx] = lab; kind[idx] = kd
        parent[idx] = p; ptopic[idx] = ptopic[p]
        if kd in (K_REPOST, K_QUOTE) and rng.random() < cfg.explicit_meta_rate:
            meta_parent[idx] = p; meta_type[idx] = 1 if kd == K_REPOST else 2
        if kd == K_AI:
            is_syn[idx] = True
        if from_campaign:
            camp_pool[kc].append(idx)
        else:
            org_pool.append(idx)
            org_pool.append(p)          # preferential attachment
        idx += 1
    assert idx == n

    # --- provenance-layer scores
    p_syn = np.where(is_syn, rng.beta(6, 2.5, n), rng.beta(1.2, 6, n))
    p_ind = np.where(organic, rng.beta(9, 1.5, n), rng.beta(2, 8, n))
    sl = np.zeros(n, bool); sl[sleepers] = True
    p_ind[sl] = rng.beta(8, 2, sl.sum())          # sleepers fool the provenance layer
    p_syn[sl] = rng.beta(1.2, 6, sl.sum())
    p_ind[seeds] = rng.beta(2.5, 6, len(seeds)) if len(seeds) else p_ind[seeds]

    # --- sort by time (parents precede children because child time > parent time) and re-index
    order = np.argsort(t, kind="stable")
    inv = np.empty(n, int); inv[order] = np.arange(n)
    def remap(arr):
        out = arr[order].copy(); m = out >= 0; out[m] = inv[out[m]]; return out
    tp = remap(parent); mp = remap(meta_parent)
    assert np.all((tp < np.arange(n)) | (tp < 0)), "parent must precede child"
    o = order
    return Corpus(X=X[o], t=t[o], label=label[o], kind=kind[o], true_parent=tp, meta_parent=mp,
                  meta_type=meta_type[o], source_cluster=cluster[o], p_syn=p_syn[o], p_ind=p_ind[o],
                  organic_root=organic[o], campaign=camp[o], bucket=np.floor(t[o]).astype(int), cfg=cfg, topic=ptopic[o])
