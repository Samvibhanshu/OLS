"""Tests of the lineage-aware RAG layer."""
import numpy as np, pytest
from opinion_lineage import *
from opinion_lineage.reference import dense_replay
from opinion_lineage.rag import *
from opinion_lineage.rag import strategies as st
from conftest import random_events


@pytest.fixture(scope="module")
def rag(models):
    c = generate_corpus(SimConfig(n=3000, contamination=0.45, seed=9))
    return LineageRAG(c, models, K=200, m=8)


def test_renderer_is_deterministic_and_does_not_touch_simulator():
    c1 = generate_corpus(SimConfig(n=800, contamination=0.4, seed=2)); c2 = generate_corpus(SimConfig(n=800, contamination=0.4, seed=2))
    assert np.array_equal(c1.X, c2.X) and render_texts(c1, 0) == render_texts(c2, 0)
    assert len(render_texts(c1, 0)) == c1.n and all(t for t in render_texts(c1, 0))


def test_reposts_quote_their_parent_text(rag):
    from opinion_lineage.rag.textgen import _base
    c, tx = rag.corpus, rag.texts
    rp = np.flatnonzero((c.kind == 1) & (c.true_parent >= 0))
    assert len(rp) > 50 and all(_base(tx[c.true_parent[i]]) in tx[i] for i in rp)


def test_retrieval_is_on_topic(rag):
    c = rag.corpus
    prec = []
    for tp in range(len(TOPICS)):
        ids, _ = rag.retrieve(make_query(tp), 100)
        prec.append(np.mean(c.topic[ids] == tp))
    assert np.mean(prec) > 0.95


def test_lineage_statistic_matches_formula_and_is_a_distribution(rag):
    a = rag.ask(make_query(1), "lineage")
    W = rag.store.W[a.evidence]; P = rag.store.P[a.evidence]
    assert np.allclose(a.S, (W[:, None] * P).sum(0) / W.sum()) and abs(a.S.sum() - 1) < 1e-6
    assert 0 <= a.amplification_share <= 1 and a.n_lineages <= a.raw_n


@pytest.mark.parametrize("strategy", ["naive_topm", "naive_all", "mmr", "dedup", "lineage"])
def test_every_answer_is_grounded(rag, strategy):
    a = rag.ask(make_query(2), strategy)
    v = verify_answer(a, rag.store.P, rag.texts)
    assert v["ok"], v
    assert len(a.cited) <= rag.m and a.text


def test_lineage_context_has_one_slot_per_lineage(rag):
    for tp in range(len(TOPICS)):
        a = rag.ask(make_query(tp), "lineage")
        roots = [rag.store.root[c["id"]] for c in a.cited]
        assert len(roots) == len(set(roots))


def test_naive_context_is_crowded_by_copies_where_lineage_context_is_not(rag):
    c = rag.corpus
    naive_roots, lin_roots = [], []
    for tp in range(len(TOPICS)):
        for strat, store in (("naive_topm", naive_roots), ("lineage", lin_roots)):
            a = rag.ask(make_query(tp), strat)
            store.append(len({int(rag.store.root[x["id"]]) for x in a.cited}) / max(len(a.cited), 1))
    assert np.mean(lin_roots) == 1.0 and np.mean(naive_roots) < 0.9


def test_retraction_updates_answers_in_place_without_models_or_reindex(rag):
    q = make_query(3); a0 = rag.ask(q, "lineage")
    roots = [i for i in a0.evidence if rag.store.parent[i] < 0][:25]
    ev = [RevisionEvent(ROOT_INDEPENDENCE, int(r), 0.0) for r in roots]
    snap = rag.store.snapshot_arrays(); M_before = rag.index.M; fp = rag.models.forward_passes
    reps = rag.retract(ev)
    assert rag.models.forward_passes == fp and rag.index.M is M_before            # nothing re-encoded or re-indexed
    assert rag.is_stale(a0)
    a1 = rag.ask(q, "lineage")
    assert not rag.is_stale(a1) and a1.effective_n < a0.effective_n
    dense = dense_replay(rag.analysis.graph, rag.store.P, rag.corpus.bucket, snap, ev, rag.cfg)
    assert np.abs(dense["W"] - rag.store.W).max() < 1e-12                             # equals dense recomputation
    assert verify_answer(a1, rag.store.P, rag.texts)["ok"]


def test_answer_abstains_when_independent_mass_collapses(models):
    c = generate_corpus(SimConfig(n=1500, contamination=0.3, seed=4)); r = LineageRAG(c, models, K=60, m=5)
    a0 = r.ask(make_query(0), "lineage")
    ids, _ = r.retrieve(make_query(0), 60)
    r.retract([RevisionEvent(ROOT_INDEPENDENCE, int(x), 0.0) for x in np.unique(r.store.root[ids])])
    a1 = r.ask(make_query(0), "lineage")
    assert a1.effective_n < 0.25 * a0.effective_n                      # retraction is partial where edges are uncertain (q < 1)
    a2 = r.ask(make_query(0), "lineage", min_mass=a0.effective_n * 0.5)  # an operator-set evidence floor makes it abstain
    assert a2.abstained and "cannot give" in a2.text and not a2.cited
    assert not np.isnan(a0.S).any()


def test_llm_generator_rejects_hallucinated_citations(rag):
    a = rag.ask(make_query(4), "lineage", generate=False)
    ok_id = a.cited[0]["id"]
    good = LLMGenerator(lambda p: f"Mostly mixed, organically [{ok_id}].")
    bad = LLMGenerator(lambda p: "Everyone hates it [999999].")
    nocite = LLMGenerator(lambda p: "Everyone hates it.")
    assert good(a, True).startswith("Mostly mixed") and good.rejected == 0
    assert "999999" not in bad(a, True) and bad.rejected == 1
    assert nocite(a, True).startswith("Organic sentiment") and nocite.rejected == 1
    p = build_prompt(a)
    assert "ONLY the evidence" in p and f"[{ok_id}]" in p and "weight=" in p and a.query in p


def test_precise_invalidation_flags_only_affected_answers(models):
    c = generate_corpus(SimConfig(n=2500, contamination=0.4, seed=12)); r = LineageRAG(c, models, K=150, m=6)
    answers = [r.ask(make_query(tp), "lineage", generate=False) for tp in range(len(TOPICS))]
    victim = next(i for i in answers[0].evidence if r.store.parent[i] < 0)
    r.retract(RevisionEvent(ROOT_INDEPENDENCE, int(victim), 0.0))
    hit = r.invalidated(answers)
    assert answers[0] in hit and len(hit) < len(answers)                # topic 0 is stale; other topics are provably untouched
    expected = [a for a in answers if np.isin(a.evidence, r.store.log[-1].affected_ids).any()]
    assert len(hit) == len(expected)
