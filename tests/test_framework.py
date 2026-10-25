import numpy as np
import pytest
from opinion_lineage import *
from opinion_lineage import aggregate as agg
from opinion_lineage.config import Config
from opinion_lineage.lineage import LineageGraph
from opinion_lineage.reference import dense_replay, full_reprocessing
from opinion_lineage.weighting import compute_weights, propagation_confidence
from conftest import make_store, random_events


def chain_graph():
    # A -> B -> C -> D (lineage 1) and F -> G -> H (lineage 2), A=0.. D=3, F=4.. H=6
    parent = np.array([-1, 0, 1, 2, -1, 4, 5])
    conf = np.array([0, .9, .8, .5, 0, .7, .6])
    z = np.zeros(7)
    return LineageGraph(parent=parent, kind=np.zeros(7, np.int8), conf=conf, sim=z, p_coord=z.copy(),
                        max_sim_prev=z, n_cand=z.astype(np.int32))


def test_weight_formula_by_hand():
    g = chain_graph(); cfg = Config(alpha=1.0, beta=0.5, gamma=0.0)
    p_ind = np.array([.9, .1, .1, .1, .8, .1, .1]); p_syn = np.array([0, 0, 0, .4, 0, 0, 0.]); z = np.zeros(7)
    W = compute_weights(g, g.conf, p_ind, p_syn, z, cfg)
    assert W[0] == pytest.approx(.9)
    assert W[1] == pytest.approx(.9 * (1 - .9))
    assert W[3] == pytest.approx(.9 * (1 - .9) * (1 - .8) * (1 - .5) * (1 - .5 * .4))
    assert W[6] == pytest.approx(.8 * (1 - .7) * (1 - .6))


def test_propagation_confidence_chain():
    g = chain_graph()
    q = propagation_confidence(g, g.conf, 0)
    assert q[0] == 1 and q[1] == pytest.approx(.9) and q[2] == pytest.approx(.72) and q[3] == pytest.approx(.36)
    assert np.all(q[4:] == 0)       # unrelated lineage never touched


def test_lineage_is_causal_and_acyclic(analysis):
    g, t = analysis.graph, analysis.corpus.t
    m = g.parent >= 0
    assert np.all(g.parent[m] < np.flatnonzero(m))
    assert np.all(t[g.parent[m]] <= t[m])
    assert np.all((g.conf[m] > 0) & (g.conf[m] <= 1)) and np.all(g.conf[~m] == 0)


def test_explicit_metadata_edges_are_high_confidence(analysis):
    c, g = analysis.corpus, analysis.graph
    i = np.flatnonzero((c.meta_parent >= 0) & (g.parent == c.meta_parent))
    assert len(i) > 20 and np.all(g.conf[i] >= analysis.cfg.explicit_conf - 1e-9)


def test_aggregates(analysis):
    a = analysis
    assert agg.observed(a.P).sum() == pytest.approx(1)
    assert agg.organic(a.P, a.W).sum() == pytest.approx(1)
    assert agg.sci(a.S_obs, agg.organic(a.P, np.ones(len(a.W)))) == pytest.approx(0, abs=1e-12)
    m = agg.masses(a.W); assert m["independent_mass"] + m["amplification_mass"] == pytest.approx(m["n"])
    assert agg.organic(a.P, np.zeros(len(a.W))) is None          # Z = 0 -> no distribution reported


def test_selective_equals_full_recomputation(analysis):
    a = analysis; st = make_store(a); snap = st.snapshot_arrays()
    ev = random_events(a, 40, seed=1)
    RollbackEngine(st).apply_all(ev)
    ref = dense_replay(a.graph, a.P, a.corpus.bucket, snap, ev, a.cfg)
    assert np.abs(st.W - ref["W"]).max() < 1e-12
    assert np.abs(st.A - ref["A"]).max() < 1e-9 and abs(st.Z - ref["Z"]) < 1e-9
    assert np.abs(st.A_b - ref["A_b"]).max() < 1e-9 and np.abs(st.Z_b - ref["Z_b"]).max() < 1e-9


def test_equivalence_with_full_model_reprocessing(models, analysis):
    a = analysis; st = make_store(a); snap = st.snapshot_arrays()
    ev = random_events(a, 15, seed=2)
    RollbackEngine(st).apply_all(ev)
    ref = full_reprocessing(models, a.corpus.X, a.graph, a.corpus.bucket, snap, ev, a.cfg)
    assert np.abs(st.A - ref["A"]).max() < 1e-7 and abs(st.Z - ref["Z"]) < 1e-7


def test_zero_model_executions_during_rollback(models, analysis):
    a = analysis; st = make_store(a)
    before = models.snapshot()
    RollbackEngine(st).apply_all(random_events(a, 30, seed=3))
    assert models.snapshot() == before


def test_locality_cost_bound(analysis):
    a = analysis; st = make_store(a); g = a.graph
    sizes = g.lineage_sizes()
    v = int(np.flatnonzero((g.parent < 0) & (sizes > 3))[0])
    inside = np.zeros(g.n, bool)
    ids, _, _ = RollbackEngine(st).descendants(v); inside[ids] = True
    W0, pi0 = st.W.copy(), st.root_pind.copy()
    rep = RollbackEngine(st).apply(RevisionEvent(ROOT_INDEPENDENCE, v, 0.01))
    assert rep.records_touched == sizes[v] == inside.sum()
    assert np.array_equal(W0[~inside], st.W[~inside]) and np.array_equal(pi0[~inside], st.root_pind[~inside])


def test_partial_retraction_scales_with_path_confidence(analysis):
    a = analysis; st = make_store(a); g = a.graph
    sizes = g.lineage_sizes()
    v = int(np.flatnonzero((g.parent < 0) & (sizes > 4))[0])
    ids, q, _ = RollbackEngine(st).descendants(v)
    W0 = st.W[ids].copy()
    RollbackEngine(st).apply(RevisionEvent(ROOT_INDEPENDENCE, v, 0.0))
    # root fully retracted (q=1); descendants retracted by the fraction q of their (already small) weight
    assert st.W[v] == pytest.approx(0.0, abs=1e-12)
    for k in range(1, len(ids)):
        assert st.W[ids[k]] == pytest.approx(W0[k] * (1 - q[k]), rel=1e-9, abs=1e-12)


def test_audit_log_reconstructs_versions(analysis):
    a = analysis; st = make_store(a); A0, Z0 = st.A.copy(), st.Z
    RollbackEngine(st).apply_all(random_events(a, 10, seed=4))
    A, Z = st.aggregate_at_version(0)
    assert np.allclose(A, A0, atol=1e-9) and abs(Z - Z0) < 1e-9 and st.version == len(st.log) == 40


def test_invalid_events_rejected(analysis):
    st = make_store(analysis); g = analysis.graph
    root = int(np.flatnonzero(g.parent < 0)[0]); child = int(np.flatnonzero(g.parent >= 0)[0])
    with pytest.raises(ValueError): RollbackEngine(st).apply(RevisionEvent(ROOT_INDEPENDENCE, child, 0.1))
    with pytest.raises(ValueError): RollbackEngine(st).apply(RevisionEvent(EDGE_CONFIDENCE, root, 0.1))
