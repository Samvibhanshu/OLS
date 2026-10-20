"""E4: Selective Dependency Rollback vs full reprocessing on a 50,000-record corpus (10k roots / 40k derivatives)."""
from common import *
import time, pandas as pd
from opinion_lineage import *
from opinion_lineage import aggregate as agg
from opinion_lineage.reference import dense_replay, full_reprocessing

style()
N = 50_000
sc = SimConfig(n=N, contamination=0.8, campaign_share=0.2, seed=2024)
t0 = time.time()
models = train_models(sc)
c = generate_corpus(sc)
print(f"corpus: {c.n} records, {(c.true_parent<0).sum()} roots, {(c.true_parent>=0).sum()} derivatives  ({time.time()-t0:.0f}s)")
t0 = time.time()
a = analyse(c, models)
print(f"initial pipeline incl. O(N^2) lineage inference: {time.time()-t0:.0f}s ; inferred roots={(a.graph.parent<0).sum()}")
g = a.graph
sizes = g.lineage_sizes()
roots = np.flatnonzero(g.parent < 0)
rng = np.random.default_rng(7)


def run_scenario(name, events):
    st = OpinionStore(g, a.P, c.p_ind, c.p_syn, g.p_coord, c.bucket, a.cfg)
    snap = st.snapshot_arrays()
    S_before = st.organic().copy()
    eng = RollbackEngine(st)
    m0 = models.snapshot()
    t = time.perf_counter(); reps = eng.apply_all(events); t_sel = time.perf_counter() - t
    sel_models = np.subtract(models.snapshot(), m0)
    touched = int(sum(r.records_touched for r in reps))
    # dense re-summation from stored witnesses (no model calls)
    t = time.perf_counter(); ref = dense_replay(g, a.P, c.bucket, snap, events, a.cfg); t_dense = time.perf_counter() - t
    # full reprocessing: every record through encoder + classifier again, then dense re-summation
    m1 = models.snapshot()
    t = time.perf_counter(); fr = full_reprocessing(models, c.X, g, c.bucket, snap, events, a.cfg); t_full = time.perf_counter() - t
    full_models = np.subtract(models.snapshot(), m1)
    S_sel, S_full = st.organic(), fr["A"] / fr["Z"]
    out = dict(
        scenario=name, events=len(events), corpus_records=N, records_touched=touched, pct_corpus_touched=100 * touched / N,
        selective_model_rows=[int(x) for x in sel_models], full_model_rows=[int(x) for x in full_models],
        t_selective_s=t_sel, t_dense_resum_s=t_dense, t_full_reprocessing_s=t_full,
        speedup_vs_full_reprocessing=t_full / t_sel, speedup_vs_dense_resum=t_dense / t_sel,
        max_abs_diff_distribution=float(np.abs(S_sel - S_full).max()),
        max_abs_diff_weights=float(np.abs(st.W - ref["W"]).max()),
        max_abs_diff_bucket_aggregate=float(np.abs(st.A_b - ref["A_b"]).max()),
        organic_before=S_before, organic_after=S_sel, shift_tvd=agg.tvd(S_before, S_sel),
        audit_versions=len(st.log))
    print(f"[{name}] events={len(events)} touched={touched} ({out['pct_corpus_touched']:.2f}%) | model rows sel={out['selective_model_rows']} "
          f"full={out['full_model_rows']} | t sel={t_sel*1e3:.1f}ms dense={t_dense*1e3:.0f}ms full={t_full*1e3:.0f}ms | "
          f"max|diff|={out['max_abs_diff_distribution']:.2e}")
    return out, st


# A) patent-style: 200 roots deemed invalid / unreliable
evA = [RevisionEvent(ROOT_INDEPENDENCE, int(r), 0.0) for r in rng.choice(roots, 200, replace=False)]
# B) mixed revision stream (root independence, edge confidence, synthetic probability)
non = np.flatnonzero(g.parent >= 0)
evB = ([RevisionEvent(ROOT_INDEPENDENCE, int(r), float(rng.uniform(0, .5))) for r in rng.choice(roots, 150, replace=False)] +
       [RevisionEvent(EDGE_CONFIDENCE, int(r), float(rng.uniform(.1, 1))) for r in rng.choice(non, 150, replace=False)] +
       [RevisionEvent(RECORD_SYNTHETIC, int(r), float(rng.uniform())) for r in rng.choice(N, 100, replace=False)])
rng.shuffle(evB)
# C) coordinated-campaign takedown: invalidate the 40 largest lineages (large affected region)
big = roots[np.argsort(-sizes[roots])[:40]]
evC = [RevisionEvent(ROOT_INDEPENDENCE, int(r), 0.0) for r in big]

results = []
for nm, ev in [("A: 200 invalid roots (patent scenario)", evA), ("B: mixed revision stream (400 events)", evB),
               ("C: takedown of the 40 largest lineages", evC)]:
    out, st = run_scenario(nm, ev)
    results.append(out)
save_json("e4_rollback.json", results)

# ---- figure
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
lab = ["A", "B", "C"]
pct = [r["pct_corpus_touched"] for r in results]
ax[0].bar(np.arange(3) - .2, [100] * 3, .4, color="#9aa0a6", label="Full reprocessing")
ax[0].bar(np.arange(3) + .2, pct, .4, color="#c8102e", label="Selective rollback")
for i, p in enumerate(pct): ax[0].text(i + .2, p + 2, f"{p:.1f}%", ha="center", fontsize=9)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(lab); ax[0].set_ylabel("% of corpus reconsidered"); ax[0].set_title("Records touched"); ax[0].legend(fontsize=8)
w = .27
ax[1].bar(np.arange(3) - w, [r["t_full_reprocessing_s"] * 1e3 for r in results], w, color="#9aa0a6", label="Full reprocessing (re-encode + re-classify)")
ax[1].bar(np.arange(3), [r["t_dense_resum_s"] * 1e3 for r in results], w, color="#4f86c6", label="Dense re-summation (stored witnesses)")
ax[1].bar(np.arange(3) + w, [r["t_selective_s"] * 1e3 for r in results], w, color="#c8102e", label="Selective rollback")
ax[1].set_yscale("log"); ax[1].set_ylim(5, 6000); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(lab); ax[1].set_ylabel("wall-clock (ms, log)")
ax[1].set_title("Correction time"); ax[1].legend(fontsize=7)
ax[2].bar(range(3), [max(r["max_abs_diff_distribution"], 1e-18) for r in results], color="#2a9d8f")
ax[2].set_yscale("log"); ax[2].set_ylim(1e-18, 1e-9); ax[2].set_xticks(range(3)); ax[2].set_xticklabels(lab)
ax[2].set_title("Max |selective − full recomputation|"); ax[2].set_ylabel("abs. difference in S_org")
ax[2].axhline(2.2e-16, color="#555", ls="--", lw=1); ax[2].text(2.4, 3e-16, "float64 eps", ha="right", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "e4_rollback.png"); plt.close()
