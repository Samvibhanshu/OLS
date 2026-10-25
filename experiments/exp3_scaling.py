"""E5: cost scaling.  (a) fixed number of revision events, growing corpus  (b) fixed corpus, growing affected lineage."""
from common import *
import time, pandas as pd
from opinion_lineage import *
from opinion_lineage.lineage import oracle_lineage
from opinion_lineage.reference import dense_replay, full_reprocessing

style()
models = train_models()


def build(n, seed, **kw):
    sc = SimConfig(n=n, contamination=0.8, campaign_share=kw.pop('campaign_share', 0.2), seed=seed, **kw)
    c = generate_corpus(sc)
    g = oracle_lineage(c.true_parent, c.kind, cfg=Config(), rng=np.random.default_rng(seed))
    E = models.encoder(c.X); P = models.clf.predict_proba(E)
    from opinion_lineage.weighting import compute_weights
    return c, g, P


def timeit(f, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); out = f(); best = min(best, time.perf_counter() - t)
    return best, out


rows = []
for n in [10_000, 25_000, 50_000, 100_000, 200_000]:
    c, g, P = build(n, 11)
    cfg = Config()
    rng = np.random.default_rng(3)
    roots = np.flatnonzero(g.parent < 0)
    ev = [RevisionEvent(ROOT_INDEPENDENCE, int(r), 0.0) for r in rng.choice(roots, 200, replace=False)]
    def sel():
        st = OpinionStore(g, P, c.p_ind, c.p_syn, g.p_coord, c.bucket, cfg)
        snap = st.snapshot_arrays()
        t = time.perf_counter(); reps = RollbackEngine(st).apply_all(ev); dt = time.perf_counter() - t
        return dt, snap, sum(r.records_touched for r in reps)
    dt_sel, snap, touched = min((sel() for _ in range(3)), key=lambda x: x[0])
    t_dense, _ = timeit(lambda: dense_replay(g, P, c.bucket, snap, ev, cfg), 2)
    t_full, _ = timeit(lambda: full_reprocessing(models, c.X, g, c.bucket, snap, ev, cfg), 2)
    rows.append(dict(N=n, touched=touched, t_selective=dt_sel, t_dense=t_dense, t_full=t_full))
    print(rows[-1])
df = pd.DataFrame(rows); df.to_csv(RES / "e5_scaling_corpus.csv", index=False)

# (b) fixed N, vary lineage size of a single revised root
c, g, P = build(100_000, 12, campaign_share=0.5, avg_campaign_size=2000.0)
sizes = g.lineage_sizes(); roots = np.flatnonzero(g.parent < 0)
cfg = Config(); st0 = OpinionStore(g, P, c.p_ind, c.p_syn, g.p_coord, c.bucket, cfg)
rows2 = []
targets = [1, 3, 10, 30, 100, 300, 600, 1000, 1500, 2000]
for tgt in targets:
    cand = roots[np.argsort(np.abs(sizes[roots] - tgt))[:5]]
    for r in cand[:1]:
        ts = []
        for _ in range(15):
            st = OpinionStore(g, P, c.p_ind, c.p_syn, g.p_coord, c.bucket, cfg)
            ts.append(RollbackEngine(st).apply(RevisionEvent(ROOT_INDEPENDENCE, int(r), 0.0)).seconds)
        rows2.append(dict(lineage_size=int(sizes[r]), t_selective=float(np.median(ts))))
d2 = pd.DataFrame(rows2).drop_duplicates("lineage_size"); d2.to_csv(RES / "e5_scaling_lineage.csv", index=False)
print(d2)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
ax[0].loglog(df.N, df.t_full * 1e3, "o-", color="#9aa0a6", label="Full reprocessing (re-encode + re-classify)")
ax[0].loglog(df.N, df.t_dense * 1e3, "s-", color="#4f86c6", label="Dense re-summation")
ax[0].loglog(df.N, df.t_selective * 1e3, "D-", color="#c8102e", lw=2, label="Selective rollback (200 events)")
ax[0].set_xlabel("Corpus size N (records)"); ax[0].set_ylabel("time (ms)"); ax[0].set_title("Cost vs corpus size"); ax[0].legend(fontsize=8)
ax[1].loglog(d2.lineage_size, d2.t_selective * 1e3, "D-", color="#c8102e", lw=2, label="measured (N = 100k)")
x = d2.lineage_size.values.astype(float); b, a0 = np.polyfit(x, d2.t_selective.values * 1e3, 1)
xs = np.logspace(0, np.log10(x.max()), 50); ax[1].loglog(xs, a0 + b * xs, "--", color="#555", label=f"linear fit: {a0*1e3:.0f} µs + {b*1e3:.2f} µs·|D|")
ax[1].set_xlabel("|D(v)| = size of affected lineage"); ax[1].set_ylabel("time (ms)"); ax[1].set_title("Cost vs affected-lineage size"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "e5_scaling.png"); plt.close()
