"""E6: retroactive correction of already-published daily indices.
A compromised feed ("sleeper" roots) looks independent and is initially trusted.  On day 20 it is exposed; the
provenance revision is applied by Selective Dependency Rollback and every historical day's index is corrected."""
from common import *
import time
from opinion_lineage import *
from opinion_lineage import aggregate as agg
from opinion_lineage.reference import dense_replay

style()
sc = SimConfig(n=20_000, contamination=0.3, sleeper_root_share=0.2, seed=31, days=30)
models = train_models(sc)
c = generate_corpus(sc)
a = analyse(c, models)
g = a.graph
st = OpinionStore(g, a.P, c.p_ind, c.p_syn, g.p_coord, c.bucket, a.cfg)
snap = st.snapshot_arrays()
TARGET = sc.target_class
B = st.B

def daily(Pw, W):
    num = np.zeros(B); den = np.zeros(B)
    np.add.at(num, c.bucket, W * Pw[:, TARGET]); np.add.at(den, c.bucket, W)
    return num / np.maximum(den, 1e-12)

days = np.arange(B)
cnt = np.bincount(c.bucket, minlength=B)
show = cnt >= 200           # drop the sparse tail bucket(s)
raw = daily(a.P, np.ones(c.n))
before = st.organic_by_bucket()[:, TARGET].copy()
truth_day = np.array([np.mean(c.label[(c.bucket == d) & c.organic_root] == TARGET) if ((c.bucket == d) & c.organic_root).any() else np.nan for d in days])
truth_all = c.truth_distribution()[TARGET]

# provenance monitor exposes the compromised feed: every sleeper root's independence is revised to 0.03
sleepers = np.flatnonzero((c.campaign == -2) & (g.parent < 0))
print("sleeper roots detected:", len(sleepers), " (share of all roots %.1f%%)" % (100 * len(sleepers) / (g.parent < 0).sum()))
events = [RevisionEvent(ROOT_INDEPENDENCE, int(r), 0.03) for r in sleepers]
S_before = st.organic().copy()
m0 = models.snapshot()
reps = RollbackEngine(st).apply_all(events)
touched = sum(r.records_touched for r in reps)
after = st.organic_by_bucket()[:, TARGET]
ref = dense_replay(g, a.P, c.bucket, snap, events, a.cfg)
S_after = st.organic()
info = dict(sleeper_roots=len(sleepers), records_touched=int(touched), pct_touched=100 * touched / c.n,
            model_rows=[int(x) for x in np.subtract(models.snapshot(), m0)],
            max_abs_diff_vs_full=float(np.abs(S_after - ref["A"] / ref["Z"]).max()),
            truth=c.truth_distribution(), observed=a.S_obs, organic_before=S_before, organic_after=S_after,
            err_raw=agg.tvd(c.truth_distribution(), a.S_obs), err_before=agg.tvd(c.truth_distribution(), S_before),
            err_after=agg.tvd(c.truth_distribution(), S_after),
            sci_before=agg.sci(a.S_obs, S_before), sci_after=agg.sci(a.S_obs, S_after), versions=st.version)
save_json("e6_retroactive.json", info); print(json.dumps(info, indent=1, default=lambda o: o.tolist() if hasattr(o,'tolist') else o))

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.1), gridspec_kw={"width_ratios": [2.3, 1]})
ax[0].plot(days[show], raw[:B][show], color="#9aa0a6", lw=1.6, label="Observed (raw)")
ax[0].plot(days[show], before[show], color="#d98e04", lw=2, label="Published organic index (before revision)")
ax[0].plot(days[show], after[show], color="#c8102e", lw=2.4, label="After Selective Dependency Rollback")
ax[0].axhline(truth_all, color="#222", ls=":", lw=1.4, label="Ground-truth organic share")
ax[0].set_xlabel("day"); ax[0].set_ylabel("share of negative sentiment")
ax[0].set_title("Historical index corrected retroactively"); ax[0].legend(fontsize=8, loc="upper right")
labs = ["Observed", "Before\nrevision", "After\nrollback"]
errs = [info["err_raw"], info["err_before"], info["err_after"]]
ax[1].bar(labs, errs, color=["#9aa0a6", "#d98e04", "#c8102e"])
for i, e in enumerate(errs): ax[1].text(i, e + .002, f"{e:.3f}", ha="center", fontsize=9)
ax[1].set_ylabel("TVD to ground truth"); ax[1].set_title(f"Corpus-level error\n({info['pct_touched']:.1f}% of records touched, 0 model calls)")
plt.tight_layout(); plt.savefig(FIG / "e6_retroactive_correction.png"); plt.close()
