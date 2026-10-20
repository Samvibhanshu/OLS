"""E1 contamination sweep, E2 weighting ablation, E3 lineage-inference quality."""
from common import *
import pandas as pd, time
from scipy import stats
from opinion_lineage import *
from opinion_lineage import aggregate as agg
from opinion_lineage import weighting as W
from opinion_lineage.baselines import methods, tune_dedup_threshold

style()
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = range(12)
N = 6000
models = train_models()

rows, qrows = [], []
t0 = time.time()
for rho in LEVELS:
    for seed in SEEDS:
        c = generate_corpus(SimConfig(n=N, contamination=rho, seed=100 + seed))
        a = analyse(c, models); ao = analyse(c, models, oracle=True)
        td = tune_dedup_threshold(a)
        truth = c.truth_distribution()
        clean_ref = agg.organic(a.P, c.organic_root.astype(float))      # same classifier, perfect provenance
        for name, w in methods(a, oracle=ao, tuned_dedup=td).items():
            S = agg.organic(a.P, w)
            rows.append(dict(rho=rho, seed=seed, method=name, err=agg.tvd(truth, S),
                             err_vs_clean=agg.tvd(clean_ref, S), ind_mass=float(w.sum())))
        qrows.append(dict(rho=rho, seed=seed, **lineage_quality(a),
                          SCI=a.SCI, true_amp_share=float((c.true_parent >= 0).mean())))
    print(f"rho={rho:.1f} done  ({time.time()-t0:.0f}s)")
df = pd.DataFrame(rows); dq = pd.DataFrame(qrows)
df.to_csv(RES / "e1_contamination_runs.csv", index=False); dq.to_csv(RES / "e3_lineage_quality_runs.csv", index=False)

# ---------- E1 table + figure
order = list(PAL.keys())
tab = df.groupby(["method", "rho"]).err.agg(["mean", "std"]).unstack("rho")
tab = tab.loc[[m for m in order if m in tab.index]]
summary = {m: {f"{int(r*100)}%": [float(tab.loc[m, ("mean", r)]), float(tab.loc[m, ("std", r)])] for r in LEVELS} for m in tab.index}
save_json("e1_contamination_summary.json", summary)
(RES / "e1_contamination_table.md").write_text(
    "| Method | " + " | ".join(f"{int(r*100)}%" for r in LEVELS) + " |\n|---|" + "---|" * len(LEVELS) + "\n" +
    "\n".join(f"| {m} | " + " | ".join(f"{tab.loc[m, ('mean', r)]:.4f} ± {tab.loc[m, ('std', r)]:.4f}" for r in LEVELS) + " |"
              for m in tab.index) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for k, (col, ttl) in enumerate([("err", "Reconstruction error vs ground-truth organic sentiment"),
                                ("err_vs_clean", "Contamination-only error (classifier bias removed)")]):
    g = df.groupby(["method", "rho"])[col].agg(["mean", "std"])
    for m in order:
        if m not in g.index.get_level_values(0): continue
        mu = g.loc[m]["mean"].values; sd = g.loc[m]["std"].values
        ls = "--" if "oracle" in m else "-"
        ax[k].plot(np.array(LEVELS) * 100, mu, ls, color=PAL[m], lw=2.2 if "Proposed" in m else 1.5, marker="o", ms=4, label=m)
        ax[k].fill_between(np.array(LEVELS) * 100, mu - sd, mu + sd, color=PAL[m], alpha=.12)
    ax[k].set_xlabel("Contamination (% derivative / amplified records)"); ax[k].set_ylabel("Total-variation distance"); ax[k].set_title(ttl)
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "e1_contamination_sweep.png"); plt.close()

# ---------- E3 lineage quality
gq = dq.groupby("rho").mean(numeric_only=True)
save_json("e3_lineage_quality.json", gq.to_dict())
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for col, lab in [("root_acc", "root identification acc."), ("derivative_precision", "derivative precision"),
                 ("derivative_recall", "derivative recall"), ("derivative_f1", "derivative F1"),
                 ("correct_lineage_acc", "correct-lineage acc.")]:
    ax[0].plot(gq.index * 100, gq[col], marker="o", label=lab)
ax[0].set_ylim(0.4, 1.02); ax[0].set_xlabel("Contamination (%)"); ax[0].set_title("Lineage inference quality"); ax[0].legend(fontsize=8)
ax[1].plot(gq.index * 100, gq.SCI, marker="o", color="#c8102e", label="measured SCI")
ax[1].plot(gq.index * 100, gq.true_amp_share, "--", color="#555", label="true share of derivative records")
ax[1].set_xlabel("Contamination (%)"); ax[1].set_title("Sentiment Contamination Index"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "e3_lineage_quality.png"); plt.close()

# ---------- E2 ablation at 40% contamination (20 fresh seeds, paired)
abl = []
for seed in range(20):
    c = generate_corpus(SimConfig(n=N, contamination=0.4, seed=500 + seed))
    a = analyse(c, models); ao = analyse(c, models, oracle=True)
    g, cfg = a.graph, a.cfg; truth = c.truth_distribution()
    coord_cfg = cfg.with_(gamma=0.3)
    variants = {
        "Full: lineage + independence + synthetic": a.W,
        "Lineage only": W.weights_lineage_only(g, g.conf, cfg),
        "Lineage + synthetic": W.weights_lineage_syn(g, g.conf, c.p_syn, cfg),
        "Lineage + synthetic + coordination (gamma=0.3)": W.compute_weights(g, g.conf, c.p_ind, c.p_syn, g.p_coord, coord_cfg),
        "Soft (harmonic) provenance weighting": W.weights_soft(g, g.conf, c.p_ind, cfg),
        "No lineage (message-level only)": W.weights_message_level(c.p_syn),
        "Oracle lineage (ground-truth parents)": ao.W,
    }
    for n_, w in variants.items():
        abl.append(dict(seed=seed, variant=n_, err=agg.tvd(truth, agg.organic(a.P, w))))
da = pd.DataFrame(abl); da.to_csv(RES / "e2_ablation_runs.csv", index=False)
piv = da.pivot(index="seed", columns="variant", values="err")
full = "Full: lineage + independence + synthetic"
res = {}
for v in piv.columns:
    d = piv[v] - piv[full]
    p = stats.wilcoxon(piv[v], piv[full]).pvalue if v != full else float("nan")
    res[v] = dict(mean=float(piv[v].mean()), std=float(piv[v].std()), delta_vs_full=float(d.mean()), wilcoxon_p=float(p))
save_json("e2_ablation.json", res)
fig, ax = plt.subplots(figsize=(8.5, 4))
o = sorted(res, key=lambda k: res[k]["mean"])
ax.barh(range(len(o)), [res[k]["mean"] for k in o], xerr=[res[k]["std"] for k in o],
        color=["#c8102e" if k == full else "#9aa0a6" for k in o], capsize=3)
ax.set_yticks(range(len(o))); ax.set_yticklabels(o, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("Reconstruction error at 40% contamination (mean ± sd, 20 seeds)"); ax.set_title("Weighting ablation")
plt.tight_layout(); plt.savefig(FIG / "e2_ablation.png"); plt.close()
print(json.dumps(res, indent=1)); print((RES / "e1_contamination_table.md").read_text()); print(gq.round(3))
