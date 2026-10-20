"""E7: threshold sensitivity (tau1 x tau2), weight-exponent sensitivity (alpha x beta) and validation-set tuning."""
from common import *
import itertools, pandas as pd
from opinion_lineage import *
from opinion_lineage import aggregate as agg
from opinion_lineage.weighting import compute_weights

style()
models = train_models()
N, RHO = 4000, 0.4

# ---- (1) tau1 x tau2 grid (lineage must be rebuilt for each setting)
T1 = [0.60, 0.65, 0.70, 0.75, 0.80]; T2 = [0.80, 0.85, 0.90, 0.95]
grid = np.full((len(T1), len(T2)), np.nan)
corp = [generate_corpus(SimConfig(n=N, contamination=RHO, seed=900 + s)) for s in range(4)]
for i, t1 in enumerate(T1):
    for j, t2 in enumerate(T2):
        if t2 <= t1: continue
        errs = []
        for c in corp:
            a = analyse(c, models, Config(tau1=t1, tau2=t2))
            errs.append(a.error())
        grid[i, j] = np.mean(errs)
    print("tau1", t1, np.round(grid[i], 4))

# ---- (2) alpha x beta (lineage fixed)
AL = [0.25, 0.5, 0.75, 0.9, 1.0]; BE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ab = np.zeros((len(AL), len(BE)))
an = [analyse(c, models) for c in corp]
for i, al in enumerate(AL):
    for j, be in enumerate(BE):
        cfg = Config(alpha=al, beta=be)
        ab[i, j] = np.mean([agg.tvd(x.corpus.truth_distribution(),
                                    agg.organic(x.P, compute_weights(x.graph, x.graph.conf, x.corpus.p_ind, x.corpus.p_syn, x.graph.p_coord, cfg)))
                            for x in an])
# ---- (3) tune (alpha, beta, gamma) on validation corpora, evaluate on held-out test corpora
val = [analyse(generate_corpus(SimConfig(n=N, contamination=r, seed=1200 + k)), models) for k, r in enumerate([.2, .3, .4, .5] * 2)]
test = [analyse(generate_corpus(SimConfig(n=N, contamination=r, seed=1500 + k)), models) for k, r in enumerate([.2, .3, .4, .5] * 4)]
def score(sets, cfg):
    return np.mean([agg.tvd(x.corpus.truth_distribution(), agg.organic(x.P, compute_weights(
        x.graph, x.graph.conf, x.corpus.p_ind, x.corpus.p_syn, x.graph.p_coord, cfg))) for x in sets])
best = min(((score(val, Config(alpha=al, beta=be, gamma=ga)), al, be, ga)
            for al, be, ga in itertools.product([0.5, 0.75, 0.9, 1.0], [0, .4, .8], [0, .3, .6])))
tuned = Config(alpha=best[1], beta=best[2], gamma=best[3])
tun = dict(best_alpha=best[1], best_beta=best[2], best_gamma=best[3], validation_err=best[0],
           test_err_tuned=score(test, tuned), test_err_default=score(test, Config()),
           test_err_raw=float(np.mean([x.corpus and agg.tvd(x.corpus.truth_distribution(), x.S_obs) for x in test])))
save_json("e7_sensitivity.json", dict(tau1=T1, tau2=T2, tau_grid=grid, alpha=AL, beta=BE, ab_grid=ab, tuning=tun))
print(tun)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
for k, (M, xl, yl, xt, yt, ttl) in enumerate([(grid, "τ2 (strong-derivation threshold)", "τ1 (candidate threshold)", T2, T1, "Threshold sensitivity (error)"),
                                              (ab, "β (synthetic attenuation)", "α (path attenuation)", BE, AL, "Weight-exponent sensitivity (error)")]):
    im = ax[k].imshow(M, cmap="viridis_r", aspect="auto", origin="lower")
    ax[k].set_xticks(range(len(xt))); ax[k].set_xticklabels(xt); ax[k].set_yticks(range(len(yt))); ax[k].set_yticklabels(yt)
    ax[k].set_xlabel(xl); ax[k].set_ylabel(yl); ax[k].set_title(ttl); ax[k].grid(False)
    for (i, j), v in np.ndenumerate(M):
        if not np.isnan(v): ax[k].text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=7, color="white" if v > np.nanmedian(M) else "black")
    plt.colorbar(im, ax=ax[k], fraction=.046)
plt.tight_layout(); plt.savefig(FIG / "e7_sensitivity.png"); plt.close()
