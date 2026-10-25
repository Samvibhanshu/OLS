"""R5: stress test - what happens when copies are disguised at the text level (rewritten instead of reposted verbatim)?
disguise = probability that a repost / quote / coordinated copy is rewritten (heavy synonyms, new opener and tail).
CAVEAT: the lineage graph is built from the embedding channel, which the renderer does not touch, so it is unaffected by
construction.  With a real encoder the graph would also degrade as rewrites drift semantically; this experiment shows when
text-level methods break, not how much a real encoder would hold up."""
from rag_common import *
import pandas as pd, time
from scipy import stats
from opinion_lineage import *
from opinion_lineage.rag import *
from opinion_lineage.rag import strategies as S_

style()
DIS = [0.0, 0.5, 1.0]; SEEDS = range(8); N, K, M = 6000, 300, 8
models = train_models()
KEEP = {"naive_all": "Naive RAG (all K)", "dedup_0.8": "Text near-duplicate collapse (tau=0.8)", "dedup_prov": "Text dedup + provenance score",
        "lineage": "Lineage-aware RAG (ours)"}

if __name__ == "__main__":
    t0 = time.time(); rows = []
    for d in DIS:
        for seed in SEEDS:
            c = generate_corpus(SimConfig(n=N, contamination=0.4, seed=700 + seed))
            rag = LineageRAG(c, models, K=K, m=M, text_disguise=d); troot = true_root(c)
            for tp in range(len(TOPICS)):
                q = make_query(tp, i=tp); truth = topic_truth(c, tp); ids, rel = rag.retrieve(q); ctx = rag.context(ids, rel)
                res = {"naive_all": S_.naive_all(ctx), "dedup_0.8": S_.text_dedup(ctx, 0.8), "dedup_prov": S_.dedup_provenance(ctx), "lineage": S_.lineage_aware(ctx)}
                for k, r in res.items():
                    rows.append(dict(disguise=d, seed=seed, topic=tp, method=k, err=agg.tvd(truth, r.S), **cited_quality(c, troot, r.cited)))
        print(f"disguise={d} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / "r5_stress_runs.csv", index=False)
    sm = df.groupby(["disguise", "method", "seed"]).mean(numeric_only=True).reset_index()
    tab = sm.groupby(["method", "disguise"]).err.agg(["mean", "std"]).unstack("disguise").loc[list(KEEP)]
    lines = ["| Method | " + " | ".join(f"{int(d*100)}% disguised" for d in DIS) + " |", "|---|" + "---|" * len(DIS)]
    for m, nm in KEEP.items():
        lines.append(f"| {nm} | " + " | ".join(f"{tab.loc[m, ('mean', d)]:.4f} ± {tab.loc[m, ('std', d)]:.4f}" for d in DIS) + " |")
    (RES / "r5_stress_table.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    tests = {}
    for d in DIS:
        a = sm[(sm.disguise == d) & (sm.method == "lineage")].sort_values("seed").err.values
        for b_ in ("dedup_0.8", "dedup_prov", "naive_all"):
            b = sm[(sm.disguise == d) & (sm.method == b_)].sort_values("seed").err.values
            tests[f"disguise={d}:{b_}"] = dict(mean_diff=float((b - a).mean()), p=float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0)
    save_json("r5_stress_tests.json", tests); print(json.dumps(tests, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1)); cols = {"naive_all": "#9aa0a6", "dedup_0.8": "#d98e04", "dedup_prov": "#2a9d8f", "lineage": "#c8102e"}
    for m, nm in KEEP.items():
        ax[0].errorbar([d * 100 for d in DIS], [tab.loc[m, ("mean", d)] for d in DIS], [tab.loc[m, ("std", d)] for d in DIS], marker="o", capsize=2,
                       color=cols[m], label=nm, lw=2.4 if m == "lineage" else 1.5)
        ax[1].plot([d * 100 for d in DIS], [sm[(sm.disguise == d) & (sm.method == m)].cite_organic_root.mean() for d in DIS], marker="o", color=cols[m], lw=2.4 if m == "lineage" else 1.5)
    ax[0].set_xlabel("% of copies disguised at text level"); ax[0].set_ylabel("answer error (TVD)"); ax[0].legend(fontsize=7); ax[0].set_title("Answer error when copies are rewritten (40% contamination)")
    ax[1].set_xlabel("% of copies disguised at text level"); ax[1].set_ylabel("share of cited posts that are organic independent posts"); ax[1].set_title("Evidence quality")
    plt.tight_layout(); plt.savefig(FIG / "r5_rag_stress.png"); plt.close(); print("done", time.time() - t0)
