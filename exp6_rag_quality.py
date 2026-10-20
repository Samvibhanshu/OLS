"""R1: answer quality of lineage-aware RAG vs RAG baselines as contamination grows.   R2: which weights matter (ablation).
R3 (grounding audit) is computed on every answer produced here.

Per (contamination, seed): a 6,000-post stream, 8 topic questions ("How do people feel about <topic>?").  Every strategy sees the
same top-K=300 candidates.  Truth = sentiment distribution of the *organic independent* posts on that topic.
"""
from rag_common import *
import pandas as pd, time
from scipy import stats
from opinion_lineage import *
from opinion_lineage import weighting as Wt
from opinion_lineage.rag import *
from opinion_lineage.rag import strategies as S_

style()
LEVELS = [0.0, 0.2, 0.4, 0.5]
SEEDS = range(8)
N, K, M = 6000, 300, 8
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]
models = train_models()
NAMES = {"naive_topm": "Naive RAG (top-m)", "naive_all": "Naive RAG (all K)", "mmr": "MMR diversification (m)",
         "dedup_0.8": "Text near-duplicate collapse (tau=0.8)", "dedup_prov": "Text dedup + provenance score", "lineage": "Lineage-aware RAG (ours)",
         "lineage_oracle": "Lineage-aware RAG (oracle lineage)"}


def variants(rag):
    s, g, cfg = rag.store, rag.analysis.graph, rag.cfg
    isroot = (g.parent < 0).astype(float)
    return {"Full weights W (ours)": s.W.copy(),
            "Lineage structure only": Wt.weights_lineage_only(g, g.conf, cfg),
            "Lineage + synthetic term": Wt.weights_lineage_syn(g, g.conf, rag.corpus.p_syn, cfg),
            "Tree-normalised (1/size)": Wt.weights_tree_normalised(g),
            "Message-level (no lineage)": Wt.weights_message_level(rag.corpus.p_syn, cfg.beta),
            "Roots only (drop all derivatives)": isroot}


def run(job):
    rho, seed = job
    c = generate_corpus(SimConfig(n=N, contamination=rho, seed=500 + seed))
    rag = LineageRAG(c, models, K=K, m=M); rag_o = LineageRAG(c, models, K=K, m=M, analysis=analyse(c, models, oracle=True))
    troot = true_root(c); var = variants(rag); rows = []
    for tp in range(len(TOPICS)):
        q = make_query(tp, i=tp); truth = topic_truth(c, tp)
        ids, rel = rag.retrieve(q); ctx = rag.context(ids, rel); ctx_o = rag_o.context(ids, rel)
        res = {"naive_topm": S_.naive_topm(ctx), "naive_all": S_.naive_all(ctx), "mmr": S_.mmr(ctx), "dedup_prov": S_.dedup_provenance(ctx), "lineage": S_.lineage_aware(ctx),
               "lineage_oracle": S_.lineage_aware(ctx_o)}
        for tau in TAUS: res[f"dedup_{tau:g}"] = S_.text_dedup(ctx, tau)
        for vn, W in var.items(): res["abl:" + vn] = S_.lineage_aware(rag.context(ids, rel, W=W), name=vn)
        for k, r in res.items():
            ans = rag.answer_from(q, r, lineage=(k == "lineage")) if k in ("naive_topm", "lineage") else None
            rows.append(dict(rho=rho, seed=seed, topic=tp, method=k, err=agg.tvd(truth, r.S) if not np.isnan(r.S).any() else np.nan,
                             err_count=agg.tvd(truth, r.S_raw), eff_n=r.effective_n, n_lineages=r.n_lineages,
                             topic_prec=float(np.mean(c.topic[ids] == tp)),
                             grounded=(verify_answer(ans, rag.store.P, rag.texts)["ok"] if ans is not None else np.nan),
                             **cited_quality(c, troot, r.cited)))
    return rows


if __name__ == "__main__":
    t0 = time.time(); rows = []
    for rho in LEVELS:
        for seed in SEEDS:
            rows += run((rho, seed))
        print(f"rho={rho} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / "r1_rag_quality_runs.csv", index=False)
    seedmean = df.groupby(["rho", "method", "seed"]).mean(numeric_only=True).reset_index()
    tab = seedmean.groupby(["method", "rho"]).err.agg(["mean", "std"]).unstack("rho")
    # optimistic baseline: text-dedup threshold tuned per setting on the ground truth
    tuned = {r: min(TAUS, key=lambda t: tab.loc[f"dedup_{t:g}", ("mean", r)]) for r in LEVELS}
    order = ["naive_topm", "naive_all", "mmr", "dedup_0.8", "dedup_prov", "lineage", "lineage_oracle"]
    lines = ["| Method | " + " | ".join(f"{int(r*100)}%" for r in LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for m in order[:5]:
        lines.append(f"| {NAMES[m]} | " + " | ".join(f"{tab.loc[m, ('mean', r)]:.4f} ± {tab.loc[m, ('std', r)]:.4f}" for r in LEVELS) + " |")
    lines.append("| Text dedup, tau tuned on ground truth* | " + " | ".join(f"{tab.loc[f'dedup_{tuned[r]:g}', ('mean', r)]:.4f} ± {tab.loc[f'dedup_{tuned[r]:g}', ('std', r)]:.4f} (tau={tuned[r]:g})" for r in LEVELS) + " |")
    for m in order[5:]:
        lines.append(f"| **{NAMES[m]}** | " + " | ".join(f"{tab.loc[m, ('mean', r)]:.4f} ± {tab.loc[m, ('std', r)]:.4f}" for r in LEVELS) + " |")
    (RES / "r1_answer_error_table.md").write_text("\n".join(lines) + "\n\n*optimistic: threshold picked using the answer.\n")
    print("\n".join(lines))
    # paired tests: lineage vs best untuned baseline per level
    tests = {}
    for r in LEVELS:
        best = min(["naive_topm", "naive_all", "mmr", "dedup_0.8", "dedup_prov"], key=lambda m: tab.loc[m, ("mean", r)])
        a = seedmean[(seedmean.rho == r) & (seedmean.method == "lineage")].sort_values("seed").err.values
        b = seedmean[(seedmean.rho == r) & (seedmean.method == best)].sort_values("seed").err.values
        bt = seedmean[(seedmean.rho == r) & (seedmean.method == f"dedup_{tuned[r]:g}")].sort_values("seed").err.values
        tests[f"{int(r*100)}%"] = dict(best_baseline=NAMES[best], mean_diff=float((b - a).mean()),
                                        wilcoxon_p=float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0,
                                        vs_tuned_dedup_mean_diff=float((bt - a).mean()),
                                        vs_tuned_dedup_p=float(stats.wilcoxon(a, bt).pvalue) if np.any(a != bt) else 1.0)
    for r in LEVELS:
        a = seedmean[(seedmean.rho == r) & (seedmean.method == "lineage")].sort_values("seed").err.values
        for base in ("naive_topm", "naive_all", "mmr", "dedup_0.8", "dedup_prov"):
            b = seedmean[(seedmean.rho == r) & (seedmean.method == base)].sort_values("seed").err.values
            tests[f"{int(r*100)}%"][f"vs_{base}"] = dict(mean_diff=float((b - a).mean()), p=float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0)
    save_json("r1_tests.json", tests); print(json.dumps(tests, indent=1))
    # evidence quality (cited posts) at 40%
    q40 = seedmean[seedmean.rho == 0.4].groupby("method")[["cite_organic_root", "cite_distinct_lineage", "cite_campaign", "topic_prec", "eff_n", "n_lineages"]].mean()
    q40 = q40.loc[order].rename(index=NAMES).round(3); (RES / "r1_evidence_quality_40.md").write_text(q40.to_markdown()); print(q40.to_string())
    gr = df[df.grounded.notna()].grounded.mean(); print("grounding audit: share of answers passing verify_answer =", gr)
    save_json("r3_grounding.json", dict(answers_checked=int(df.grounded.notna().sum()), share_ok=float(gr)))
    # ablation @40%
    ab = seedmean[(seedmean.rho == 0.4) & seedmean.method.str.startswith("abl:")].groupby("method").err.agg(["mean", "std"])
    ab.index = [i[4:] for i in ab.index]
    full = seedmean[(seedmean.rho == 0.4) & (seedmean.method == "abl:Full weights W (ours)")].sort_values("seed").err.values
    abl_p = {v: float(stats.wilcoxon(seedmean[(seedmean.rho == 0.4) & (seedmean.method == "abl:" + v)].sort_values("seed").err.values, full).pvalue)
             for v in ab.index if v != "Full weights W (ours)"}
    save_json("r2_ablation_tests.json", abl_p)
    ab = ab.loc[[v for v in ['Full weights W (ours)', 'Lineage structure only', 'Lineage + synthetic term', 'Tree-normalised (1/size)', 'Message-level (no lineage)', 'Roots only (drop all derivatives)'] if v in ab.index]]
    (RES / "r2_weight_ablation_40.md").write_text(ab.round(4).to_markdown()); print(ab.round(4))

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    cols = {"naive_topm": "#9aa0a6", "naive_all": "#b8bcc2", "mmr": "#8d6cab", "dedup_0.8": "#d98e04", "dedup_prov": "#2a9d8f", "lineage": "#c8102e", "lineage_oracle": "#7a0a1b"}
    for m in order:
        ax[0].errorbar([r * 100 for r in LEVELS], [tab.loc[m, ("mean", r)] for r in LEVELS], [tab.loc[m, ("std", r)] for r in LEVELS],
                       marker="o", ms=4, capsize=2, color=cols[m], lw=2.4 if m == "lineage" else 1.5, label=NAMES[m])
    ax[0].plot([r * 100 for r in LEVELS], [tab.loc[f"dedup_{tuned[r]:g}", ("mean", r)] for r in LEVELS], "k--", lw=1, label="Text dedup, tau tuned on truth*")
    ax[0].set_xlabel("contamination (% derivative / amplified posts)"); ax[0].set_ylabel("answer error (TVD to organic truth)")
    ax[0].set_title("Answer error vs contamination"); ax[0].legend(fontsize=7)
    x = np.arange(len(order)); w = 0.27
    for k, (col, lab) in enumerate([("cite_organic_root", "organic independent posts"), ("cite_distinct_lineage", "distinct lineages"), ("cite_campaign", "campaign / sleeper posts")]):
        ax[1].bar(x + (k - 1) * w, [seedmean[(seedmean.rho == 0.4) & (seedmean.method == m)][col].mean() for m in order], w,
                  label=lab, color=["#4f86c6", "#2a9d8f", "#c8102e"][k])
    ax[1].set_xticks(x); ax[1].set_xticklabels(["naive\ntop-m", "naive\nall K", "MMR", "text\ndedup", "dedup+\nprov.", "lineage\n(ours)", "lineage\n(oracle)"], fontsize=8)
    ax[1].set_title("Quality of the cited evidence (40% contamination)"); ax[1].set_ylabel("share of the m cited posts"); ax[1].legend(fontsize=7)
    ax[2].barh(range(len(ab)), ab["mean"], xerr=ab["std"], color=["#c8102e"] + ["#9aa0a6"] * (len(ab) - 1), capsize=2)
    ax[2].set_yticks(range(len(ab))); ax[2].set_yticklabels(ab.index, fontsize=8); ax[2].invert_yaxis()
    ax[2].set_xlabel("answer error (TVD)"); ax[2].set_title("Which weights matter (40%)")
    plt.tight_layout(); plt.savefig(FIG / "r1_rag_quality.png"); plt.close(); print("done", time.time() - t0)
