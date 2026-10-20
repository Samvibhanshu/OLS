"""R4: retroactive correction of a RAG system.  A compromised "sleeper" feed (20% of roots) is trusted, then exposed.
What does each system do with the news, what does it cost, and are answers right afterwards?

Systems compared (same candidates, same questions, same truth = organic independent posts on the topic):
  Naive RAG + delete exposed posts        remove the exposed roots from the candidate list; descendants stay (nothing links them)
  Naive RAG + delete the whole family     ORACLE: also removes every true descendant (needs ground truth, unavailable in practice)
  Text dedup + provenance score           collapse near-duplicates, weight by the live provenance score P_ind (exposed roots -> 0.03)
  Lineage-aware RAG (selective rollback)  apply the revision through the lineage graph in place (no re-embedding, no re-index)
Also measured: cost vs a conventional rebuild, exactness vs full recomputation, and precise cache invalidation.
"""
from rag_common import *
import pandas as pd, time
from scipy import stats
from opinion_lineage import *
from opinion_lineage.rag import *
from opinion_lineage.rag import strategies as S_

style()
SEEDS = range(8)
N, K, M = 6000, 300, 8
models = train_models()
SYSTEMS = ["Naive RAG + delete exposed posts", "Naive RAG + delete whole family (oracle)", "Text dedup + provenance score",
           "Lineage-aware RAG (selective rollback)"]


def evaluate(rag, c, troot, family, exposed, phase, seed, rows, answers=None):
    for tp in range(len(TOPICS)):
        q = make_query(tp, i=tp); truth = topic_truth(c, tp)
        ids, rel = rag.retrieve(q)
        keep_roots = ~np.isin(ids, exposed) if phase == "after" else np.ones(len(ids), bool)
        keep_fam = ~family[ids] if phase == "after" else np.ones(len(ids), bool)
        res = {SYSTEMS[0]: S_.naive_all(rag.context(ids[keep_roots], rel[keep_roots])),
               SYSTEMS[1]: S_.naive_all(rag.context(ids[keep_fam], rel[keep_fam])),
               SYSTEMS[2]: S_.dedup_provenance(rag.context(ids, rel)), SYSTEMS[3]: S_.lineage_aware(rag.context(ids, rel))}
        for name, r in res.items():
            w = r.weights / max(r.weights.sum(), 1e-12)
            rows.append(dict(seed=seed, phase=phase, topic=tp, system=name, err=agg.tvd(truth, r.S),
                             family_share=float((w * family[r.evidence]).sum()), cite_family=float(family[r.cited].mean()) if len(r.cited) else 0.0))
        if answers is not None:
            answers.append(rag.answer_from(q, res[SYSTEMS[3]], lineage=True, generate=False))


def run(seed):
    c = generate_corpus(SimConfig(n=N, contamination=0.3, sleeper_root_share=0.2, seed=600 + seed))
    rag = LineageRAG(c, models, K=K, m=M); g = rag.analysis.graph; troot = true_root(c)
    family = (c.campaign[troot] == -2)                                  # true descendants of the compromised feed
    exposed = np.flatnonzero((c.campaign == -2) & (g.parent < 0))       # roots the provenance monitor exposes
    rows, cached = [], []
    evaluate(rag, c, troot, family, exposed, "before", seed, rows, cached)
    # many cached answers (3 phrasings x 8 topics) for the invalidation study
    cache = [rag.ask(make_query(tp, i=k), "lineage", generate=False) for tp in range(len(TOPICS)) for k in range(3)]
    fp = rag.models.forward_passes
    ev = [RevisionEvent(ROOT_INDEPENDENCE, int(r), 0.03) for r in exposed]
    reps = rag.retract(ev)
    sel_s = sum(r.seconds for r in reps); touched = sum(r.records_touched for r in reps)
    model_rows = {k: rag.models.forward_passes[k] - fp[k] for k in fp}
    evaluate(rag, c, troot, family, exposed, "after", seed, rows)
    # reference: full reprocessing + re-index, then the same lineage-aware answers with the recomputed weights
    ref = rag.reference_rebuild()
    dW = float(np.abs(ref["W"] - rag.store.W).max()); dS = 0.0
    for tp in range(len(TOPICS)):
        ids, rel = rag.retrieve(make_query(tp, i=tp))
        a = S_.lineage_aware(rag.context(ids, rel)).S; b = S_.lineage_aware(rag.context(ids, rel, W=ref["W"])).S
        dS = max(dS, float(np.abs(a - b).max()))
    # precise invalidation after exposing ONE lineage (the largest exposed one) on a fresh system
    rag2 = LineageRAG(c, models, K=K, m=M)
    cache2 = [rag2.ask(make_query(tp, i=k), "lineage", generate=False) for tp in range(len(TOPICS)) for k in range(3)]
    sizes = rag2.store.g.lineage_sizes(); r1 = int(exposed[np.argmax(sizes[exposed])])
    rag2.retract(RevisionEvent(ROOT_INDEPENDENCE, r1, 0.03))
    inv = len(rag2.invalidated(cache2))
    cost = dict(seed=seed, exposed_roots=len(exposed), events=len(ev), records_touched=int(touched), pct_touched=100 * touched / c.n,
                selective_ms=1e3 * sel_s, rebuild_s=ref["seconds"], index_rebuild_s=ref["index_seconds"], model_rows_selective=sum(model_rows.values()),
                model_rows_rebuild=2 * c.n, max_dW=dW, max_dS=dS, answers=len(cache2), invalidated_single_lineage=inv)
    return rows, cost


if __name__ == "__main__":
    t0 = time.time(); rows, costs = [], []
    for s in SEEDS:
        r, cst = run(s); rows += r; costs.append(cst); print(f"seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); dc = pd.DataFrame(costs)
    df.to_csv(RES / "r4_correction_runs.csv", index=False); dc.to_csv(RES / "r4_correction_costs.csv", index=False)
    sm = df.groupby(["system", "phase", "seed"]).mean(numeric_only=True).reset_index()
    out = {}
    lines = ["| System | error before | error after | sleeper-family share of evidence before | after | sleeper posts among cited, after |", "|---|---|---|---|---|---|"]
    for s_ in SYSTEMS:
        b = sm[(sm.system == s_) & (sm.phase == "before")]; a = sm[(sm.system == s_) & (sm.phase == "after")]
        out[s_] = dict(err_before=[b.err.mean(), b.err.std()], err_after=[a.err.mean(), a.err.std()], fam_before=b.family_share.mean(),
                       fam_after=a.family_share.mean(), cite_family_after=a.cite_family.mean())
        lines.append(f"| {s_} | {b.err.mean():.4f} ± {b.err.std():.4f} | {a.err.mean():.4f} ± {a.err.std():.4f} | {b.family_share.mean():.1%} | "
                     f"{a.family_share.mean():.1%} | {a.cite_family.mean():.1%} |")
    (RES / "r4_correction_table.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    L, D = SYSTEMS[3], SYSTEMS[2]
    def pair(col, s1, s2):
        a = sm[(sm.system == s1) & (sm.phase == "after")].sort_values("seed")[col].values
        b = sm[(sm.system == s2) & (sm.phase == "after")].sort_values("seed")[col].values
        return dict(mean_diff=float((b - a).mean()), p=float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0)
    out["tests_after"] = {f"err: lineage vs {s_}": pair("err", L, s_) for s_ in SYSTEMS[:3]}
    out["tests_after"].update({f"family_share: lineage vs {s_}": pair("family_share", L, s_) for s_ in SYSTEMS[:3]})
    cs = dc.mean(numeric_only=True).to_dict(); out["cost"] = cs
    save_json("r4_correction.json", out); print(json.dumps(out["tests_after"], indent=1)); print(dc.mean(numeric_only=True).round(4).to_string())
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.2)); x = np.arange(len(SYSTEMS)); w = .38
    short = ["naive +\ndelete roots", "naive + delete\nfamily (oracle)", "text dedup +\nprovenance", "lineage-aware\n(rollback)"]
    for k, (ph, col) in enumerate([("err_before", "#d98e04"), ("err_after", "#c8102e")]):
        ax[0].bar(x + (k - .5) * w, [out[s_][ph][0] for s_ in SYSTEMS], w, yerr=[out[s_][ph][1] for s_ in SYSTEMS], capsize=2, color=col,
                  label=["before exposure", "after exposure"][k])
    ax[0].set_xticks(x); ax[0].set_xticklabels(short, fontsize=7.5); ax[0].set_ylabel("answer error (TVD)"); ax[0].legend(fontsize=8)
    ax[0].set_title("Answer error before / after the feed is exposed")
    for k, (ph, col) in enumerate([("fam_before", "#d98e04"), ("fam_after", "#c8102e")]):
        ax[1].bar(x + (k - .5) * w, [out[s_][ph] for s_ in SYSTEMS], w, color=col)
    ax[1].set_xticks(x); ax[1].set_xticklabels(short, fontsize=7.5); ax[1].set_ylabel("share of answer weight from the compromised feed")
    ax[1].set_title("Residual influence of the exposed feed")
    ax[2].bar(["selective\nrollback", "rebuild\n(re-encode + re-index)"], [cs["selective_ms"] / 1e3, cs["rebuild_s"]], color=["#c8102e", "#9aa0a6"])
    ax[2].set_ylabel("seconds"); ax[2].set_title(f"Cost of applying the news\n({cs['pct_touched']:.1f}% of posts touched, {cs['model_rows_selective']:.0f} model rows vs {cs['model_rows_rebuild']:.0f})")
    plt.tight_layout(); plt.savefig(FIG / "r4_rag_correction.png"); plt.close(); print("done", time.time() - t0)
