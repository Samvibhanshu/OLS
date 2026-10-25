"""R6: efficiency of the RAG layer - build time, per-question latency (retrieve + aggregate + generate + audit), and the cost of a
retraction vs a full rebuild, for growing streams (2 CPU cores, pure Python/NumPy, stand-in models)."""
from rag_common import *
import pandas as pd, time
from opinion_lineage import *
from opinion_lineage.rag import *

style(); models = train_models(); rows = []
for n in (6000, 12000, 24000):
    c = generate_corpus(SimConfig(n=n, contamination=0.4, sleeper_root_share=0.1, seed=900))
    t0 = time.perf_counter(); a = analyse(c, models); t_lineage = time.perf_counter() - t0
    t0 = time.perf_counter(); rag = LineageRAG(c, models, K=300, m=8, analysis=a); t_rag = time.perf_counter() - t0
    qs = [make_query(tp, i=k) for tp in range(len(TOPICS)) for k in range(3)]
    rag.ask(qs[0], "lineage")                                      # warm-up
    lat = {"naive_topm": [], "lineage": []}
    for q in qs * 3:
        for s_ in lat:
            t0 = time.perf_counter(); ans = rag.ask(q, s_); verify_answer(ans, rag.store.P, rag.texts); lat[s_].append(1e3 * (time.perf_counter() - t0))
    g = rag.analysis.graph; ex = np.flatnonzero((c.campaign == -2) & (g.parent < 0))
    reps = rag.retract([RevisionEvent(ROOT_INDEPENDENCE, int(r), 0.03) for r in ex]); sel = sum(r.seconds for r in reps)
    ref = rag.reference_rebuild()
    rows.append(dict(posts=n, lineage_build_s=t_lineage, index_and_store_s=t_rag, naive_p50_ms=np.percentile(lat["naive_topm"], 50),
                     lineage_p50_ms=np.percentile(lat["lineage"], 50), lineage_p95_ms=np.percentile(lat["lineage"], 95),
                     retract_events=len(ex), retract_ms=1e3 * sel, rebuild_s=ref["seconds"]))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(RES / "r6_rag_latency.csv", index=False); (RES / "r6_rag_latency.md").write_text(df.round(3).to_markdown(index=False))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
ax[0].plot(df.posts, df.naive_p50_ms, "o-", color="#9aa0a6", label="naive RAG p50"); ax[0].plot(df.posts, df.lineage_p50_ms, "o-", color="#c8102e", label="lineage-aware p50")
ax[0].plot(df.posts, df.lineage_p95_ms, "s--", color="#e8a0a8", label="lineage-aware p95"); ax[0].set_xlabel("posts indexed"); ax[0].set_ylabel("ms per question"); ax[0].legend(); ax[0].set_title("Answer latency (incl. grounding audit)")
ax[1].plot(df.posts, df.retract_ms / 1e3, "o-", color="#c8102e", label="selective rollback"); ax[1].plot(df.posts, df.rebuild_s, "o-", color="#9aa0a6", label="rebuild (stand-in models)")
ax[1].set_xlabel("posts indexed"); ax[1].set_ylabel("seconds"); ax[1].legend(); ax[1].set_title("Applying a source retraction")
plt.tight_layout(); plt.savefig(FIG / "r6_rag_latency.png"); plt.close()
