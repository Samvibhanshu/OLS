#!/usr/bin/env python3
"""Ask the lineage-aware RAG a question about public opinion and compare it with a plain RAG.

    python3 rag_demo.py --list-topics
    python3 rag_demo.py "How do people feel about the metro fare hike?"
    python3 rag_demo.py --topic 2 --expose        # then expose the compromised feed and watch the answer change, in place

The stream is synthetic (see README); posts are rendered from the simulator's latent state.
"""
import argparse, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from opinion_lineage import *
from opinion_lineage.rag import *

ap = argparse.ArgumentParser()
ap.add_argument("question", nargs="?"); ap.add_argument("--topic", type=int); ap.add_argument("--list-topics", action="store_true")
ap.add_argument("--expose", action="store_true", help="expose the compromised 'sleeper' feed afterwards and re-answer")
ap.add_argument("--contamination", type=float, default=0.4); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
if a.list_topics:
    for i, (name, _) in enumerate(TOPICS): print(i, make_query(i))
    sys.exit(0)
q = a.question or make_query(a.topic if a.topic is not None else 0)
print("building stream, lineage graph and index ...", file=sys.stderr)
models = train_models()
c = generate_corpus(SimConfig(n=6000, contamination=a.contamination, sleeper_root_share=0.1 if a.expose else 0.0, seed=a.seed))
rag = LineageRAG(c, models)
tp = next((i for i, (name, _) in enumerate(TOPICS) if name in q), None)
truth = None
if tp is not None:
    m = c.organic_root & (c.topic == tp); truth = np.bincount(c.label[m], minlength=3) / m.sum()
fmt = lambda v: " / ".join(f"{100 * x:.0f}%" for x in v)
print(f"\nQ: {q}\n" + (f"(ground truth among organic independent posts: {fmt(truth)} neg/neu/pos)\n" if truth is not None else ""))
for strat, title in (("naive_topm", "PLAIN RAG (top-8 context)"), ("lineage", "LINEAGE-AWARE RAG")):
    ans = rag.ask(q, strat); print(f"=== {title} ===\n{ans.text}\n")
if a.expose:
    g = rag.analysis.graph; ex = np.flatnonzero((c.campaign == -2) & (g.parent < 0))
    before = rag.ask(q, "lineage")
    t0 = time.perf_counter(); reps = rag.retract([RevisionEvent(ROOT_INDEPENDENCE, int(r), 0.03) for r in ex]); dt = time.perf_counter() - t0
    print(f"*** provenance monitor exposes a compromised feed ({len(ex)} posts). Applied in place: {sum(r.records_touched for r in reps)} posts "
          f"re-weighted in {1e3 * dt:.0f} ms, 0 posts re-embedded, index untouched. Cached answer stale: {rag.is_stale(before)} ***\n")
    ans = rag.ask(q, "lineage"); print("=== LINEAGE-AWARE RAG, after exposure ===\n" + ans.text)
