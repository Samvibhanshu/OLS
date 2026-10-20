"""Quick tour: the patent's A->B->C->D example, then a 3,000-record stream with one lineage drawn and rolled back."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, networkx as nx
from opinion_lineage import *
from opinion_lineage import aggregate as agg
from opinion_lineage.lineage import LineageGraph
from opinion_lineage.weighting import compute_weights

# ---- 1. toy: A -> B -> C -> D  and unrelated F -> G -> H
parent = np.array([-1, 0, 1, 2, -1, 4, 5]); conf = np.array([0, .9, .8, .5, 0, .7, .6]); z = np.zeros(7)
g = LineageGraph(parent, np.zeros(7, np.int8), conf, z, z.copy(), z, z.astype(np.int32))
P = np.array([[.1, .1, .8]] * 4 + [[.7, .2, .1]] * 3)         # lineage 1 is positive, lineage 2 negative
p_ind = np.array([.9, .1, .1, .1, .8, .1, .1]); cfg = Config(beta=0)
st = OpinionStore(g, P, p_ind, z, z, np.zeros(7, int), cfg)
print("weights before:", np.round(st.W, 3), " organic:", np.round(st.organic(), 3))
rep = RollbackEngine(st).apply(RevisionEvent(ROOT_INDEPENDENCE, 0, 0.0))     # A is later found unreliable
print(f"A invalidated -> touched {rep.records_touched} of 7 records (A,B,C,D); F,G,H untouched")
print("weights after :", np.round(st.W, 3), " organic:", np.round(st.organic(), 3))

# ---- 2. stream
models = train_models()
c = generate_corpus(SimConfig(n=3000, contamination=0.45, seed=8))
a = analyse(c, models)
print(f"\nstream: truth={np.round(c.truth_distribution(),3)} observed={np.round(a.S_obs,3)} organic={np.round(a.S_org,3)}  SCI={a.SCI:.3f}")
m = agg.masses(a.W); print(f"independent mass {m['independent_mass']:.0f} vs amplification mass {m['amplification_mass']:.0f} of {m['n']} records")
sizes = a.graph.lineage_sizes(); root = int(np.argmax(np.where(a.graph.parent < 0, sizes, 0)))
nodes = [root]; k = 0
while k < len(nodes):
    u = nodes[k]; nodes += a.graph.child_idx[a.graph.child_ptr[u]:a.graph.child_ptr[u + 1]].tolist(); k += 1
G = a.graph.to_networkx(nodes)
pos = nx.nx_agraph.graphviz_layout(G, prog="dot") if False else None
levels = {n: int(a.graph.depth[n]) for n in G}
for n in G: G.nodes[n]["subset"] = levels[n]
pos = nx.multipartite_layout(G, subset_key="subset", align="horizontal")
fig, ax = plt.subplots(figsize=(8, 4.5))
nx.draw_networkx_edges(G, pos, ax=ax, alpha=.35, arrows=False)
sc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[40 + 600 * a.W[n] for n in G], node_color=[a.W[n] for n in G], cmap="viridis_r")
ax.set_title(f"Largest inferred opinion lineage ({len(nodes)} records) — node size/colour = contribution weight W_i"); ax.axis("off")
plt.colorbar(sc, ax=ax, fraction=.03, label="W_i"); plt.tight_layout()
out = pathlib.Path(__file__).parent / "results" / "figures"; out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "demo_lineage.png", dpi=150)
print("saved", out / "demo_lineage.png")
