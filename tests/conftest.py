import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pytest
from opinion_lineage import *


@pytest.fixture(scope="session")
def models():
    return train_models()


@pytest.fixture(scope="session")
def analysis(models):
    c = generate_corpus(SimConfig(n=2500, contamination=0.4, seed=5))
    return analyse(c, models)


def make_store(a):
    return OpinionStore(a.graph, a.P, a.corpus.p_ind, a.corpus.p_syn, a.graph.p_coord, a.corpus.bucket, a.cfg)


def random_events(a, k, seed=0):
    rng = np.random.default_rng(seed)
    g = a.graph
    roots = np.flatnonzero(g.parent < 0); non = np.flatnonzero(g.parent >= 0)
    ev = [RevisionEvent(ROOT_INDEPENDENCE, int(r), float(rng.uniform(0, .4))) for r in rng.choice(roots, k, replace=False)]
    ev += [RevisionEvent(EDGE_CONFIDENCE, int(r), float(rng.uniform(.1, 1))) for r in rng.choice(non, k, replace=False)]
    ev += [RevisionEvent(RECORD_SYNTHETIC, int(r), float(rng.uniform())) for r in rng.choice(g.n, k, replace=False)]
    ev += [RevisionEvent(RECORD_COORDINATION, int(r), float(rng.uniform())) for r in rng.choice(g.n, k, replace=False)]
    rng.shuffle(ev)
    return ev
