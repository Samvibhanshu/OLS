from common import *
from opinion_lineage import aggregate as agg


def true_root(c):
    r = np.arange(c.n)
    for i in range(c.n):
        if c.true_parent[i] >= 0: r[i] = r[c.true_parent[i]]
    return r


def topic_truth(c, tp):
    m = c.organic_root & (c.topic == tp)
    return np.bincount(c.label[m], minlength=c.cfg.n_classes) / m.sum()


def cited_quality(c, troot, cited_ids):
    cited_ids = np.asarray(cited_ids, int)
    if len(cited_ids) == 0:
        return dict(cite_organic_root=np.nan, cite_distinct_lineage=np.nan, cite_campaign=np.nan)
    return dict(cite_organic_root=float(np.mean(c.organic_root[cited_ids] & (c.true_parent[cited_ids] < 0))),
                cite_distinct_lineage=float(len(set(troot[cited_ids].tolist())) / len(cited_ids)),
                cite_campaign=float(np.mean(c.campaign[cited_ids] != -1)))
