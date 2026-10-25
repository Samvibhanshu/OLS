"""Lineage-structural contribution weights (patent §6):

    W_i = P_ind(r(i)) * prod_{e in path(r(i),i)} (1 - alpha*c_e) * (1 - beta*P_syn,i) * (1 - gamma*P_coord,i)
"""
from __future__ import annotations
import numpy as np
from .config import Config
from .lineage import LineageGraph


def path_products(g: LineageGraph, conf: np.ndarray, alpha: float):
    """pp[i] = prod over the path root->i of (1 - alpha*c_e); evaluated level by level (vectorised)."""
    pp = np.ones(g.n)
    for lv in g.levels[1:]:
        pp[lv] = pp[g.parent[lv]] * (1.0 - alpha * conf[lv])
    return pp


def compute_weights(g: LineageGraph, conf, p_ind, p_syn, p_coord, cfg: Config, parts: bool = False):
    pp = path_products(g, conf, cfg.alpha)
    root_pind = p_ind[g.root]
    W = root_pind * pp * (1.0 - cfg.beta * p_syn) * (1.0 - cfg.gamma * p_coord)
    return (W, pp, root_pind) if parts else W


def propagation_confidence(g: LineageGraph, conf: np.ndarray, v: int) -> np.ndarray:
    """q_{v,u} = product of edge confidences on path(v,u) (edges strictly below v); q_{v,v}=1; 0 outside D(v)."""
    q = np.zeros(g.n)
    q[v] = 1.0
    for lv in g.levels[g.depth[v] + 1:]:
        q[lv] = q[g.parent[lv]] * conf[lv]
    return q


# ---- alternative weighting schemes used in the ablation ----------------------------------------------------------
def weights_lineage_only(g, conf, cfg):
    return path_products(g, conf, cfg.alpha)


def weights_lineage_syn(g, conf, p_syn, cfg):
    return path_products(g, conf, cfg.alpha) * (1 - cfg.beta * p_syn)


def weights_soft(g, conf, p_ind, cfg):
    """harmonic ("soft") discount: W = P_ind(root) / (1 + alpha * sum of edge confidences on the path)"""
    s = np.zeros(g.n)
    for lv in g.levels[1:]:
        s[lv] = s[g.parent[lv]] + conf[lv]
    return p_ind[g.root] / (1.0 + cfg.alpha * s)


def weights_tree_normalised(g):
    """every lineage tree contributes exactly one unit of mass, shared equally among its members"""
    size = np.bincount(g.root, minlength=g.n)
    return 1.0 / size[g.root]


def weights_message_level(p_syn, beta=1.0):
    """bot-score-style per-message authenticity weighting, no lineage structure"""
    return 1.0 - beta * p_syn
