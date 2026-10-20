"""Observed / organic sentiment, Sentiment Contamination Index, independent vs amplification mass."""
from __future__ import annotations
import numpy as np


def observed(P: np.ndarray) -> np.ndarray:
    return P.mean(axis=0)


def organic(P: np.ndarray, W: np.ndarray):
    Z = W.sum()
    return None if Z <= 0 else (W[:, None] * P).sum(axis=0) / Z


def sci(S_obs: np.ndarray, S_org: np.ndarray) -> float:
    """Sentiment Contamination Index = total-variation distance between observed and organic distributions"""
    return 0.5 * float(np.abs(S_obs - S_org).sum())


def masses(W: np.ndarray) -> dict:
    """Independent opinion mass = effective number of independent opinions (sum of weights);
    amplification mass = the remainder N - sum(W) that repetition / synthetic content added."""
    n = len(W)
    return {"independent_mass": float(W.sum()), "amplification_mass": float(n - W.sum()), "n": n}


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())
