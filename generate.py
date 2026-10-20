"""Answer objects, the grounded generators and the guardrails.

ExtractiveGenerator  deterministic, template-based: every number is computed from the evidence and every quote is a verbatim
                     post, so it cannot hallucinate.  This is what the experiments use.
LLMGenerator         builds a lineage-annotated prompt for any `complete(prompt) -> str` callable (an LLM API client), rejects
                     answers that cite ids outside the evidence and falls back to the extractive answer.  It is exercised in the
                     tests with a stub model only; no real LLM was called while building this package (no network access).
verify_answer        recomputes an answer's statistics from the store and checks every citation (grounding audit).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

LABELS = ["negative", "neutral", "positive"]


@dataclass
class Answer:
    query: str
    strategy: str
    S: np.ndarray                  # answer distribution (nan if abstained)
    S_raw: np.ndarray              # what a plain count of the retrieved posts says
    evidence: np.ndarray           # records the statistic is computed from
    weights: np.ndarray            # their weights at answer time
    cited: list                    # [{id, text, weight, copies, relevance}]
    n_lineages: int
    version: int                   # store version the answer was computed at
    flags: list = field(default_factory=list)
    text: str = ""

    @property
    def effective_n(self) -> float: return float(self.weights.sum())
    @property
    def raw_n(self) -> int: return len(self.evidence)
    @property
    def amplification_share(self) -> float: return 1.0 - self.effective_n / max(self.raw_n, 1)
    @property
    def abstained(self) -> bool: return bool(np.isnan(self.S).any())


def _pct(v): return " / ".join(f"{100 * x:.0f}% {l}" for x, l in zip(v, LABELS))


class ExtractiveGenerator:
    def __init__(self, quote_chars: int = 140):
        self.q = quote_chars

    def __call__(self, a: Answer, lineage: bool) -> str:
        lines = []
        if a.abstained:
            lines.append(f"I cannot give an organic sentiment figure for this question: none of the {a.raw_n} retrieved posts carries "
                         "independent weight (their sources were retracted or fully explained by amplification).")
        elif lineage:
            lines.append(f"Organic sentiment among the {a.raw_n} most relevant posts: {_pct(a.S)}.")
            lines.append(f"A plain count would say {_pct(a.S_raw)}. The difference is amplification: these {a.raw_n} posts reduce to "
                         f"about {a.effective_n:.1f} independent opinions across {a.n_lineages} lineages ({100 * a.amplification_share:.0f}% of the "
                         "volume is reposts, paraphrases, AI rewrites or coordinated copies).")
        else:
            lines.append(f"Sentiment among the {a.raw_n} posts considered: {_pct(a.S)}.")
        for f in a.flags:
            lines.append("Note: " + f)
        if a.cited:
            lines.append("Evidence:")
            for c in a.cited:
                tail = (f" (weight {c['weight']:.2f}" + (f", {c['copies']} copies/derivatives in this result" if c['copies'] else "") + ")") if lineage else ""
                lines.append(f"  [{c['id']}] \"{c['text'][:self.q]}\"{tail}")
        return "\n".join(lines)


def build_prompt(a: Answer) -> str:
    ev = "\n".join(f"[{c['id']}] weight={c['weight']:.2f} copies_in_result={c['copies']} :: {c['text']}" for c in a.cited)
    stats = (f"organic sentiment (lineage-weighted): {_pct(a.S)}\nraw count: {_pct(a.S_raw)}\n"
             f"retrieved posts: {a.raw_n}; independent opinions (sum of weights): {a.effective_n:.1f}; lineages: {a.n_lineages}")
    return ("You answer questions about public opinion using ONLY the evidence below.\n"
            "Rules: (1) report the organic sentiment, not the raw count, and say how much of the volume is amplification; "
            "(2) cite evidence as [id] and never cite an id that is not listed; (3) if flags are present, mention them; "
            "(4) do not add facts that are not in the evidence.\n\n"
            f"Question: {a.query}\n\nStatistics:\n{stats}\n\nFlags: {'; '.join(a.flags) or 'none'}\n\nEvidence:\n{ev}\n\nAnswer:")


_CITE = re.compile(r"\[(\d+)\]")


def check_citations(text: str, allowed) -> tuple[bool, list]:
    bad = [int(x) for x in _CITE.findall(text) if int(x) not in set(int(i) for i in allowed)]
    return len(bad) == 0, bad


class LLMGenerator:
    """complete: any callable str -> str (e.g. a thin wrapper around an LLM API).  Untested against a real model here."""
    def __init__(self, complete: Callable[[str], str], fallback: ExtractiveGenerator | None = None):
        self.complete, self.fallback = complete, fallback or ExtractiveGenerator()
        self.rejected = 0

    def __call__(self, a: Answer, lineage: bool) -> str:
        out = self.complete(build_prompt(a))
        ok, bad = check_citations(out, [c["id"] for c in a.cited])
        if not ok or not _CITE.search(out):
            self.rejected += 1                     # hallucinated / missing citations: refuse the LLM text
            return self.fallback(a, lineage)
        return out


def verify_answer(a: Answer, P: np.ndarray, texts: list[str], tol: float = 1e-9) -> dict:
    """Grounding audit: statistics recomputable from (evidence, weights, P); every citation is evidence and quoted verbatim."""
    if a.abstained:
        return dict(stats_ok=True, citations_in_evidence=True, quotes_verbatim=True, ok=True)
    S = (a.weights[:, None] * P[a.evidence]).sum(0) / a.weights.sum()
    ev = set(int(i) for i in a.evidence)
    ci = all(int(c["id"]) in ev for c in a.cited)
    qv = all(c["text"] == texts[int(c["id"])] for c in a.cited)
    so = bool(np.abs(S - a.S).max() < tol)
    return dict(stats_ok=so, citations_in_evidence=ci, quotes_verbatim=qv, ok=so and ci and qv)
