"""Hyper-parameters of the Opinion-Lineage framework (patent §7, thresholds in "Edge-Insertion Rule")."""
from __future__ import annotations
from dataclasses import dataclass, replace

# Edge types of the Opinion-Lineage Graph
NONE, REPOST_OF, QUOTES, PARAPHRASES, AI_DERIVATIVE_OF, COORDINATED_WITH, UNCERTAIN_DEPENDENCY = range(7)
EDGE_NAMES = ["NONE", "REPOST_OF", "QUOTES", "PARAPHRASES", "AI_DERIVATIVE_OF",
              "COORDINATED_WITH", "UNCERTAIN_DEPENDENCY"]

# Ground-truth record kinds produced by the simulator
K_INDEPENDENT, K_REPOST, K_QUOTE, K_PARAPHRASE, K_AI, K_COORD = range(6)
KIND_NAMES = ["independent", "repost", "quote", "paraphrase", "ai_derivative", "coordinated"]


@dataclass(frozen=True)
class Config:
    # similarity / coordination thresholds  [tau1=0.70 tau2=0.85 tau3=0.70 tau4=0.75]
    tau1: float = 0.70
    tau2: float = 0.85
    tau3: float = 0.70
    tau4: float = 0.75
    # contribution-weight exponents:  W_i = P_ind(r(i)) * prod(1-alpha*c_e) * (1-beta*P_syn) * (1-gamma*P_coord)
    alpha: float = 1.0
    beta: float = 0.8
    gamma: float = 0.0          # the patent's selected configuration disables coordination attenuation
    # edge-confidence functions
    explicit_conf: float = 0.98   # REPOST_OF / QUOTES from platform metadata
    para_lo: float = 0.60         # PARAPHRASES: c = para_lo + (para_hi-para_lo)*(S-tau2)/(1-tau2)
    para_hi: float = 0.97
    uncertain_max: float = 0.50   # UNCERTAIN_DEPENDENCY: c = uncertain_max*(S-tau1)/(1-tau1)  (reduced value)
    coord_tau_days: float = 0.10  # temporal decay of the coordination score
    coord_cross_cluster: float = 0.60
    n_classes: int = 3
    model_version: str = "enc-v1/clf-v1"

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)
