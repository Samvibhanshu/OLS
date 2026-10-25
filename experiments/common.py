import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = ROOT / "results"; FIG = RES / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

PAL = {"Raw aggregation": "#9aa0a6", "Deduplication (tau=0.90)": "#d98e04", "Deduplication (tuned tau)": "#f0b429",
       "Synthetic filtering": "#2a9d8f", "Message-level weighting": "#8d6cab",
       "Tree-normalised (1/size)": "#4f86c6", "Proposed (inferred lineage)": "#c8102e",
       "Proposed (oracle lineage)": "#7a0a1b"}


def style():
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25, "font.size": 10, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "legend.frameon": False})


def save_json(name, obj):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    (RES / name).write_text(json.dumps(obj, indent=2, default=conv))
