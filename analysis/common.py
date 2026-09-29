"""Shared paths and plot configuration for all reproducible analyses."""

from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "recomputed"
SUP = OUT / "supplementary"


def prepare_output():
    OUT.mkdir(parents=True, exist_ok=True)
    SUP.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.usetex": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 200,
        }
    )
