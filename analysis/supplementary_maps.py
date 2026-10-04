"""Generate S6 with the full MP-SPMe and S8 from retained reduced-model data."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import OUT, SUP, prepare_output

GRID = np.linspace(0, 0.25, 11)


def lamp_map(plot_only=False, workers=1):
    """Figure S6 uses the full production MP-SPMe, with resumable checkpoints."""
    from analysis.fullmodel_lamp_map import run_map

    run_map(workers=workers, plot_only=plot_only)


def integrated_map():
    data = pd.read_csv(OUT / "Figure08_realistic_maps.csv")
    data["s1"] = data["s1"].round(6)
    data["s2"] = data["s2"].round(6)
    grid = np.round(GRID, 6)
    names = ("common_LLI_plus_dLAMn", "same_LLI_LAMn_trajectory")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    maximum = data[data.family.isin(names)].qex_norm.max()
    for ax, name, title in zip(
        axs, names, ("Common LLI + $\\Delta$LAM$_n$", "Same LLI+LAM$_n$ path")
    ):
        z = data[data.family == name].pivot(index="s1", columns="s2", values="qex_norm")
        im = ax.imshow(
            z.loc[grid, grid].to_numpy(),
            origin="lower",
            extent=[0, 25, 0, 25],
            vmin=0,
            vmax=maximum,
            cmap="viridis",
            aspect="auto",
        )
        ax.set(title=title)
    axs[0].set(xlabel="Additional LAM$_n$ in cell 2 (%)", ylabel="Common LLI (%)")
    axs[1].set(xlabel="Cell 2 severity (%)", ylabel="Cell 1 severity (%)")
    fig.colorbar(im, ax=axs, label=r"$q_{\rm excess}$")
    for ax, label in zip(axs, "ab"):
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
            color="white",
        )
    for ext in ("pdf", "png"):
        fig.savefig(
            SUP / f"FigureS08_integrated_heterogeneity.{ext}",
            bbox_inches="tight",
            dpi=320,
        )
    plt.close(fig)


if __name__ == "__main__":
    prepare_output()
    lamp_map()
    integrated_map()
