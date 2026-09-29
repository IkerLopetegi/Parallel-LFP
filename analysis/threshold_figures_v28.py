"""Render V28 full-model threshold plots from verified sampled-grid outputs."""

from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import OUT, SUP, prepare_output
from lfp_parallel import model as m
from analysis import section37_full_threshold_refinement as threshold_model

SOURCE = OUT / "thresholds"
FAMILIES = threshold_model.FAMILIES
BACKGROUNDS = threshold_model.BACKGROUNDS
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def verified_data():
    audit = json.loads((SOURCE / "reproduction_audit.json").read_text())
    digest = hashlib.sha256(Path(m.__file__).read_bytes() +
                            Path(threshold_model.__file__).read_bytes()).hexdigest()
    if digest != audit["model_and_script_sha256"] or audit["rows"] != 16:
        raise RuntimeError("Full-model threshold results are from different source")
    summary = pd.read_csv(SOURCE / "Section37_full_thresholds_combined.csv")
    files = list(SOURCE.glob("Section37_full_points_*.csv"))
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if len(files) != 8 or data.duplicated(["family", "background", "delta"]).any():
        raise RuntimeError("Missing or duplicate threshold trajectories")
    if set(data.family) != set(FAMILIES) or set(data.background) != set(BACKGROUNDS):
        raise RuntimeError("Full-model threshold trajectory matrix is incomplete")
    for _, row in summary[summary.status == "reached"].iterrows():
        points = data[(data.family == row.family) & (data.background == row.background)]
        first = points[points.M_peak >= row.target].sort_values("delta").iloc[0]
        if not np.isclose(first.delta, row.delta_crit_sampled):
            raise RuntimeError("Threshold is not the first sampled crossing")
    return data, summary


def save(fig, name, supplementary=False):
    target = SUP if supplementary else OUT
    for ext in ("pdf", "png"):
        fig.savefig(target / f"{name}.{ext}", bbox_inches="tight", dpi=320)
    plt.close(fig)


def plot_main(data, summary):
    fig, axs = plt.subplots(1, 3, figsize=(8.7, 2.8), constrained_layout=True)
    labels = ("Common LLI + $\\Delta$LAM$_n$", "Same LLI+LAM$_n$ path")
    for ax, family, title in zip(axs[:2], FAMILIES, labels):
        for bg, color in zip(BACKGROUNDS, COLORS):
            z = data[(data.family == family) & (data.background == bg)].sort_values("delta")
            ax.plot(z.delta * 100, z.M_peak, lw=1.25, color=color,
                    label=f"{bg * 100:g}% background")
        for y, style in ((0.8, "--"), (0.95, ":")):
            ax.axhline(y, color="0.35", ls=style, lw=0.8)
        ax.set(xlabel="Additional severity (pp)", ylabel=r"$M_{\rm peak}$",
               title=title, ylim=(0, 1.09), xlim=(0, 7))
    axs[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    for family, marker, color, name in zip(FAMILIES, ("o", "s"),
                                           ("#0072B2", "#D55E00"),
                                           ("Common LLI", "Same trajectory")):
        for target, ls in ((0.8, "-"), (0.95, "--")):
            z = summary[(summary.family == family) & (summary.target == target)]
            axs[2].plot(z.background * 100, z.capacity_difference_pct.abs(),
                        marker=marker, ls=ls, color=color,
                        label=f"{name}, {target:.2f}")
    axs[2].set(xlabel="Background severity (%)",
               ylabel=r"$|\Delta Q|/\bar Q$ (%) at threshold",
               yscale="log", title="Capacity difference", xticks=[0, 5, 10, 15])
    axs[2].legend(frameon=False, fontsize=5.9)
    for ax, label in zip(axs, "abc"):
        ax.text(.02, .98, label, transform=ax.transAxes,
                va="top", fontweight="bold")
    save(fig, "Figure08_threshold_heterogeneity")


def plot_summary(summary):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.85), constrained_layout=True)
    for ax, family, title in zip(axs, FAMILIES,
                                 ("Common LLI + $\\Delta$LAM$_n$", "Same trajectory")):
        for target, style in ((0.8, "o-"), (0.95, "s--")):
            z = summary[(summary.family == family) & (summary.target == target)]
            ax.plot(z.background * 100, z.delta_crit_sampled * 100, style,
                    label=f"$M_{{\\rm peak}}\\geq{target:.2f}$")
        ax.set(title=title, xlabel="Background severity (%)",
               ylabel="First sampled heterogeneity (percentage points)",
               xticks=[0, 5, 10, 15])
        ax.legend(frameon=False)
    for ax, label in zip(axs, "ab"):
        ax.text(.02, .98, label, transform=ax.transAxes,
                va="top", fontweight="bold")
    save(fig, "FigureS07_threshold_summary", True)


def main():
    prepare_output()
    data, summary = verified_data()
    plot_main(data, summary)
    plot_summary(summary)
    print(f"Verified {len(data)} full-model points and {len(summary)} threshold rows")


if __name__ == "__main__":
    main()
