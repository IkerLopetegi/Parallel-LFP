"""Generate V28 supplementary maps S6 and S8 from the corrected reduced model."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfp_parallel import model as m
from analysis.common import OUT, SUP, prepare_output
from analysis.np_sensitivity import base_at_np, design_np

GRID = np.linspace(0, 0.25, 11)


def capacity_cell(base, lli, lamp):
    p = m.make_mixed_degraded_cell(base, lli=lli, lamp=lamp, kinetic_coupling="decoupled")
    return p, m.lowrate_capacity(p)


def lamp_map():
    base = m.get_reference_params()
    previous = OUT / "FigureS06_LAMp_NP_maps.csv"
    rows = []
    if previous.exists():
        rows = pd.read_csv(previous).to_dict("records")
        for row in rows:
            if "did not reach voltage cutoff" in row["status"]:
                row["status"] = "electrode_bound_first"
    for name, design in (("reference", base), ("NP_1.20", base_at_np(base, 1.20))):
        for lli in GRID:
            a, qa = capacity_cell(design, lli, 0)
            for lamp in GRID:
                if any(r["design"] == name and np.isclose(r["LLI"], lli) and
                       np.isclose(r["dLAMp"], lamp) for r in rows):
                    continue
                b, qb = capacity_cell(design, lli, lamp)
                try:
                    sim = m.simulate_ocvr_pair_fast_stateR(
                        a, b, 1.0, "discharge", 3.35, dq_frac=1e-3,
                        cap1=qa, cap2=qb)
                    met = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
                    peak, status = met["M_peak"], "voltage_cutoff"
                except m.NumericalError as exc:
                    if "did not reach voltage cutoff" not in str(exc):
                        raise
                    peak, status = np.nan, "electrode_bound_first"
                rows.append(dict(design=name, design_NP=design_np(design),
                                 LLI=lli, dLAMp=lamp, M_peak=peak, status=status))
                pd.DataFrame(rows).to_csv(previous, index=False)
        print(f"Finished {name}", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(previous, index=False)
    data["LLI"] = data["LLI"].round(6)
    data["dLAMp"] = data["dLAMp"].round(6)
    grid = np.round(GRID, 6)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for ax, name, title in zip(axs, ("reference", "NP_1.20"),
                               ("Reference balance", "N/P = 1.20 (positive loading)")):
        z = data[data.design == name].pivot(index="LLI", columns="dLAMp", values="M_peak")
        z = z.loc[grid, grid].to_numpy()
        image = ax.imshow(np.ma.masked_invalid(z), origin="lower", extent=[0, 25, 0, 25],
                          vmin=0, vmax=1.05, cmap="viridis", aspect="auto")
        ax.set(title=title, xlabel="Additional LAM$_p$ (%)", ylabel="Common LLI (%)")
    fig.colorbar(image, ax=axs, label=r"$M_{\rm peak}$")
    fig.text(.5, -.025, "White cells: electrode bound reached before voltage cutoff",
             ha="center", fontsize=7)
    for ax, label in zip(axs, "ab"):
        ax.text(.02, .98, label, transform=ax.transAxes,
                va="top", fontweight="bold", color="white")
    for ext in ("pdf", "png"):
        fig.savefig(SUP / f"FigureS06_LAMp_NP_maps.{ext}", bbox_inches="tight", dpi=320)
    plt.close(fig)
    print(data.groupby("design").status.value_counts().to_string(), flush=True)


def integrated_map():
    data = pd.read_csv(OUT / "Figure08_realistic_maps.csv")
    data["s1"] = data["s1"].round(6)
    data["s2"] = data["s2"].round(6)
    grid = np.round(GRID, 6)
    names = ("common_LLI_plus_dLAMn", "same_LLI_LAMn_trajectory")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    maximum = data[data.family.isin(names)].qex_norm.max()
    for ax, name, title in zip(axs, names,
                               ("Common LLI + $\\Delta$LAM$_n$", "Same LLI+LAM$_n$ path")):
        z = data[data.family == name].pivot(index="s1", columns="s2", values="qex_norm")
        im = ax.imshow(z.loc[grid, grid].to_numpy(), origin="lower", extent=[0, 25, 0, 25],
                       vmin=0, vmax=maximum, cmap="viridis", aspect="auto")
        ax.set(title=title)
    axs[0].set(xlabel="Additional LAM$_n$ in cell 2 (%)", ylabel="Common LLI (%)")
    axs[1].set(xlabel="Cell 2 severity (%)", ylabel="Cell 1 severity (%)")
    fig.colorbar(im, ax=axs, label=r"$q_{\rm excess}$")
    for ax, label in zip(axs, "ab"):
        ax.text(.02, .98, label, transform=ax.transAxes,
                va="top", fontweight="bold", color="white")
    for ext in ("pdf", "png"):
        fig.savefig(SUP / f"FigureS08_integrated_heterogeneity.{ext}",
                    bbox_inches="tight", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    prepare_output()
    lamp_map()
    integrated_map()
