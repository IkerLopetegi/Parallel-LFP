"""Generate final paper and SI figures from the saved data and model recipes.

Stages core and diffusivity rerun simulations; saved, si and graphical replot data.
"""
from pathlib import Path
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from analysis.common import OUT, SUP, prepare_output
from lfp_parallel import model as m
from analysis import core_studies as mainplots
from analysis.figure02_hysteresis import main as render_fig02
from analysis.robustness import plot as plot_robustness
from analysis.threshold_figures import main as render_thresholds
from analysis.supplementary_maps import lamp_map, integrated_map
from analysis.np_sensitivity import design_np

BLUE, ORANGE, GREEN, RED = "#0072B2", "#D55E00", "#009E73", "#CC79A7"


def save(fig, path, dpi=320):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def render_core_main_figures():
    base = m.get_reference_params()
    render_fig02([])
    sn = mainplots.figure3_lowrate(base)
    mainplots.figure4_dynamics(base, sn)
    mainplots.figure5_model_fidelity(base, sn)


def render_np_figure():
    base = m.get_reference_params()
    # Figure 7: publication plot from the completed, checked sweep.
    data = pd.read_csv(OUT / "Figure07_NP_design_sensitivity.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    colors = {"LLI": BLUE, "LAMn": ORANGE, "LAMp": GREEN}
    for mode, rows in data.groupby("mode", sort=False):
        rows = rows.sort_values("NP_ratio")
        ax[0].plot(rows.NP_ratio, rows.M_peak, "o-", ms=3,
                   label={"LLI":"LLI","LAMn":r"LAM$_n$","LAMp":r"LAM$_p$"}[mode], color=colors[mode])
        ax[1].plot(rows.NP_ratio, rows.qex_norm, "o-", ms=3,
                   label={"LLI":"LLI","LAMn":r"LAM$_n$","LAMp":r"LAM$_p$"}[mode], color=colors[mode])
    for a in ax:
        a.set_xlabel("Beginning-of-life N/P ratio")
        a.axvline(design_np(base), ls=":", color="0.45", lw=0.9)
        a.legend(frameon=False, fontsize=8)
    ax[0].set_ylabel(r"$M_{\mathrm{peak}}$")
    ax[1].set_ylabel(r"$q_{\mathrm{excess}}$")
    for a, label in zip(ax, "ab"):
        a.text(.02,1.02,label,transform=a.transAxes,va="bottom",fontweight="bold",clip_on=False)
    save(fig, OUT / "Figure07_NP_design_sensitivity")


def render_saved_main_figures():
    base = m.get_reference_params()
    print("Plotting robustness, N/P, threshold, and resistance figures", flush=True)
    robustness = pd.read_csv(OUT / "Figure06_parameter_robustness.csv")
    if len(robustness) != 25 or robustness.M_peak.isna().any():
        raise RuntimeError("The saved robustness table is incomplete")
    plot_robustness(robustness)

    render_np_figure()

    # Figure 9: draw both panels from the existing, complete sweep tables.
    tab = pd.read_csv(OUT / "Figure09_surface_area_coupling.csv")
    contact = pd.read_csv(OUT / "Figure09_contact_map.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.1, 5.15), constrained_layout=True)
    styles = (("decoupled", BLUE, "-"), ("coupled", ORANGE, "--"))
    for name, color, ls in styles:
        z = tab[tab.coupling == name].sort_values("dLAMn")
        x = z.dLAMn * 100
        ax[0, 0].plot(x, z.ASR2 / z.ASR1, color=color, ls=ls, label=name)
        ax[0, 1].plot(x, z.M_peak, color=color, ls=ls, label=name)
        ax[1, 0].plot(x, z.qex_norm, color=color, ls=ls, label=name)
    ax[0, 0].set(xlabel="Additional LAM$_n$ in cell 2 (%)",
                 ylabel=r"$R_{2,\mathrm{eff}}/R_{1,\mathrm{eff}}$")
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[0, 1].set(xlabel="Additional LAM$_n$ in cell 2 (%)",
                 ylabel=r"$M_{\mathrm{peak}}$")
    ax[1, 0].set(xlabel="Additional LAM$_n$ in cell 2 (%)",
                 ylabel=r"$q_{\mathrm{excess}}$")
    z = contact.pivot(index="contact_fraction_R0", columns="dLAMn",
                      values="qex_norm").sort_index()
    im = ax[1, 1].imshow(z.to_numpy(), origin="lower", aspect="auto",
                         extent=[0, 25, 0, 50], cmap="magma")
    fig.colorbar(im, ax=ax[1, 1], pad=0.015,
                 label=r"$q_{\mathrm{excess}}$")
    ax[1, 1].set(xlabel="Additional LAM$_n$ in cell 2 (%)",
                 ylabel="Added contact resistance / fresh ASR (%)")
    for a, label in zip(ax.flat, "abcd"):
        a.text(.02, .98, label, transform=a.transAxes, va="top",
               fontweight="bold")
    save(fig, OUT / "Figure09_resistance_surface_area")

    # Verified threshold trajectories create Fig. 8 and SI Fig. S7.
    render_thresholds()


def render_si_from_tables():
    print("Reformatting SI plots from saved numerical tables", flush=True)
    # S1 grid convergence.
    grid = pd.read_csv(SUP / "TableS01_grid_convergence.csv")
    labels = ["/".join(str(int(v)) for v in row)
              for row in grid[["Nneg", "Nsep", "Npos", "Npsd", "Nr_neg"]].values]
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.8), constrained_layout=True)
    ax[0].plot(range(len(grid)), grid.M_peak, "o-")
    ax[1].plot(range(len(grid)), grid.qex_norm, "o-")
    for a in ax:
        a.set_xticks(range(len(grid)), labels, rotation=28, ha="right")
        a.set_xlabel(r"$N_n/N_s/N_p/N_R/N_r$")
    ax[0].set_ylabel(r"$M_{\mathrm{peak}}$")
    ax[1].set_ylabel(r"$q_{\mathrm{excess}}$")
    save(fig, SUP / "FigureS01_grid_convergence")

    # S2 particle-size-dependent regular-solution verification.
    base = m.get_reference_params()
    x = np.linspace(.001, .999, 1200)
    radii = np.array([20, 30, 50, 100, 200]) * 1e-9
    om = m.omega_rt_from_particle_radius(radii)
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    ax[0].plot(radii * 1e9, om, "o-")
    ax[0].axhline(2, color="0.55", ls=":", lw=.9)
    ax[0].set(xlabel="Equivalent spherical radius (nm)",
              ylabel=r"$\Omega/k_BT$")
    for radius in radii[1:]:
        p = copy.deepcopy(base)
        p["disc"]["Npsd"] = 1
        p["lfp"]["R_mean"] = float(radius)
        p["lfp"]["psd_cv"] = 0
        p = m.update_derived_params(p)
        p["QLi_fresh"], p["QLi"], p["Q_nominal_ref"] = (
            base["QLi_fresh"], base["QLi"], base["Q_nominal_ref"])
        ax[1].plot(x, m.lfp_ocp_zk0d(x, np.ones_like(x), p),
                   label=f"{radius*1e9:.0f} nm charge")
        ax[1].plot(x, m.lfp_ocp_zk0d(x, -np.ones_like(x), p), ls="--")
    ax[1].set(xlabel="LFP lithiation, $x_p$", ylabel="LFP OCP (V)")
    ax[1].legend(frameon=False, fontsize=7, ncol=2)
    for a, label in zip(ax, "ab"):
        a.text(.02, .98, label, transform=a.transAxes, va="top",
               fontweight="bold")
    save(fig, SUP / "FigureS02_ZK_verification")

    # S3 capacity-protocol sensitivity.
    data = pd.read_csv(SUP / "TableS03_capacity_protocol_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(4.6, 2.7), constrained_layout=True)
    ax.plot(data.C_rate, data.matched_LAMn * 100, "o-")
    ax.set(xlabel="Capacity-characterization rate (C)",
           ylabel="Matched LAM$_n$ severity (%)")
    save(fig, SUP / "FigureS03_capacity_protocol_sensitivity")

    # S4 full-model C-rate sensitivity.
    data = pd.read_csv(SUP / "TableS06_Crate_sensitivity.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.8), constrained_layout=True)
    ax[0].plot(data.C_rate, data.M_peak, "o-")
    ax[1].plot(data.C_rate, data.qex_norm, "o-")
    ax[0].set(xlabel="C-rate", ylabel=r"$M_{\mathrm{peak}}$")
    ax[1].set(xlabel="C-rate", ylabel=r"$q_{\mathrm{excess}}$")
    save(fig, SUP / "FigureS04_Crate_sensitivity")

    # S6 and S8 use cached, complete map results.
    lamp_map()
    integrated_map()

    # S9: same-trajectory full-model peak and excess-throughput maps.
    data = pd.read_csv(SUP / "FigureS09_same_mixed_trajectory.csv")
    peak = data.pivot(index="s1", columns="s2", values="M_peak")
    qex = data.pivot(index="s1", columns="s2", values="qex_norm")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for a, table, label in zip(ax, (peak, qex),
                               (r"$M_{\mathrm{peak}}$", r"$q_{\mathrm{excess}}$")):
        im = a.imshow(table.to_numpy(), origin="lower", extent=[0, 25, 0, 25],
                      aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=a, pad=.015, label=label)
    ax[0].set(xlabel="Cell 2 severity (%)", ylabel="Cell 1 severity (%)")
    ax[1].set(xlabel="Cell 2 severity (%)", ylabel="Cell 1 severity (%)")
    for a, label in zip(ax, "ab"):
        a.text(.02, .98, label, transform=a.transAxes, va="top",
               fontweight="bold", color="white")
    save(fig, SUP / "FigureS09_same_mixed_trajectory")

    # S10: plot the stored 1C branch-current and negative-electrode potentials.
    names = ("FigureS10_common5_dLAMn5_1.0C.csv",
             "FigureS10_trajectory5_vs10_1.0C.csv")
    titles = ("Common 5% LLI; cell 2 +5% LAM$_n$",
              "5% LLI+LAM$_n$ vs 10% LLI+LAM$_n$")
    fig, ax = plt.subplots(2, 2, figsize=(7.05, 5.0), constrained_layout=True,
                           sharex="col")
    for col, (name, title) in enumerate(zip(names, titles)):
        table = pd.read_csv(SUP / name)
        for cell, color in ((1, BLUE), (2, ORANGE)):
            z = table[table.cell == cell]
            ax[0, col].plot(z.t_s / 60, z.I_fraction, color=color,
                            label=f"Cell {cell}")
            ax[1, col].plot(z.t_s / 60, z.Eneg * 1000, color=color,
                            label=f"Cell {cell}")
        ax[0, col].axhline(.5, ls=":", color="0.45", lw=.8)
        ax[0, col].set(title=title, ylabel=r"$I_i/|I_{\mathrm{app}}|$")
        ax[1, col].axhline(0, ls="--", color="0.4", lw=.8)
        ax[1, col].set(xlabel="Time (min)",
                       ylabel=r"$\phi_{s,n}-\phi_{e,n}$ (mV)")
        ax[0, col].legend(frameon=False, fontsize=7)
        ax[1, col].legend(frameon=False, fontsize=7)
    for a, label in zip(ax.flat, "abcd"):
        a.text(.02, .98, label, transform=a.transAxes, va="top",
               fontweight="bold")
    save(fig, SUP / "FigureS10_negative_electrode_potential")

    # S11: replot the stored full MP-SPMe trajectories.
    data = pd.read_csv(SUP / "full_model_electrode_trajectories.csv")
    x = data.time_s / 60
    fig, ax = plt.subplots(2, 2, figsize=(7.1, 4.65), constrained_layout=True)
    for branch, color, label in ((1, BLUE, "10% LLI"),
                                 (2, ORANGE, "22.94% LAM$_n$")):
        ax[0, 0].plot(x, data[f"xn_surface_{branch}"], color=color, label=label)
        ax[0, 1].plot(x, data[f"xp_volume_mean_{branch}"], color=color, label=label)
        ax[1, 0].plot(x, data[f"Un_{branch}"], color=color, label=label)
        ax[1, 1].plot(x, data[f"Up_volume_mean_{branch}"], color=color, label=label)
    for a, title in zip(ax.flat, ("Graphite surface lithiation", "Mean LFP lithiation",
                                  "Graphite OCP (V)", "Mean LFP particle OCP (V)")):
        a.set(xlabel="Time (min)", ylabel=title)
    ax[0, 0].legend(frameon=False, fontsize=7)
    for a, label in zip(ax.flat, "abcd"):
        a.text(.02, .98, label, transform=a.transAxes, va="top",
               fontweight="bold")
    save(fig, SUP / "FigureS11_fullmodel_electrode_trajectories")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "core", "saved", "si", "diffusivity", "graphical"), default="all")
    stage = parser.parse_args().stage
    prepare_output()
    if stage in ("all", "core"):
        render_core_main_figures()
    if stage in ("all", "saved"):
        render_saved_main_figures()
    if stage in ("all", "si"):
        render_si_from_tables()
    if stage in ("all", "diffusivity"):
        # Rerun the charge/discharge × constant/Ecker cases at 0.5C and 1C
        # needed for the SI sensitivity figure.
        from analysis.diffusivity_sensitivity import main as render_diffusivity
        render_diffusivity()
    if stage in ("all", "graphical"):
        sys.path.insert(0, str(HERE))
        exec(compile((HERE / "graphical_abstract.py").read_text(),
                     str(HERE / "graphical_abstract.py"), "exec"),
             {"__name__": "__main__", "__file__": str(HERE / "graphical_abstract.py")})
    print("Reformatted manuscript and SI figure assets", flush=True)


if __name__ == "__main__":
    main()
