"""Focused full-model checks for the September 2026 manuscript revision.

Run from a repository checkout with ``python analysis/manuscript_revision_checks.py``.
Outputs are saved under ``results/supplementary/``.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lfp_parallel import model as m
from analysis.np_sensitivity_anode_loading import base_at_np as anode_base_at_np
from analysis.np_sensitivity import base_at_np as cathode_base_at_np

OUT = ROOT / "results" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)


def simulate_pair(cells, initial_voltage, direction):
    y0, _ = m.init_parallel_at_common_ocv(
        cells, initial_voltage, 0.4 if direction == "charge" else 0.75
    )
    return m.simulate_cc_halfcycle(
        cells, y0, 1.0, direction, max_step=8, rtol=8e-5, atol=7e-7
    )


def matched_design_check(base):
    rows = []
    for path, make_design in (("positive loading", cathode_base_at_np),
                              ("negative loading", anode_base_at_np)):
        for ratio in (1.0, 1.2):
            design = make_design(base, ratio)
            # Use the same nominal current reference for the C/20 capacity
            # surrogate on both loading paths. The 1C full-model current uses
            # the sum of the two characterized branch capacities.
            design["Q_nominal_ref"] = base["Q_nominal_ref"]
            cells = [design, m.make_degraded_cell(design, "LAMp", 0.20)]
            s = simulate_pair(cells, 3.35, "discharge")
            met = m.compute_current_metrics(s["t"], s["I"], s["Iapp"])
            qbar = np.mean([m.lowrate_capacity(p) for p in cells])
            row = dict(path=path, NP=design["Qn"] / design["Qp"],
                       L_pos_um=design["geom"]["L_pos"]*1e6,
                       L_neg_um=design["geom"]["L_neg"]*1e6, LLI_common=0.0,
                       LAMp_difference=0.20, M_peak=met["M_peak"],
                       M_rms=met["M_rms"], q_excess=met["Q_excess_Ahm2"] / qbar,
                       duration_min=s["t"][-1] / 60)
            rows.append(row)
            print("matched design", row, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "matched_NP_design.csv", index=False)


def full_model_electrode_check(base):
    lli = m.make_degraded_cell(base, "LLI", 0.10)
    lamn, severity, _ = m.match_capacity_lowrate_ocvr(
        base, lli, "LAMn", bounds=(0, 0.5), C_rate=0.05
    )
    cells = [lli, lamn]
    s = simulate_pair(cells, 3.30, "charge")
    applied = lambda _t: s["Iapp"]
    rows = []
    for t, y in zip(s["t"], s["y"]):
        v, currents, ev = m.solve_parallel_voltage(t, y, cells, applied)
        row = dict(time_s=t, V=v, I1=currents[0], I2=currents[1],
                   mismatch=abs(currents[0] - currents[1]) / abs(s["Iapp"]))
        offset = 0
        for branch, (p, e) in enumerate(zip(cells, ev), start=1):
            state = m.unpack_cell_state(y[offset:offset + p["Nstate"]], p)
            offset += p["Nstate"]
            w = p["lfp"]["psd"]["w_volume"]
            row[f"xn_surface_{branch}"] = e["xn_surface"]
            row[f"xp_volume_mean_{branch}"] = np.mean(state["xp"] @ w)
            row[f"Un_{branch}"] = e["Un"]
            # Diagnostic PSD/through-thickness average, not a branch OCV.
            row[f"Up_volume_mean_{branch}"] = np.mean(e["Up"] @ w)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "full_model_electrode_trajectories.csv", index=False)
    fig, ax = plt.subplots(2, 2, figsize=(7.1, 4.65), constrained_layout=True)
    colors = ("#0072B2", "#D55E00")
    labels = ("10% LLI", f"{severity*100:.2f}% LAM$_n$")
    x = df.time_s / 60
    for k, color, label in zip((1, 2), colors, labels):
        ax[0, 0].plot(x, df[f"xn_surface_{k}"], color=color, label=label)
        ax[0, 1].plot(x, df[f"xp_volume_mean_{k}"], color=color, label=label)
        ax[1, 0].plot(x, df[f"Un_{k}"], color=color, label=label)
        ax[1, 1].plot(x, df[f"Up_volume_mean_{k}"], color=color, label=label)
    ax[0, 0].set_ylabel("Graphite surface lithiation")
    ax[0, 1].set_ylabel("Mean LFP lithiation")
    ax[1, 0].set_ylabel("Graphite OCP (V)")
    ax[1, 1].set_ylabel("Mean LFP particle OCP (V)")
    for a, letter in zip(ax.flat, "abcd"):
        a.set_xlabel("Time (min)")
        a.text(0.03, 0.95, letter, transform=a.transAxes,
               va="top", fontweight="bold")
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "FigureS11_fullmodel_electrode_trajectories.pdf")
    fig.savefig(OUT / "FigureS11_fullmodel_electrode_trajectories.png", dpi=220)
    plt.close(fig)
    print("full-model electrode check", severity, df.iloc[-1].to_dict(), flush=True)


def cutoff_check(base):
    rows = []
    for cutoff in (3.63, 3.64, 3.65):
        p1 = m.make_mixed_degraded_cell(base, 0.10, 0, 0, "decoupled")
        p2 = m.make_mixed_degraded_cell(base, 0.10, 0.0135, 0, "decoupled")
        for p in (p1, p2):
            p["Vmax"] = cutoff
        y0, _ = m.init_parallel_at_common_ocv([p1, p2], 3.30, 0.4)
        s = m.simulate_cc_halfcycle(
            [p1, p2], y0, 1.0, "charge", max_step=4,
            rtol=1e-5, atol=1e-7
        )
        met = m.compute_current_metrics(s["t"], s["I"], s["Iapp"])
        row = dict(Vmax=cutoff, common_LLI=0.10, delta_LAMn=0.0135,
                   M_peak=met["M_peak"], M_rms=met["M_rms"],
                   duration_min=s["t"][-1] / 60)
        rows.append(row)
        print("cutoff", row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "cutoff_sensitivity.csv", index=False)


if __name__ == "__main__":
    p0 = m.get_reference_params()
    full_model_electrode_check(p0)
    matched_design_check(p0)
    cutoff_check(p0)
