"""Main-text N/P study: change positive loading at fixed negative electrode/inventory.

N/P uses the reference stoichiometric spans specified in balancing_reference.
This is a design ratio, not Qn/Qp over the full 0--1 composition range.
"""

from pathlib import Path
import argparse
import copy
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m
from analysis.common import OUT, prepare_output


def design_np(p):
    a = p["balancing_reference"]
    return p["Qn"] * (a["xn_100"] - a["xn_0"]) / (p["Qp"] * (a["xp_0"] - a["xp_100"]))


def base_at_np(base, target):
    if target <= 0:
        raise ValueError("N/P must be positive")
    p = copy.deepcopy(base)
    p["geom"]["L_pos"] *= design_np(base) / target
    p = m.update_derived_params(p)
    # Inventory, nominal LLI reference and negative geometry remain fixed.
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=list(np.linspace(0.90, 1.20, 13))
    )
    args = parser.parse_args()
    prepare_output()
    base = m.get_reference_params()
    rows = []
    for ratio in args.ratios:
        p1 = base_at_np(base, ratio)
        fresh = m.lowrate_ocvr_characterization(p1)
        for mode, direction, voltage in [
            ("LLI", "charge", 3.30),
            ("LAMn", "charge", 3.30),
            ("LAMp", "discharge", 3.35),
        ]:
            p2 = m.make_degraded_cell(p1, mode, 0.20)
            y, _ = m.init_parallel_at_common_ocv(
                [p1, p2], voltage, 0.4 if direction == "charge" else 0.75
            )
            sim = m.simulate_cc_halfcycle(
                [p1, p2], y, 1.0, direction, max_step=8, rtol=8e-5, atol=7e-7
            )
            met = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
            row = dict(
                NP_ratio=ratio,
                mode=mode,
                M_peak=met["M_peak"],
                M_rms=met["M_rms"],
                qex_norm=met["Q_excess_Ahm2"]
                / (0.5 * (m.lowrate_capacity(p1) + m.lowrate_capacity(p2))),
                positive_utilization=fresh["xp_low"] - fresh["xp_high"],
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(
                OUT / "Figure07_NP_design_sensitivity.csv", index=False
            )
            print(row, flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    table = pd.DataFrame(rows)
    for mode, g in table.groupby("mode"):
        g = g.sort_values("NP_ratio")
        axs[0].plot(g.NP_ratio, g.M_peak, "o-", label=mode)
        axs[1].plot(g.NP_ratio, g.qex_norm, "o-", label=mode)
    for ax in axs:
        ax.set_xlabel("Beginning-of-life N/P ratio")
        ax.axvline(design_np(base), ls=":", color="gray")
        ax.legend(frameon=False)
    axs[0].set_ylabel(r"$M_{\mathrm{peak}}$")
    axs[1].set_ylabel(r"$q_{\mathrm{excess}}$")
    fig.savefig(OUT / "Figure07_NP_design_sensitivity.pdf")
    fig.savefig(OUT / "Figure07_NP_design_sensitivity.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
