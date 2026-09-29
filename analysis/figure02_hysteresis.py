"""Produce V28 Figure 2: isolated 20% degradation, C/2, charge/discharge."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfp_parallel import model as m
from analysis.common import OUT, prepare_output

RATE = 0.5
SEVERITY = 0.20
STEP = 5e-4


def run_case(p, step=STEP):
    charge, discharge = m.simulate_mp0d_cycle_fixed(p, RATE, dq_frac=step)
    if not charge["reached_cutoff"] or not discharge["reached_cutoff"]:
        raise m.NumericalError("Figure 2 half-cycle missed a terminal-voltage cutoff")
    for sim in (charge, discharge):
        if not np.all(np.isfinite(sim["V"])):
            raise m.NumericalError("Nonfinite Figure 2 voltage")
        raw_xn = np.array([m.mp0d_xn_from_xp(x, p) for x in sim["state"]])
        if raw_xn.min() < 0 or raw_xn.max() > 1:
            raise m.NumericalError("Graphite composition outside [0,1]")
    return charge, discharge


def main():
    prepare_output()
    base = m.get_reference_params()
    fresh = run_case(m.make_degraded_cell(base, "fresh", 0))
    colors = {"LLI": "#0072B2", "LAMn": "#D55E00", "LAMp": "#E69F00"}
    fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.92), sharey=True,
                            constrained_layout=True)
    rows = []
    for ax, mode, label in zip(axs, colors, ("LLI", r"LAM$_n$", r"LAM$_p$")):
        p = m.make_degraded_cell(base, mode, SEVERITY)
        ch, ds = run_case(p)
        # Refine one timestep setting to verify that the plotted endpoints
        # and capacity are not determined by the fixed-step discretization.
        ch2, ds2 = run_case(p, STEP / 2)
        for sim, ls, caption in ((fresh[0], "-", "Fresh, charge"),
                                 (fresh[1], "--", "Fresh, discharge")):
            ax.plot(sim["xn"], sim["V"], ls, color="#202020", lw=1.2,
                    label=caption)
        for sim, ls, caption in ((ch, "-", f"20% {label}, charge"),
                                 (ds, "--", f"20% {label}, discharge")):
            ax.plot(sim["xn"], sim["V"], ls, color=colors[mode], lw=1.35,
                    label=caption)
        ax.set(xlabel=r"Bulk graphite lithiation, $x_n$", xlim=(0, 1),
               ylim=(2.48, 3.69))
        ax.legend(frameon=False, fontsize=5.9, loc="best")
        rows.append(dict(mode=mode, severity=SEVERITY, C_rate=RATE,
                         dq_frac=STEP, charge_capacity_Ahm2=ch["capacity_Ahm2"],
                         discharge_capacity_Ahm2=ds["capacity_Ahm2"],
                         half_step_charge_capacity_Ahm2=ch2["capacity_Ahm2"],
                         half_step_discharge_capacity_Ahm2=ds2["capacity_Ahm2"],
                         charge_endpoint_V=ch["V"][-1],
                         discharge_endpoint_V=ds["V"][-1]))
    axs[0].set_ylabel("Terminal voltage (V)")
    for ax, label in zip(axs, "abc"):
        ax.text(0.015, 0.98, label, transform=ax.transAxes,
                va="top", fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure02_degradation_hysteresis.{ext}",
                    bbox_inches="tight", dpi=320)
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "Figure02_metrics.csv", index=False)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
