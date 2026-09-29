from pathlib import Path
import sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m


from analysis.common import OUT, SUP, prepare_output


def main():
    prepare_output()
    base = m.get_reference_params()
    base["neg"]["Ds_model"] = "ecker2015"
    base["neg"]["kinetics"] = "butler_volmer"

    cases = {
        "common5_dLAMn5": (
            m.make_mixed_degraded_cell(base, 0.05, 0, 0, "decoupled"),
            m.make_mixed_degraded_cell(base, 0.05, 0.05, 0, "decoupled"),
            "Common 5% LLI; cell 2 +5% LAM$_n$",
        ),
        "trajectory5_vs10": (
            m.make_mixed_degraded_cell(base, 0.05, 0.05, 0, "decoupled"),
            m.make_mixed_degraded_cell(base, 0.10, 0.10, 0, "decoupled"),
            "5% LLI+LAM$_n$ vs 10% LLI+LAM$_n$",
        ),
    }

    runs = [("common5_dLAMn5", 0.5), ("common5_dLAMn5", 1.0), ("trajectory5_vs10", 1.0)]
    summary = []
    histories = {}
    for key, cr in runs:
        p1, p2, label = cases[key]
        y, _ = m.init_parallel_at_common_ocv([p1, p2], 3.30, 0.4)
        sim = m.simulate_cc_halfcycle(
            [p1, p2],
            y,
            cr,
            "charge",
            max_step=8,
            rtol=1e-5,
            atol=1e-7,
        )
        mismatch = np.abs(sim["I"][:, 0] - sim["I"][:, 1]) / abs(sim["Iapp"])
        kp = int(np.argmax(mismatch))
        rec = int(np.argmax(np.abs(sim["I"][kp]))) + 1
        slices = []
        k = 0
        for p in [p1, p2]:
            slices.append(slice(k, k + p["Nstate"]))
            k += p["Nstate"]
        long = []
        for it, (tt, yy, V) in enumerate(zip(sim["t"], sim["y"], sim["V"])):
            for cell, (sl, p) in enumerate(zip(slices, [p1, p2]), 1):
                e = m.cell_at_voltage(yy[sl], p, float(V))
                long.append(
                    dict(
                        t_s=tt,
                        cell=cell,
                        V_cell=V,
                        I=sim["I"][it, cell - 1],
                        I_fraction=sim["I"][it, cell - 1] / abs(sim["Iapp"]),
                        Un=e["Un"],
                        eta_n=e["eta_n"],
                        Eneg=e["Eneg"],
                        xn_surface=e["xn_surface"],
                    )
                )
        df = pd.DataFrame(long)
        histories[(key, cr)] = (sim, df, label)
        for cell in [1, 2]:
            z = df[df.cell == cell]
            summary.append(
                dict(
                    case=key,
                    C_rate=cr,
                    cell=cell,
                    M_peak=float(mismatch.max()),
                    peak_recipient=rec,
                    min_Eneg_V=float(z.Eneg.min()),
                    min_eta_n_V=float(z.eta_n.min()),
                    max_xn_surface=float(z.xn_surface.max()),
                    duration_min=float(sim["t"][-1] / 60),
                )
            )
        df.to_csv(SUP / f"FigureS10_{key}_{cr:.1f}C.csv", index=False)

    S = pd.DataFrame(summary)
    S.to_csv(SUP / "TableS06_negative_electrode_potential.csv", index=False)

    fig, axs = plt.subplots(
        2, 2, figsize=(7.05, 5.0), constrained_layout=True, sharex="col"
    )
    for col, key in enumerate(["common5_dLAMn5", "trajectory5_vs10"]):
        sim, df, label = histories[(key, 1.0)]
        ax = axs[0, col]
        for cell in [1, 2]:
            z = df[df.cell == cell]
            ax.plot(z.t_s / 60, z.I_fraction, label=f"Cell {cell}")
        ax.axhline(0.5, ls=":", lw=1)
        ax.set_ylabel("$I_i/|I_{app}|$")
        ax.set_title(label, fontsize=8.4)
        ax.legend(frameon=False)
        ax.text(
            0.015,
            0.98,
            "ab"[col],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=9.3,
        )
        ax = axs[1, col]
        for cell in [1, 2]:
            z = df[df.cell == cell]
            ax.plot(z.t_s / 60, 1000 * z.Eneg, label=f"Cell {cell}")
        ax.axhline(0, ls="--", lw=1)
        ax.set(xlabel="Time (min)", ylabel="$\\phi_{s,n}-\\phi_{e,n}$ (mV)")
        ax.legend(frameon=False)
        ax.text(
            0.015,
            0.98,
            "cd"[col],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=9.3,
        )
    fig.savefig(SUP / "FigureS10_negative_electrode_potential.pdf", bbox_inches="tight")
    fig.savefig(
        SUP / "FigureS10_negative_electrode_potential.png", bbox_inches="tight", dpi=320
    )
    plt.close(fig)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
