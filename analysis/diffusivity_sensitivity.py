from pathlib import Path
import sys, copy, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m


def panel(ax, label):
    ax.text(
        0.015,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
        fontsize=9.4,
    )


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=320)
    plt.close(fig)


def qex_norm(sim, p1, p2):
    met = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
    q1 = m.lowrate_capacity(p1, 0.05)
    q2 = m.lowrate_capacity(p2, 0.05)
    return met, met["Q_excess_Ahm2"] / (0.5 * (q1 + q2))


from analysis.common import OUT, SUP, prepare_output


def main():
    prepare_output()
    base = m.get_reference_params()
    p_lli = m.make_degraded_cell(base, "LLI", 0.10)
    p_lamn, sn, _ = m.match_capacity_lowrate_ocvr(
        base, p_lli, "LAMn", bounds=(0, 0.35), C_rate=0.05
    )
    p_lamp, sp, _ = m.match_capacity_lowrate_ocvr(
        base, p_lli, "LAMp", bounds=(0, 0.50), C_rate=0.05
    )
    print("matched severities", sn, sp, flush=True)

    rates = [0.5, 1.0]
    models = [("Constant $D_{s,n}$", "constant"), ("Ecker $D_{s,n}(x)$", "ecker2015")]
    rows = []
    sims = {}
    for direction, p_other, V0 in [
        ("charge", p_lamn, 3.30),
        ("discharge", p_lamp, 3.35),
    ]:
        for Cr in rates:
            for label, dsmodel in models:
                a = copy.deepcopy(p_lli)
                b = copy.deepcopy(p_other)
                a["neg"]["Ds_model"] = dsmodel
                b["neg"]["Ds_model"] = dsmodel
                soc_guess = 0.4 if direction == "charge" else 0.75
                y, _ = m.init_parallel_at_common_ocv([a, b], V0, soc_guess)
                max_step = 8
                t0 = time.time()
                sim = m.simulate_cc_halfcycle(
                    [a, b], y, Cr, direction, max_step=max_step, rtol=8e-5, atol=7e-7
                )
                met, qex = qex_norm(sim, a, b)
                mismatch = np.abs(sim["I"][:, 0] - sim["I"][:, 1]) / abs(sim["Iapp"])
                k = int(np.argmax(mismatch))
                recipient = int(np.argmax(np.abs(sim["I"][k, :]))) + 1
                rows.append(
                    dict(
                        direction=direction,
                        C_rate=Cr,
                        diffusivity_model=dsmodel,
                        M_peak=met["M_peak"],
                        M_rms=met["M_rms"],
                        qex_norm=qex,
                        peak_recipient_cell=recipient,
                        duration_min=sim["t"][-1] / 60,
                        runtime_s=time.time() - t0,
                        success=sim["success"],
                    )
                )
                sims[(direction, Cr, dsmodel)] = sim
                print(rows[-1], flush=True)

    T = pd.DataFrame(rows)
    T.to_csv(SUP / "TableS03_graphite_diffusivity_sensitivity.csv", index=False)

    # Figure S5: physical diffusivity comparison, 1C current-sharing histories, and moderate-rate summary.
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)
    ax = axs[0, 0]
    x = np.linspace(0.02, 0.98, 300)
    ax.semilogy(x, np.full_like(x, base["neg"]["Ds"]), label="Constant $D_{s,n}$")
    ax.semilogy(
        x, m.graphite_diffusivity_ecker2015(x, base), label="Ecker $D_{s,n}(x)$"
    )
    ax.set(xlabel="Graphite stoichiometry $x_n$", ylabel="$D_{s,n}$ (m$^2$ s$^{-1}$)")
    ax.legend(frameon=False)
    panel(ax, "a")

    for ax, direction, title in [
        (axs[0, 1], "charge", "Charge, 1C"),
        (axs[1, 0], "discharge", "Discharge, 1C"),
    ]:
        for label, dsmodel in models:
            sim = sims[(direction, 1.0, dsmodel)]
            ax.plot(
                sim["t"] / 60,
                np.abs(sim["I"][:, 0] - sim["I"][:, 1]) / abs(sim["Iapp"]),
                label=label,
            )
        ax.set(xlabel="Time (min)", ylabel="$|I_1-I_2|/|I_{app}|$")
        ax.text(
            0.98,
            0.94,
            title,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
        )
        ax.legend(frameon=False)
    panel(axs[0, 1], "b")
    panel(axs[1, 0], "c")

    ax = axs[1, 1]
    markers = {"charge": "o", "discharge": "s"}
    for direction in ["charge", "discharge"]:
        for label, dsmodel in models:
            z = T[
                (T.direction == direction) & (T.diffusivity_model == dsmodel)
            ].sort_values("C_rate")
            ls = "-" if dsmodel == "constant" else "--"
            lab = ("Charge" if direction == "charge" else "Discharge") + (
                ", constant" if dsmodel == "constant" else ", Ecker"
            )
            ax.plot(z.C_rate, z.M_peak, marker=markers[direction], ls=ls, label=lab)
    ax.set(xlabel="C-rate", ylabel="$M_{peak}$", xticks=rates)
    ax.legend(frameon=False, ncol=1, fontsize=6.6)
    panel(ax, "d")

    save(fig, SUP / "FigureS05_graphite_diffusivity_sensitivity")
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
