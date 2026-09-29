from pathlib import Path
import sys, copy
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m

COLS = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00"]


def design_np(p):
    a = p["balancing_reference"]
    qn = p["Qn"] * (a["xn_100"] - a["xn_0"])
    qp = p["Qp"] * (a["xp_0"] - a["xp_100"])
    return qn / qp


def base_at_np(base, target):
    p = copy.deepcopy(base)
    np0 = design_np(p)
    # Vary negative coating loading via thickness, a common cell-design lever.
    p["geom"]["L_neg"] = base["geom"]["L_neg"] * target / np0
    p = m.update_derived_params(p)
    # Cathode loading/lithium inventory is fixed as N/P is changed by anode loading.
    p["QLi_fresh"] = base["QLi_fresh"]
    p["QLi"] = p["QLi_fresh"]
    # Define C rate and percentage LLI against each design's own fresh C/20 usable capacity.
    p["Q_nominal_ref"] = base["Q_nominal_ref"]
    for _ in range(2):
        q = m.lowrate_capacity(p, 0.05)
        p["Q_nominal_ref"] = q * 3600.0
    return p


def metrics(sim, q1, q2):
    met = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
    met["qex_norm"] = met["Q_excess_Ahm2"] / (0.5 * (q1 + q2))
    return met


from analysis.common import OUT, SUP, prepare_output


def main():
    prepare_output()
    base = m.get_reference_params()
    print("baseline design NP", design_np(base))
    ratios = np.round(np.linspace(0.90, 1.20, 13), 4)
    # Shared 5% LLI background. Incremental mismatches remain <=25% total LLI.
    sevs_lli = np.array(
        [0.05, 0.10, 0.15, 0.20]
    )  # additional LLI; totals 10,15,20,25% in cell 2
    sevs_lam = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    rows = []
    for r in ratios:
        b = base_at_np(base, float(r))
        p1 = m.make_mixed_degraded_cell(b, lli=0.05)
        q1 = m.lowrate_capacity(p1, 0.05)
        # additional LLI
        for d in sevs_lli:
            p2 = m.make_mixed_degraded_cell(b, lli=0.05 + d)
            q2 = m.lowrate_capacity(p2, 0.05)
            sim = m.simulate_ocvr_pair_fast_stateR(
                p1, p2, 1, "charge", 3.30, dq_frac=9e-4, cap1=q1, cap2=q2
            )
            z = metrics(sim, q1, q2)
            rows.append([r, "LLI", d, z["M_peak"], z["M_rms"], z["qex_norm"], q1, q2])
        # LAMn mismatch -> charge
        for d in sevs_lam:
            p2 = m.make_mixed_degraded_cell(b, lli=0.05, lamn=d)
            q2 = m.lowrate_capacity(p2, 0.05)
            sim = m.simulate_ocvr_pair_fast_stateR(
                p1, p2, 1, "charge", 3.30, dq_frac=9e-4, cap1=q1, cap2=q2
            )
            z = metrics(sim, q1, q2)
            rows.append([r, "LAMn", d, z["M_peak"], z["M_rms"], z["qex_norm"], q1, q2])
        # LAMp mismatch -> discharge
        for d in sevs_lam:
            p2 = m.make_mixed_degraded_cell(b, lli=0.05, lamp=d)
            q2 = m.lowrate_capacity(p2, 0.05)
            sim = m.simulate_ocvr_pair_fast_stateR(
                p1, p2, 1, "discharge", 3.35, dq_frac=9e-4, cap1=q1, cap2=q2
            )
            z = metrics(sim, q1, q2)
            rows.append([r, "LAMp", d, z["M_peak"], z["M_rms"], z["qex_norm"], q1, q2])
        print("done NP", r)
    T = pd.DataFrame(
        rows,
        columns=[
            "NP_ratio",
            "Mismatch_mode",
            "Additional_severity",
            "M_peak",
            "M_rms",
            "qex_norm",
            "Q1",
            "Q2",
        ],
    )
    T.to_csv(OUT / "Figure07_NP_sensitivity.csv", index=False)

    fig, axs = plt.subplots(
        1, 3, figsize=(7.35, 2.9), sharey=True, constrained_layout=True
    )
    configs = [
        ("LLI", sevs_lli, "Additional LLI", "Charge"),
        ("LAMn", sevs_lam, "Additional LAM$_n$", "Charge"),
        ("LAMp", sevs_lam, "Additional LAM$_p$", "Discharge"),
    ]
    for ax, (mode, sevs, title, half) in zip(axs, configs):
        for c, d in zip(COLS, sevs):
            z = T[(T.Mismatch_mode == mode) & (np.isclose(T.Additional_severity, d))]
            ax.plot(
                z.NP_ratio,
                z.M_peak,
                marker="o",
                ms=3.1,
                color=c,
                label=f"{int(round(100*d))}%",
            )
        ax.axvline(design_np(base), color="0.25", lw=0.9, ls="--")
        ax.axvspan(1.03, 1.20, color="0.7", alpha=0.12, lw=0)
        ax.set(xlabel="Beginning-of-life N/P ratio", title=f"{half}: {title}")
        ax.set_xlim(0.895, 1.205)
        ax.set_ylim(0, 1.08)
        ax.grid(alpha=0.16, lw=0.5)
    axs[0].set_ylabel("$M_{peak}=\\max |I_1-I_2|/|I_{app}|$")
    axs[2].legend(
        title="Mismatch",
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    for ax, l in zip(axs, "abc"):
        ax.text(0.02, 0.97, l, transform=ax.transAxes, va="top", fontweight="bold")
    fig.text(
        0.50,
        -0.03,
        "Both cells share 5% LLI; shaded region indicates the commonly studied design range near N/P > 1.\nDashed line: reference parameterization.",
        ha="center",
        fontsize=7.2,
    )
    fig.savefig(OUT / "Figure07_NP_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure07_NP_sensitivity.png", bbox_inches="tight", dpi=320)
    plt.close(fig)

    # Full MP-SPMe verification at selected N/P values and larger mismatches where design sensitivity matters.
    verify = []
    for r in [1.00, 1.10, 1.20]:
        b = base_at_np(base, r)
        p1 = m.make_mixed_degraded_cell(b, lli=0.05)
        for mode, d, half, V0 in [
            ("LLI", 0.15, "charge", 3.30),
            ("LAMn", 0.15, "charge", 3.30),
            ("LAMp", 0.25, "discharge", 3.35),
        ]:
            if mode == "LLI":
                p2 = m.make_mixed_degraded_cell(b, lli=0.05 + d)
            elif mode == "LAMn":
                p2 = m.make_mixed_degraded_cell(b, lli=0.05, lamn=d)
            else:
                p2 = m.make_mixed_degraded_cell(b, lli=0.05, lamp=d)
            y, _ = m.init_parallel_at_common_ocv(
                [p1, p2], V0, 0.4 if half == "charge" else 0.75
            )
            sim = m.simulate_cc_halfcycle(
                [p1, p2], y, 1, half, max_step=9, rtol=1.1e-4, atol=1e-6
            )
            q1 = m.lowrate_capacity(p1, 0.05)
            q2 = m.lowrate_capacity(p2, 0.05)
            z = metrics(sim, q1, q2)
            verify.append([r, mode, d, z["M_peak"], z["M_rms"], z["qex_norm"]])
            print("full", r, mode, z["M_peak"], z["qex_norm"])
    V = pd.DataFrame(
        verify,
        columns=[
            "NP_ratio",
            "Mismatch_mode",
            "Additional_severity",
            "M_peak",
            "M_rms",
            "qex_norm",
        ],
    )
    V.to_csv(SUP / "TableS04_NP_fullmodel_verification.csv", index=False)
    print(T.groupby("Mismatch_mode")["M_peak"].agg(["min", "max"]))


if __name__ == "__main__":
    main()
