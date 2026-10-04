"""Numerical recipes for main Figures 3–5 and 9."""
from pathlib import Path
import sys, copy, math, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m

from analysis.common import ROOT, OUT, SUP, prepare_output

COL = {
    "fresh": "#202020",
    "lli": "#0072B2",
    "lamn": "#D55E00",
    "lamp": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#6F6F6F",
    "cyan": "#56B4E9",
    "light": "#BDBDBD",
}
CAP_RATE = 0.05
GRID = np.linspace(0, 0.25, 11)


def save(fig, name, supp=False):
    out = SUP if supp else OUT
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=320)
    plt.close(fig)


def panel(ax, label):
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
        fontsize=9.5,
    )


def make_grid(base, Nneg, Nsep, Npos, Npsd, Nr):
    p = copy.deepcopy(base)
    p["disc"].update(Nneg=Nneg, Nsep=Nsep, Npos=Npos, Npsd=Npsd, Nr_neg=Nr)
    p = m.update_derived_params(p)
    p["QLi_fresh"] = base["QLi_fresh"]
    p["QLi"] = base["QLi_fresh"]
    p["Q_nominal_ref"] = base["Q_nominal_ref"]
    return p


def qex_norm(sim, p1, p2):
    met = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
    q1 = m.lowrate_capacity(p1, CAP_RATE)
    q2 = m.lowrate_capacity(p2, CAP_RATE)
    met["qex_norm"] = met["Q_excess_Ahm2"] / (0.5 * (q1 + q2))
    met["capdiff"] = (q2 - q1) / (0.5 * (q1 + q2))
    met["Q1"] = q1
    met["Q2"] = q2
    return met


def figure3_lowrate(base=None):
    """Generate Figure 3 with full MP-SPMe and full-model capacity matching."""
    from analysis.figure03_fullmodel import main
    return main([])


def figure4_dynamics(base, sn):
    # Charge: equal low-rate capacity; discharge: realistic <=25% pure LAMp comparison.
    p1 = m.make_degraded_cell(base, "LLI", 0.10)
    p2 = m.make_degraded_cell(base, "LAMn", sn)
    y, _ = m.init_parallel_at_common_ocv([p1, p2], 3.30, 0.4)
    ch = m.simulate_cc_halfcycle(
        [p1, p2], y, 1, "charge", max_step=8, rtol=8e-5, atol=7e-7
    )
    p3 = m.make_degraded_cell(base, "LLI", 0.10)
    p4, sp, _ = m.match_capacity_lowrate_ocvr(
        base, p3, "LAMp", bounds=(0, 0.50), C_rate=CAP_RATE
    )
    y, _ = m.init_parallel_at_common_ocv([p3, p4], 3.35, 0.75)
    ds = m.simulate_cc_halfcycle(
        [p3, p4], y, 1, "discharge", max_step=8, rtol=8e-5, atol=7e-7
    )
    mch = qex_norm(ch, p1, p2)
    mds = qex_norm(ds, p3, p4)
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.15), constrained_layout=True)
    axs[0, 0].plot(ch["t"] / 60, ch["I"][:, 0], color=COL["lli"], label=r"10\% LLI")
    axs[0, 0].plot(
        ch["t"] / 60, ch["I"][:, 1], color=COL["lamn"], label=f"{sn*100:.1f}\\% LAM$_n$"
    )
    axs[0, 0].axhline(ch["Iapp"] / 2, color="0.7", lw=0.8, ls=":")
    axs[0, 0].set(ylabel="Branch current (A m$^{-2}$)")
    axs[0, 0].legend(frameon=False)
    axs[0, 1].plot(ds["t"] / 60, ds["I"][:, 0], color=COL["lli"], label=r"10\% LLI")
    axs[0, 1].plot(
        ds["t"] / 60, ds["I"][:, 1], color=COL["lamp"], label=f"{sp*100:.1f}\\% LAM$_p$"
    )
    axs[0, 1].axhline(ds["Iapp"] / 2, color="0.7", lw=0.8, ls=":")
    axs[0, 1].set(ylabel="Branch current (A m$^{-2}$)")
    axs[0, 1].legend(frameon=False)
    axs[1, 0].plot(
        ch["t"] / 60,
        np.abs(ch["I"][:, 0] - ch["I"][:, 1]) / abs(ch["Iapp"]),
        color=COL["green"],
    )
    axs[1, 0].set(xlabel="Time (min)", ylabel="$|I_1-I_2|/|I_{app}|$", ylim=(0, 1.08))
    axs[1, 0].text(
        0.04,
        0.88,
        f'$M_{{peak}}={mch["M_peak"]:.2f}$\n$q_{{ex}}={mch["qex_norm"]:.3f}$',
        transform=axs[1, 0].transAxes,
        fontsize=7.8,
    )
    axs[1, 1].plot(
        ds["t"] / 60,
        np.abs(ds["I"][:, 0] - ds["I"][:, 1]) / abs(ds["Iapp"]),
        color=COL["green"],
    )
    axs[1, 1].set(xlabel="Time (min)", ylabel="$|I_1-I_2|/|I_{app}|$", ylim=(0, 1.08))
    axs[1, 1].text(
        0.04,
        0.88,
        f'$M_{{peak}}={mds["M_peak"]:.2f}$\n$q_{{ex}}={mds["qex_norm"]:.3f}$\n$\\Delta Q={100*mds["capdiff"]:.1f}\\%$',
        transform=axs[1, 1].transAxes,
        fontsize=7.8,
    )
    for a, l in zip(axs.flat, "abcd"):
        panel(a, l)
    save(fig, "Figure04_parallel_dynamics")
    pd.DataFrame(
        [
            {"case": "LLI10_vs_capacity-matched_LAMn", "matched_LAMn": sn, **mch},
            {"case": "LLI10_vs_capacity-matched_LAMp", "matched_LAMp": sp, **mds},
        ]
    ).to_csv(OUT / "Figure04_metrics.csv", index=False)
    return ch, p1, p2


def figure5_model_fidelity(base, sn):
    p1 = m.make_degraded_cell(base, "LLI", 0.10)
    p2 = m.make_degraded_cell(base, "LAMn", sn)
    ocv = m.simulate_ocvr_pair_fast_stateR(p1, p2, 1, "charge", 3.30, dq_frac=7e-4)
    b1 = make_grid(base, 6, 3, 6, 1, 9)
    s1 = m.make_degraded_cell(b1, "LLI", 0.10)
    s2 = m.make_degraded_cell(b1, "LAMn", sn)
    y, _ = m.init_parallel_at_common_ocv([s1, s2], 3.30, 0.4)
    single = m.simulate_cc_halfcycle(
        [s1, s2], y, 1, "charge", max_step=9, rtol=1e-4, atol=1e-6
    )
    bm = make_grid(base, 6, 3, 6, 9, 9)
    q1 = m.make_degraded_cell(bm, "LLI", 0.10)
    q2 = m.make_degraded_cell(bm, "LAMn", sn)
    y, _ = m.init_parallel_at_common_ocv([q1, q2], 3.30, 0.4)
    multi = m.simulate_cc_halfcycle(
        [q1, q2], y, 1, "charge", max_step=9, rtol=1e-4, atol=1e-6
    )
    fig, ax = plt.subplots(figsize=(3.4, 2.55), constrained_layout=True)
    for s, lab, c, ls in [
        (ocv, "OCV-R", COL["lli"], "-"),
        (single, "single-radius electrochemical", COL["lamn"], "--"),
        (multi, "multiparticle electrochemical", COL["lamp"], "-."),
    ]:
        prog = s["t"] / s["t"][-1]
        val = (s["I"][:, 0] - s["I"][:, 1]) / abs(s["Iapp"])
        ax.plot(prog, val, label=lab, color=c, ls=ls)
    ax.axhline(0, color="0.75", lw=0.7)
    ax.set(
        xlabel="Normalized half-cycle progress",
        ylabel="$(I_1-I_2)/|I_{app}|$",
        xlim=(0, 1),
    )
    ax.legend(frameon=False)
    panel(ax, "a")
    save(fig, "Figure05_model_fidelity")
    rows = []
    for s, lab in [
        (ocv, "OCV-R"),
        (single, "single-radius electrochemical"),
        (multi, "multiparticle electrochemical"),
    ]:
        rows.append({"model": lab, **qex_norm(s, p1, p2)})
    pd.DataFrame(rows).to_csv(OUT / "Figure05_model_fidelity.csv", index=False)


def get_cell_cached(cache, base, lli, lamn, lamp, coupling="decoupled"):
    key = (round(lli, 5), round(lamn, 5), round(lamp, 5), coupling)
    if key not in cache:
        p = m.make_mixed_degraded_cell(base, lli, lamn, lamp, coupling)
        cache[key] = (p, m.lowrate_capacity(p, CAP_RATE))
    return cache[key]


def fast_pair_metrics(p1, p2, mode, V0, cap1=None, cap2=None, Rextra2=0):
    s = m.simulate_ocvr_pair_fast_stateR(
        p1, p2, 1, mode, V0, Rextra2=Rextra2, dq_frac=1e-3, cap1=cap1, cap2=cap2
    )
    met = m.compute_current_metrics(s["t"], s["I"], s["Iapp"])
    q1 = cap1 if cap1 is not None else m.lowrate_capacity(p1, CAP_RATE)
    q2 = cap2 if cap2 is not None else m.lowrate_capacity(p2, CAP_RATE)
    met["qex_norm"] = met["Q_excess_Ahm2"] / (0.5 * (q1 + q2))
    met["capdiff"] = (q2 - q1) / (0.5 * (q1 + q2))
    return met


def figure9_resistance(base):
    sevs = np.linspace(0, 0.25, 11)
    rows = []
    R0 = m.estimate_effective_asr(
        m.make_mixed_degraded_cell(base, 0.10, 0, 0, "coupled")
    )
    for s in sevs:
        for cp in ["decoupled", "coupled"]:
            p1 = m.make_mixed_degraded_cell(base, 0.10, 0, 0, cp)
            p2 = m.make_mixed_degraded_cell(base, 0.10, s, 0, cp)
            q1 = m.lowrate_capacity(p1)
            q2 = m.lowrate_capacity(p2)
            z = fast_pair_metrics(p1, p2, "charge", 3.30, q1, q2)
            rows.append(
                [
                    s,
                    cp,
                    m.estimate_effective_asr(p1),
                    m.estimate_effective_asr(p2),
                    z["M_peak"],
                    z["qex_norm"],
                ]
            )
    T = pd.DataFrame(
        rows, columns=["dLAMn", "coupling", "ASR1", "ASR2", "M_peak", "qex_norm"]
    )
    T.to_csv(OUT / "Figure09_surface_area_coupling.csv", index=False)
    contact = np.linspace(0, 0.5, 11)
    H = np.zeros((len(contact), len(sevs)))
    HQ = np.zeros((len(contact), len(sevs)))
    for iy, rr in enumerate(contact):
        for ix, s in enumerate(sevs):
            p1 = m.make_mixed_degraded_cell(base, 0.10, 0, 0, "coupled")
            p2 = m.make_mixed_degraded_cell(base, 0.10, s, 0, "coupled")
            q1 = m.lowrate_capacity(p1)
            q2 = m.lowrate_capacity(p2)
            z = fast_pair_metrics(p1, p2, "charge", 3.30, q1, q2, Rextra2=rr * R0)
            H[iy, ix] = z["M_peak"]
            HQ[iy, ix] = z["qex_norm"]
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.25), constrained_layout=True)
    for cp, c, ls in [("decoupled", COL["lli"], "-"), ("coupled", COL["lamn"], "--")]:
        z = T[T.coupling == cp]
        axs[0, 0].plot(z.dLAMn * 100, z.ASR2 / z.ASR1, label=cp, color=c, ls=ls)
        axs[0, 1].plot(z.dLAMn * 100, z.M_peak, label=cp, color=c, ls=ls)
        axs[1, 0].plot(z.dLAMn * 100, z.qex_norm, label=cp, color=c, ls=ls)
    axs[0, 0].set(
        xlabel="Additional LAM$_n$ in cell 2 (%)", ylabel="$R_{2,eff}/R_{1,eff}$"
    )
    axs[0, 0].legend(frameon=False)
    axs[0, 1].set(xlabel="Additional LAM$_n$ in cell 2 (%)", ylabel="$M_{peak}$")
    axs[1, 0].set(
        xlabel="Additional LAM$_n$ in cell 2 (%)", ylabel="$Q_{excess}/\\bar Q$"
    )
    im = axs[1, 1].imshow(
        HQ, origin="lower", extent=[0, 25, 0, 50], aspect="auto", cmap="magma"
    )
    fig.colorbar(im, ax=axs[1, 1], pad=0.015, label=r"$Q_{excess}/\bar Q$")
    axs[1, 1].set(
        xlabel="Additional LAM$_n$ in cell 2 (%)",
        ylabel="Added contact resistance / fresh ASR (%)",
    )
    for a, l in zip(axs.flat, "abcd"):
        panel(a, l)
    save(fig, "Figure09_resistance_surface_area")
    pd.DataFrame(
        [
            {
                "contact_fraction_R0": contact[iy],
                "dLAMn": sevs[ix],
                "M_peak": H[iy, ix],
                "qex_norm": HQ[iy, ix],
            }
            for iy in range(len(contact))
            for ix in range(len(sevs))
        ]
    ).to_csv(OUT / "Figure09_contact_map.csv", index=False)



def supplementary_studies(base, sn):
    # S2 grid convergence on central charge case
    grids = [(4, 2, 4, 5, 7), (6, 3, 6, 7, 9), (8, 4, 8, 9, 11), (10, 5, 10, 13, 13)]
    rows = []
    for g in grids:
        b = make_grid(base, *g)
        p1 = m.make_degraded_cell(b, "LLI", 0.10)
        p2 = m.make_degraded_cell(b, "LAMn", sn)
        y, _ = m.init_parallel_at_common_ocv([p1, p2], 3.30, 0.4)
        s = m.simulate_cc_halfcycle(
            [p1, p2], y, 1, "charge", max_step=10, rtol=1.2e-4, atol=1e-6
        )
        z = qex_norm(s, p1, p2)
        rows.append([*g, z["M_peak"], z["qex_norm"], s["t"][-1] / 60])
    G = pd.DataFrame(
        rows,
        columns=[
            "Nneg",
            "Nsep",
            "Npos",
            "Npsd",
            "Nr_neg",
            "M_peak",
            "qex_norm",
            "duration_min",
        ],
    )
    G.to_csv(SUP / "TableS01_grid_convergence.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.8), constrained_layout=True)
    lab = [
        f"{a}/{b}/{c}/{d}/{e}"
        for a, b, c, d, e in G[["Nneg", "Nsep", "Npos", "Npsd", "Nr_neg"]].values
    ]
    axs[0].plot(range(len(G)), G.M_peak, marker="o")
    axs[1].plot(range(len(G)), G.qex_norm, marker="o")
    for ax in axs:
        ax.set_xticks(range(len(G)), lab, rotation=35, ha="right")
        ax.set_xlabel("$N_n/N_s/N_p/N_R/N_r$")
    axs[0].set(ylabel="$M_{peak}$")
    axs[1].set(ylabel="$Q_{excess}/\\bar Q$")
    save(fig, "FigureS01_grid_convergence", supp=True)

    # S3 capacity protocol rate sensitivity
    rates = [0.02, 0.05, 0.1]
    rows = []
    for cr in rates:
        pt = m.make_degraded_cell(base, "LLI", 0.10)
        pn, sn2, mn = m.match_capacity_lowrate_ocvr(
            base, pt, "LAMn", bounds=(0, 0.25), C_rate=cr
        )
        rows.append([cr, m.lowrate_capacity(pt, cr), sn2, mn["error_Ahm2"]])
    T = pd.DataFrame(
        rows, columns=["C_rate", "LLI10_capacity", "matched_LAMn", "error"]
    )
    T.to_csv(SUP / "TableS03_capacity_protocol_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 2.7), constrained_layout=True)
    ax.plot(T.C_rate, T.matched_LAMn * 100, marker="o")
    ax.set(
        xlabel="Capacity-characterization rate (C)",
        ylabel="Matched LAM$_n$ severity (%)",
    )
    save(fig, "FigureS03_capacity_protocol_sensitivity", supp=True)

    # S6 C-rate sensitivity, full model central pair
    rows = []
    for cr in [0.1, 0.25, 0.5, 1.0]:
        p1 = m.make_degraded_cell(base, "LLI", 0.10)
        p2 = m.make_degraded_cell(base, "LAMn", sn)
        y, _ = m.init_parallel_at_common_ocv([p1, p2], 3.30, 0.4)
        s = m.simulate_cc_halfcycle(
            [p1, p2],
            y,
            cr,
            "charge",
            max_step=4,
            rtol=1e-5,
            atol=1e-7,
        )
        z = qex_norm(s, p1, p2)
        rows.append([cr, z["M_peak"], z["M_rms"], z["qex_norm"], s["t"][-1] / 60])
    C = pd.DataFrame(
        rows, columns=["C_rate", "M_peak", "M_rms", "qex_norm", "duration_min"]
    )
    C.to_csv(SUP / "TableS06_Crate_sensitivity.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.8), constrained_layout=True)
    axs[0].plot(C.C_rate, C.M_peak, marker="o")
    axs[1].plot(C.C_rate, C.qex_norm, marker="o")
    axs[0].set(xlabel="C-rate", ylabel="$M_{peak}$")
    axs[1].set(xlabel="C-rate", ylabel="$Q_{excess}/\\bar Q$")
    save(fig, "FigureS04_Crate_sensitivity", supp=True)

