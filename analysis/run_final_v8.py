from pathlib import Path
import sys, copy, math, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


def figure1_hysteresis(base):
    # Low-overpotential Zelic-Katrasnik branches; particle-size-dependent Omega.
    b = make_grid(base, 10, 5, 10, 31, 13)
    b["num"]["x_min"] = b["num"]["balance_min"]
    fresh = m.make_degraded_cell(b, "fresh", 0)
    cases = [
        ("LLI", m.make_degraded_cell(b, "LLI", 0.10), COL["lli"]),
        ("LAM$_n$", m.make_degraded_cell(b, "LAMn", 0.10), COL["lamn"]),
        ("LAM$_p$", m.make_degraded_cell(b, "LAMp", 0.10), COL["lamp"]),
    ]
    chf, dsf = m.simulate_mp0d_cycle_fixed(fresh, 0.02, dq_frac=0.0015)
    fig, axs = plt.subplots(
        1, 3, figsize=(7.25, 2.92), sharey=True, constrained_layout=True
    )
    rows = []
    for ax, (lab, p, c) in zip(axs, cases):
        ch, ds = m.simulate_mp0d_cycle_fixed(p, 0.02, dq_frac=0.0015)
        ax.plot(chf["xn"], chf["V"], color=COL["fresh"], lw=1.2, label="Fresh, charge")
        ax.plot(
            dsf["xn"],
            dsf["V"],
            color=COL["fresh"],
            lw=1.2,
            ls="--",
            label="Fresh, discharge",
        )
        ax.plot(ch["xn"], ch["V"], color=c, label=f"{lab}, charge")
        ax.plot(ds["xn"], ds["V"], color=c, ls="--", label=f"{lab}, discharge")
        ax.set(xlabel="Bulk graphite lithiation, $x_n$", xlim=(0, 1), ylim=(2.48, 3.69))
        # Plateau zoom makes the thermodynamic hysteresis visible without hiding the cutoffs.
        iax = inset_axes(
            ax, width="46%", height="38%", loc="lower center", borderpad=1.1
        )
        for z, ls_, cc in [
            (chf, "-", COL["fresh"]),
            (dsf, "--", COL["fresh"]),
            (ch, "-", c),
            (ds, "--", c),
        ]:
            iax.plot(z["xn"], z["V"], color=cc, ls=ls_, lw=1.0)
        iax.set_xlim(0.08, 0.72)
        iax.set_ylim(3.17, 3.40)
        iax.tick_params(labelsize=6.2)
        # quantify charge-discharge loop width for the degraded cell
        lo = max(ch["xn"].min(), ds["xn"].min())
        hi = min(ch["xn"].max(), ds["xn"].max())
        xx = np.linspace(lo, hi, 500)
        vc = np.interp(xx, ch["xn"], ch["V"])
        vd = np.interp(xx, ds["xn"][::-1], ds["V"][::-1])
        gap = vc - vd
        rows.append(
            [
                lab,
                float(np.nanmedian(gap) * 1e3),
                float(np.nanmax(gap) * 1e3),
                ch["capacity_Ahm2"],
                ds["capacity_Ahm2"],
            ]
        )
    axs[0].set_ylabel("Quasi-static terminal voltage (V)")
    axs[0].legend(frameon=False, fontsize=6.1, loc="upper left")
    for a, l in zip(axs, "abc"):
        panel(a, l)
    save(fig, "Figure01_hysteresis_degradation")
    pd.DataFrame(
        rows,
        columns=[
            "Mode",
            "median_loop_mV",
            "max_loop_mV",
            "Qcharge_Ahm2",
            "Qdischarge_Ahm2",
        ],
    ).to_csv(OUT / "Figure01_metrics.csv", index=False)


def figure3_lowrate(base):
    lli = 0.10
    p_lli = m.make_degraded_cell(base, "LLI", lli)
    p_lamn, sn, meta = m.match_capacity_lowrate_ocvr(
        base, p_lli, "LAMn", bounds=(0, 0.25), C_rate=CAP_RATE
    )
    p_lamp, sp, _ = m.match_capacity_lowrate_ocvr(
        base, p_lli, "LAMp", bounds=(0, 0.50), C_rate=CAP_RATE
    )
    cells = [p_lli, p_lamn, p_lamp]
    labs = [
        r"10\% LLI",
        f"{sn*100:.1f}\\% LAM$_n$ (capacity matched)",
        f"{sp*100:.1f}\\% LAM$_p$ (capacity matched)",
    ]
    cols = [COL["lli"], COL["lamn"], COL["lamp"]]
    fig, axs = plt.subplots(1, 2, figsize=(7.25, 2.95), constrained_layout=True)
    rows = []
    for p, lab, c in zip(cells, labs, cols):
        z = m.lowrate_ocvr_characterization(p, CAP_RATE, n_curve=800)
        qnch = z["q_charge"] / z["capacity_Ahm2"]
        qnds = z["q_discharge"] / z["capacity_Ahm2"]
        axs[0].plot(
            qnch,
            z["V_charge"],
            color=c,
            label=f'{lab}\n$Q={z["capacity_Ahm2"]:.2f}$ Ah m$^{{-2}}$',
        )
        axs[1].plot(
            qnds, z["V_discharge"], color=c, label=lab
        )
        rows.append(
            [
                lab,
                z["capacity_Ahm2"],
                z["xp_high"],
                z["xp_low"],
                z["xn_high"],
                z["xn_low"],
            ]
        )
    axs[0].set(xlabel="Normalized charged capacity", ylabel="C/20 OCV--R voltage (V)")
    axs[1].set(xlabel="Normalized discharged capacity")
    for ax in axs:
        ax.axhline(base["Vmin"], color="0.82", lw=0.8)
        ax.axhline(base["Vmax"], color="0.82", lw=0.8)
        ax.set(xlim=(0, 1), ylim=(2.47, 3.69))
    axs[0].legend(frameon=False, fontsize=6.3, loc="best")
    for a, l in zip(axs, "ab"):
        panel(a, l)
    save(fig, "Figure03_lowrate_characterization")
    pd.DataFrame(
        rows,
        columns=["Case", "Capacity_Ahm2", "xp_high", "xp_low", "xn_high", "xn_low"],
    ).to_csv(OUT / "Figure03_lowrate.csv", index=False)
    return sn


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
    fig, ax = plt.subplots(figsize=(7.0, 3.15), constrained_layout=True)
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


def figure6_robustness(base, sn):
    from analysis.robustness_v28 import run_all

    return run_all(base)


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


def figure7_pure_maps(base):
    cache = {}
    n = len(GRID)
    Mn = np.zeros((n, n))
    Qn = np.zeros((n, n))
    Dn = np.zeros((n, n))
    Mp = np.zeros((n, n))
    Qp = np.zeros((n, n))
    Dp = np.zeros((n, n))
    rows = []
    for iy, lli in enumerate(GRID):
        p1, c1 = get_cell_cached(cache, base, lli, 0, 0)
        for ix, lam in enumerate(GRID):
            pn, cn = get_cell_cached(cache, base, 0, lam, 0)
            z = fast_pair_metrics(p1, pn, "charge", 3.30, c1, cn)
            Mn[iy, ix] = z["M_peak"]
            Qn[iy, ix] = z["qex_norm"]
            Dn[iy, ix] = z["capdiff"]
            rows.append(
                {
                    "family": "LLI_vs_LAMn",
                    "LLI": lli,
                    "LAM": lam,
                    **{k: z[k] for k in ["M_peak", "M_rms", "qex_norm", "capdiff"]},
                }
            )
            pp, cp = get_cell_cached(cache, base, 0, 0, lam)
            z = fast_pair_metrics(p1, pp, "discharge", 3.35, c1, cp)
            Mp[iy, ix] = z["M_peak"]
            Qp[iy, ix] = z["qex_norm"]
            Dp[iy, ix] = z["capdiff"]
            rows.append(
                {
                    "family": "LLI_vs_LAMp",
                    "LLI": lli,
                    "LAM": lam,
                    **{k: z[k] for k in ["M_peak", "M_rms", "qex_norm", "capdiff"]},
                }
            )
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.7), constrained_layout=True)
    ext = [0, 25, 0, 25]
    datasets = [
        (Mn, "Charge: pure LLI vs pure LAM$_n$", "$M_{peak}$", Dn),
        (Qn, "Charge: pure LLI vs pure LAM$_n$", "$Q_{excess}/\\bar Q$", Dn),
        (Mp, "Discharge: pure LLI vs pure LAM$_p$", "$M_{peak}$", Dp),
        (Qp, "Discharge: pure LLI vs pure LAM$_p$", "$Q_{excess}/\\bar Q$", Dp),
    ]
    for ax, (Z, title, cbar, D) in zip(axs.flat, datasets):
        im = ax.imshow(Z, origin="lower", extent=ext, aspect="auto", cmap="viridis")
        cb = fig.colorbar(im, ax=ax, pad=0.015)
        cb.set_label(cbar)
        # Equal-capacity contour only if zero is within the map.
        if np.nanmin(D) <= 0 <= np.nanmax(D):
            ax.contour(
                GRID * 100,
                GRID * 100,
                D,
                levels=[0],
                colors="white",
                linewidths=1.3,
                linestyles="--",
            )
        ax.set(xlabel="LAM severity (%)", ylabel="LLI severity (%)", title=title)
    for a, l in zip(axs.flat, "abcd"):
        panel(a, l)
    save(fig, "Figure07_pure_mode_maps")
    pd.DataFrame(rows).to_csv(OUT / "Figure07_pure_mode_maps.csv", index=False)


def realistic_maps(base):
    cache = {}
    n = len(GRID)
    data = {
        k: np.zeros((n, n))
        for k in ["A_M", "A_Q", "B_M", "B_Q", "C_M", "C_Q", "D_M", "D_Q"]
    }
    rows = []
    # A: common LLI background, additional LAMn in cell 2 (charge)
    # B: common LLI background, additional LAMp in cell 2 (discharge)
    # C: same LLI+LAMn trajectory at different severity s1 vs s2 (charge)
    # D: common LLI+LAMn background, additional LAMp in cell 2 (discharge)
    for iy, s1 in enumerate(GRID):
        cA, qA = get_cell_cached(cache, base, s1, 0, 0)
        cD, qD = get_cell_cached(cache, base, s1, s1, 0)
        for ix, s2 in enumerate(GRID):
            pA, q2 = get_cell_cached(cache, base, s1, s2, 0)
            z = fast_pair_metrics(cA, pA, "charge", 3.30, qA, q2)
            data["A_M"][iy, ix] = z["M_peak"]
            data["A_Q"][iy, ix] = z["qex_norm"]
            rows.append(
                {
                    "family": "common_LLI_plus_dLAMn",
                    "s1": s1,
                    "s2": s2,
                    **{k: z[k] for k in ["M_peak", "M_rms", "qex_norm", "capdiff"]},
                }
            )
            pB, q2 = get_cell_cached(cache, base, s1, 0, s2)
            z = fast_pair_metrics(cA, pB, "discharge", 3.35, qA, q2)
            data["B_M"][iy, ix] = z["M_peak"]
            data["B_Q"][iy, ix] = z["qex_norm"]
            rows.append(
                {
                    "family": "common_LLI_plus_dLAMp",
                    "s1": s1,
                    "s2": s2,
                    **{k: z[k] for k in ["M_peak", "M_rms", "qex_norm", "capdiff"]},
                }
            )
            c1, q1 = get_cell_cached(cache, base, s1, s1, 0)
            c2, q2 = get_cell_cached(cache, base, s2, s2, 0)
            z = fast_pair_metrics(c1, c2, "charge", 3.30, q1, q2)
            data["C_M"][iy, ix] = z["M_peak"]
            data["C_Q"][iy, ix] = z["qex_norm"]
            rows.append(
                {
                    "family": "same_LLI_LAMn_trajectory",
                    "s1": s1,
                    "s2": s2,
                    **{k: z[k] for k in ["M_peak", "M_rms", "qex_norm", "capdiff"]},
                }
            )
            pD, q2 = get_cell_cached(cache, base, s1, s1, s2)
            z = fast_pair_metrics(cD, pD, "discharge", 3.35, qD, q2)
            data["D_M"][iy, ix] = z["M_peak"]
            data["D_Q"][iy, ix] = z["qex_norm"]
            rows.append(
                {
                    "family": "common_LLI_LAMn_plus_dLAMp",
                    "s1": s1,
                    "s2": s2,
                    **{k: z[k] for k in ["M_peak", "M_rms", "qex_norm", "capdiff"]},
                }
            )
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "Figure08_realistic_maps.csv", index=False)

    def centers_to_edges(v):
        dv = np.diff(v)
        left = v[0] - dv[0] / 2
        right = v[-1] + dv[-1] / 2
        mids = v[:-1] + dv / 2
        return np.r_[left, mids, right]

    g = GRID * 100
    # Main-text Figure 8 compares instantaneous peak takeover with the
    # time-averaged severity of the same two realistic charge-side families.
    A_peak = (
        T[T.family == "common_LLI_plus_dLAMn"]
        .pivot(index="s1", columns="s2", values="M_peak")
        .loc[GRID, GRID[1:]]
        .to_numpy()
    )
    A_rms = (
        T[T.family == "common_LLI_plus_dLAMn"]
        .pivot(index="s1", columns="s2", values="M_rms")
        .loc[GRID, GRID[1:]]
        .to_numpy()
    )
    C_peak = (
        T[T.family == "same_LLI_LAMn_trajectory"]
        .pivot(index="s1", columns="s2", values="M_peak")
        .loc[GRID, GRID]
        .to_numpy()
    )
    C_rms = (
        T[T.family == "same_LLI_LAMn_trajectory"]
        .pivot(index="s1", columns="s2", values="M_rms")
        .loc[GRID, GRID]
        .to_numpy()
    )
    xA = g[1:]
    yA = g

    fig, axs = plt.subplots(
        2, 2, figsize=(7.25, 5.45), constrained_layout=True, sharex="col", sharey="row"
    )
    # Top row: M_peak. A common 0--1 scale makes the widespread saturation visible.
    im = axs[0, 0].pcolormesh(
        centers_to_edges(xA),
        centers_to_edges(yA),
        A_peak,
        cmap="viridis",
        vmin=0,
        vmax=1,
        shading="auto",
    )
    axs[0, 1].pcolormesh(
        centers_to_edges(g),
        centers_to_edges(g),
        C_peak,
        cmap="viridis",
        vmin=0,
        vmax=1,
        shading="auto",
    )
    cb = fig.colorbar(im, ax=axs[0, :], pad=0.015)
    cb.set_label(r"$M_{\mathrm{peak}}$")
    # Bottom row: M_RMS. Share the scale between the two studies so the
    # persistence/severity contrast is directly comparable.
    rms_max = max(np.nanmax(A_rms), np.nanmax(C_rms))
    im = axs[1, 0].pcolormesh(
        centers_to_edges(xA),
        centers_to_edges(yA),
        A_rms,
        cmap="viridis",
        vmin=0,
        vmax=rms_max,
        shading="auto",
    )
    axs[1, 1].pcolormesh(
        centers_to_edges(g),
        centers_to_edges(g),
        C_rms,
        cmap="viridis",
        vmin=0,
        vmax=rms_max,
        shading="auto",
    )
    cb = fig.colorbar(im, ax=axs[1, :], pad=0.015)
    cb.set_label(r"$M_{\mathrm{RMS}}$")

    for ax in axs[:, 0]:
        ax.set_ylabel("Common LLI in both cells (%)")
        ax.set_xlim(xA[0] - 1.25, xA[-1] + 1.25)
        ax.set_xticks([5, 10, 15, 20, 25])
    for ax in axs[:, 1]:
        ax.set_ylabel("Cell 1 severity (%)")
        ax.set_xlim(g[0] - 1.25, g[-1] + 1.25)
        ax.set_xticks([0, 5, 10, 15, 20, 25])
    for ax in axs.flat:
        ax.set_ylim(g[0] - 1.25, g[-1] + 1.25)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
    axs[1, 0].set_xlabel("Additional LAM$_n$ in cell 2 (%)")
    axs[1, 1].set_xlabel("Cell 2 severity, LLI=LAM$_n$ (%)")
    # Panel labels use a white backing so they remain legible over every map color.
    for a, l in zip(axs.flat, "abcd"):
        a.text(
            0.018,
            0.982,
            l,
            transform=a.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=9.5,
            bbox=dict(
                boxstyle="round,pad=0.14", fc="white", ec="0.2", lw=0.35, alpha=0.92
            ),
        )
    save(fig, "Figure08_realistic_degradation_maps")

    # Supplement: integrated-severity maps
    specs = [
        (
            "A_Q",
            "Common LLI + differential LAM$_n$",
            "Additional LAM$_n$ in cell 2 (%)",
            "Common LLI in both cells (%)",
        ),
        (
            "B_Q",
            "Common LLI + differential LAM$_p$",
            "Additional LAM$_p$ in cell 2 (%)",
            "Common LLI in both cells (%)",
        ),
        (
            "C_Q",
            "Same LLI+LAM$_n$ modes, different severity",
            "Cell 2 severity, LLI=LAM$_n$ (%)",
            "Cell 1 severity, LLI=LAM$_n$ (%)",
        ),
        (
            "D_Q",
            "LLI+LAM$_n$ background + differential LAM$_p$",
            "Additional LAM$_p$ in cell 2 (%)",
            "Common LLI=LAM$_n$ background (%)",
        ),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.7), constrained_layout=True)
    for ax, (key, title, xlab, ylab) in zip(axs.flat, specs):
        source = data[key]
        im = ax.imshow(
            source, origin="lower", extent=[0, 25, 0, 25], aspect="auto", cmap="viridis"
        )
        cb = fig.colorbar(im, ax=ax, pad=0.015)
        cb.set_label("$q_{\\mathrm{excess}}$")
        ax.set(xlabel=xlab, ylabel=ylab, title=title)
    for a, l in zip(axs.flat, "abcd"):
        panel(a, l)
    save(fig, "FigureS04_realistic_qex_maps", supp=True)


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


def supplementary(base, sn, real_data):
    # S1: direct verification of particle-size-dependent regular solution + low-overpotential hysteresis branches.
    x = np.linspace(0.001, 0.999, 1200)
    radii = np.array([20, 30, 50, 100, 200]) * 1e-9
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    om = m.omega_rt_from_particle_radius(radii)
    axs[0].plot(radii * 1e9, om, marker="o")
    axs[0].axhline(2, color="0.7", ls=":", lw=0.8)
    axs[0].set(xlabel="Equivalent spherical radius (nm)", ylabel="$\\Omega/k_BT$")
    for r, w in zip(radii[1:], om[1:]):
        pp = copy.deepcopy(base)
        pp["disc"]["Npsd"] = 1
        pp["lfp"]["R_mean"] = float(r)
        pp["lfp"]["psd_cv"] = 0
        pp = m.update_derived_params(pp)
        pp["QLi_fresh"] = base["QLi_fresh"]
        pp["QLi"] = base["QLi"]
        pp["Q_nominal_ref"] = base["Q_nominal_ref"]
        uc = m.lfp_ocp_zk0d(x, np.ones_like(x), pp)
        ud = m.lfp_ocp_zk0d(x, -np.ones_like(x), pp)
        axs[1].plot(x, uc, label=f"{r*1e9:.0f} nm charge")
        axs[1].plot(x, ud, ls="--", alpha=0.85)
    axs[1].set(xlabel="LFP lithiation, $x_p$", ylabel="LFP OCP (V)")
    axs[1].legend(frameon=False, fontsize=6.0, ncol=2)
    for a, l in zip(axs, "ab"):
        panel(a, l)
    save(fig, "FigureS01_ZK_verification", supp=True)

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
    save(fig, "FigureS02_grid_convergence", supp=True)

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

    # S5: same fully mixed trajectory different severity: theta=[s,0.5s,0.5s]
    cache = {}
    n = len(GRID)
    M = np.zeros((n, n))
    Q = np.zeros((n, n))
    for iy, s1 in enumerate(GRID):
        p1, q1 = get_cell_cached(cache, base, s1, 0.5 * s1, 0.5 * s1)
        for ix, s2 in enumerate(GRID):
            p2, q2 = get_cell_cached(cache, base, s2, 0.5 * s2, 0.5 * s2)
            z = fast_pair_metrics(p1, p2, "charge", 3.30, q1, q2)
            M[iy, ix] = z["M_peak"]
            Q[iy, ix] = z["qex_norm"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for ax, Z, lab in [(axs[0], M, "$M_{peak}$"), (axs[1], Q, "$Q_{excess}/\\bar Q$")]:
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[0, 25, 0, 25],
            aspect="auto",
            cmap="viridis" if ax is axs[0] else "magma",
        )
        fig.colorbar(im, ax=ax, pad=0.015, label=lab)
        ax.set(xlabel="Cell 2 severity $s_2$ (%)", ylabel="Cell 1 severity $s_1$ (%)")
    save(fig, "FigureS05_same_mixed_trajectory", supp=True)

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
    save(fig, "FigureS06_Crate_sensitivity", supp=True)

    # S7 Full-model verification of realistic cases including requested 5/5 vs 10/10.
    cases = [
        ("common LLI 10% vs +LAMn 10%", (0.10, 0, 0), (0.10, 0.10, 0), "charge", 3.30),
        (
            "common LLI 10% vs +LAMp 10%",
            (0.10, 0, 0),
            (0.10, 0, 0.10),
            "discharge",
            3.35,
        ),
        ("5% LLI+5% LAMn vs 10%+10%", (0.05, 0.05, 0), (0.10, 0.10, 0), "charge", 3.30),
        (
            "10% LLI+LAMn vs +10% LAMp",
            (0.10, 0.10, 0),
            (0.10, 0.10, 0.10),
            "discharge",
            3.35,
        ),
    ]
    rows = []
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)
    for ax, (lab, a, b, mode, V0) in zip(axs.flat, cases):
        p1 = m.make_mixed_degraded_cell(base, *a)
        p2 = m.make_mixed_degraded_cell(base, *b)
        q1 = m.lowrate_capacity(p1)
        q2 = m.lowrate_capacity(p2)
        fast = m.simulate_ocvr_pair_fast_stateR(
            p1, p2, 1, mode, V0, dq_frac=8e-4, cap1=q1, cap2=q2
        )
        mf = qex_norm(fast, p1, p2)
        y, _ = m.init_parallel_at_common_ocv(
            [p1, p2], V0, 0.4 if mode == "charge" else 0.75
        )
        full = m.simulate_cc_halfcycle(
            [p1, p2], y, 1, mode, max_step=4, rtol=1e-5, atol=1e-7
        )
        mm = qex_norm(full, p1, p2)
        ax.plot(
            fast["t"] / fast["t"][-1],
            np.abs(fast["I"][:, 0] - fast["I"][:, 1]) / abs(fast["Iapp"]),
            label="OCV-R",
            color=COL["gray"],
        )
        ax.plot(
            full["t"] / full["t"][-1],
            np.abs(full["I"][:, 0] - full["I"][:, 1]) / abs(full["Iapp"]),
            label="MP-SPMe",
            color=COL["green"],
        )
        ax.set(title=lab, xlabel="Normalized progress", ylabel="$|I_1-I_2|/|I_{app}|$")
        rows.append(
            [lab, mode, mf["M_peak"], mm["M_peak"], mf["qex_norm"], mm["qex_norm"]]
        )
    axs[0, 0].legend(frameon=False)
    save(fig, "FigureS07_realistic_fullmodel_verification", supp=True)
    pd.DataFrame(
        rows,
        columns=[
            "case",
            "mode",
            "Mpeak_OCVR",
            "Mpeak_MPSPMe",
            "qex_OCVR",
            "qex_MPSPMe",
        ],
    ).to_csv(SUP / "TableS07_realistic_fullmodel_verification.csv", index=False)


def summary(base, sn):
    lines = []
    lines.append(f"Matched LAMn to 10% LLI at C/20: {sn*100:.3f}%")
    for f in [
        "Figure01_metrics.csv",
        "Figure04_metrics.csv",
        "Figure05_model_fidelity.csv",
        "Figure06_parameter_robustness.csv",
        "Figure09_surface_area_coupling.csv",
    ]:
        p = OUT / f
        if p.exists():
            lines.append("\n" + f + "\n" + pd.read_csv(p).to_string(index=False))
    (OUT / "RESULTS_SUMMARY.txt").write_text("\n".join(lines))


def main():
    prepare_output()
    base = m.get_reference_params()
    print("1 Figure 1 hysteresis")
    figure1_hysteresis(base)
    print("3 Figure 3 low-rate")
    sn = figure3_lowrate(base)
    print("matched LAMn", sn)
    print("4 Figure 4 dynamics")
    figure4_dynamics(base, sn)
    print("5 Figure 5 fidelity")
    figure5_model_fidelity(base, sn)
    print("6 Figure 6 robustness")
    figure6_robustness(base, sn)
    print("7 Figure 7 pure maps")
    figure7_pure_maps(base)
    print("8 Figure 8 realistic maps")
    rd = realistic_maps(base)
    print("9 Figure 9 resistance")
    figure9_resistance(base)
    print("S supplementary")
    supplementary(base, sn, rd)
    summary(base, sn)
    print("DONE", OUT)


if __name__ == "__main__":
    main()
