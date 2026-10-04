"""One-at-a-time robustness protocol with capacity matching per variant.

The parameter values follow SI Table S2. The reference spatial grid is used throughout the parameter sweep.
Each row records both the requested variant and its own C/20 LAMn match.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import inspect
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfp_parallel import model as m
from analysis.common import OUT, prepare_output

VARIANTS = (
    ("Mean $R_p$ (nm)", "rmean_nm", (36.5, 50, 70, 100)),
    ("PSD CV", "psd_cv", (0.10, 0.25, 0.35)),
    ("$D_{s,n}$", "ds_neg", (3e-15, 1e-14, 3e-14)),
    ("$j_{0,p}$ multiplier", "j0_pos_mult", (0.5, 1.0, 1.5)),
    ("$j_{0,n}$ multiplier", "j0_neg_mult", (0.5, 1.0, 1.5)),
    ("$\\kappa_e$", "kappa", (0.55, 0.78, 1.20)),
    ("$D_e$", "de", (0.75e-10, 1.2e-10, 3e-10)),
    ("$R_{contact}$", "rcontact", (0.0, 0.001, 0.003)),
)
GRID = dict(Nneg=10, Nsep=5, Npos=10, Npsd=13, Nr_neg=13)
SOLVER = dict(max_step=4, rtol=1e-5, atol=1e-7)


def reference_grid(base):
    p = copy.deepcopy(base)
    p["disc"].update(GRID)
    p = m.update_derived_params(p)
    p["QLi_fresh"] = base["QLi_fresh"]
    p["QLi"] = base["QLi"]
    p["Q_nominal_ref"] = base["Q_nominal_ref"]
    return p


def points():
    return [(label, key, value) for label, key, values in VARIANTS for value in values]


def run_point(base, label, key, value):
    start = time.perf_counter()
    varied = m.apply_parameter_variant(reference_grid(base), key, value)
    p1 = m.make_degraded_cell(varied, "LLI", 0.10)
    p2, severity, match = m.match_capacity_lowrate_ocvr(
        varied, p1, "LAMn", bounds=(0.0, 0.40), C_rate=0.05
    )
    y, _ = m.init_parallel_at_common_ocv([p1, p2], 3.30, 0.4)
    sim = m.simulate_cc_halfcycle([p1, p2], y, 1.0, "charge", **SOLVER)
    metrics = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
    return dict(
        Parameter=label,
        Key=key,
        Value=value,
        matched_LAMn=severity,
        protocol_sha256=PROTOCOL_HASH,
        target_capacity_Ahm2=match["target_capacity"],
        matched_capacity_Ahm2=match["matched_capacity"],
        matching_error_Ahm2=match["error_Ahm2"],
        M_peak=metrics["M_peak"],
        M_rms=metrics["M_rms"],
        qex_norm=metrics["Q_excess_Ahm2"]
        / (0.5 * (match["target_capacity"] + match["matched_capacity"])),
        duration_min=sim["t"][-1] / 60,
        time_s=time.perf_counter() - start,
    )


# Plot layout is excluded: changing a legend cannot invalidate numerical data.
PROTOCOL_HASH = hashlib.sha256(
    Path(m.__file__).read_bytes()
    + json.dumps(
        dict(variants=VARIANTS, grid=GRID, solver=SOLVER), sort_keys=True
    ).encode()
    + inspect.getsource(reference_grid).encode()
    + inspect.getsource(run_point).encode()
).hexdigest()


def plot(df):
    # Use the full page width: two side-by-side metrics and eight parameter
    # categories are too compressed at single-column width. Reserve separate
    # rows for the shared x label and legend so neither overlaps the axes.
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.65))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.36, top=0.96, wspace=0.34)
    categories = list(dict.fromkeys(df["Parameter"]))
    palette = (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#A6761D",
        "#7F7F7F",
    )
    handles = []
    for i, cat in enumerate(categories):
        rows = df[df.Parameter == cat]
        color = palette[i % len(palette)]
        for ax, field in zip(axs, ("M_peak", "qex_norm")):
            ax.scatter(np.full(len(rows), i), rows[field], s=18, color=color, zorder=3)
            ax.plot(
                [i, i],
                [rows[field].min(), rows[field].max()],
                color=color,
                lw=0.8,
                alpha=0.75,
            )
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="none", color=color, label=cat)
        )
    for ax in axs:
        ax.set_xlim(-0.5, len(categories) - 0.5)
        ax.set_xticks([])
    fig.supxlabel("Parameter varied", y=0.235, fontsize=9)
    axs[0].set_ylabel(r"$M_{\mathrm{peak}}$")
    axs[1].set_ylabel(r"$Q_{\mathrm{excess}}/\bar Q$")
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.015),
        columnspacing=1.0,
        handletextpad=0.35,
    )
    for ax, label in zip(axs, "ab"):
        ax.text(
            0.012,
            0.985,
            label,
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
            fontsize=9.5,
        )
    for ext in ("pdf", "png"):
        fig.savefig(
            OUT / f"Figure06_parameter_robustness.{ext}", bbox_inches="tight", dpi=320
        )
    plt.close(fig)


def run_all(base=None):
    prepare_output()
    base = m.get_reference_params() if base is None else base
    output = OUT / "Figure06_parameter_robustness.csv"
    expected = points()
    existing = (
        pd.read_csv(output)
        if output.exists() and "protocol_sha256" in pd.read_csv(output, nrows=0).columns
        else pd.DataFrame()
    )
    if len(existing) and set(existing.protocol_sha256) != {PROTOCOL_HASH}:
        raise RuntimeError(
            "Robustness checkpoint belongs to a different source revision"
        )
    rows = []
    for label, key, value in expected:
        old = (
            existing[
                (existing.Key == key)
                & np.isclose(existing.Value, value, rtol=1e-12, atol=0)
            ]
            if len(existing)
            else pd.DataFrame()
        )
        row = old.iloc[0].to_dict() if len(old) else run_point(base, label, key, value)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(
            f"{key} {value}: LAMn={row['matched_LAMn']:.6f} M_peak={row['M_peak']:.6f}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    if (
        len(df) != len(expected)
        or df.isna().any().any()
        or df.matching_error_Ahm2.abs().max() > 1e-4
    ):
        raise m.NumericalError("Incomplete or inaccurately matched robustness study")
    plot(df)
    (OUT / "Figure06_protocol.json").write_text(
        json.dumps(
            dict(
                numerical_recipe_sha256=PROTOCOL_HASH,
                grid=GRID,
                solver=SOLVER,
                C_rate=1.0,
                capacity_rate=0.05,
                initial_voltage=3.30,
                variants=expected,
            ),
            indent=2,
        )
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    run_all()
