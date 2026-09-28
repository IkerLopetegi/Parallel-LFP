"""Section 3.7 degradation-heterogeneity threshold analysis.

The reduced OCV-R model is used for dense maps of the heterogeneity required
to reach prescribed current-redistribution levels. Selected points can be
rerun with the full MP-SPMe for verification.

Main families:
  1. common LLI background + differential LAMn in cell 2;
  2. same LLI+LAMn trajectory at different severity;
  3. common LLI background + differential LLI.

The default dense study resolves background degradation from 0 to 15% in
2.5 %-point increments and branch-to-branch heterogeneity from 0 to 5% in
0.25 %-point increments. Thresholds are extracted for M_peak = 0.50, 0.80,
and 0.95, corresponding approximately to 75/25, 90/10, and 97.5/2.5
current splits while both branch currents retain the applied-current sign.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_section37"
OUT.mkdir(exist_ok=True)

CAP_RATE = 0.05
C_RATE = 1.0
COMMON = np.arange(0.0, 0.150001, 0.025)
DELTA = np.arange(0.0, 0.050001, 0.0025)
THRESHOLDS = (0.50, 0.80, 0.95)


def reduced_pair_metrics(p1, p2, q1, q2):
    sim = m.simulate_ocvr_pair_fast_stateR(
        p1, p2, C_RATE, "charge", 3.30,
        dq_frac=2e-3, cap1=q1, cap2=q2
    )
    z = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
    z["qex_norm"] = z["Q_excess_Ahm2"] / (0.5 * (q1 + q2))
    z["capdiff"] = (q2 - q1) / (0.5 * (q1 + q2))
    return z


def run_dense_reduced():
    base = m.get_reference_params()
    cache = {}
    rows = []

    def cell(lli, lamn):
        key = (round(lli, 6), round(lamn, 6))
        if key not in cache:
            p = m.make_mixed_degraded_cell(base, lli, lamn, 0.0, "decoupled")
            cache[key] = (p, m.lowrate_capacity(p, CAP_RATE))
        return cache[key]

    families = (
        "common_LLI_dLAMn",
        "same_LLI_LAMn_deltaSeverity",
        "common_LLI_dLLI",
    )

    for family in families:
        for background in COMMON:
            if family == "same_LLI_LAMn_deltaSeverity":
                p1, q1 = cell(background, background)
            else:
                p1, q1 = cell(background, 0.0)

            for delta in DELTA:
                if family == "common_LLI_dLAMn":
                    p2, q2 = cell(background, delta)
                elif family == "same_LLI_LAMn_deltaSeverity":
                    if background + delta > 0.25:
                        continue
                    p2, q2 = cell(background + delta, background + delta)
                else:
                    if background + delta > 0.25:
                        continue
                    p2, q2 = cell(background + delta, 0.0)

                z = reduced_pair_metrics(p1, p2, q1, q2)
                rows.append({
                    "family": family,
                    "background": background,
                    "delta": delta,
                    "M_peak": z["M_peak"],
                    "M_rms": z["M_rms"],
                    "qex_norm": z["qex_norm"],
                    "capdiff": z["capdiff"],
                    "Q1": q1,
                    "Q2": q2,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Section37_fine_reduced.csv", index=False)

    threshold_rows = []
    for (family, background), group in df.groupby(["family", "background"]):
        group = group.sort_values("delta")
        for threshold in THRESHOLDS:
            hit = group[group["M_peak"] >= threshold]
            if len(hit):
                r = hit.iloc[0]
                threshold_rows.append({
                    "family": family,
                    "background": background,
                    "threshold": threshold,
                    "delta_crit": r["delta"],
                    "M_peak": r["M_peak"],
                    "M_rms": r["M_rms"],
                    "qex_norm": r["qex_norm"],
                    "capdiff": r["capdiff"],
                })
            else:
                threshold_rows.append({
                    "family": family,
                    "background": background,
                    "threshold": threshold,
                    "delta_crit": np.nan,
                    "M_peak": np.nan,
                    "M_rms": np.nan,
                    "qex_norm": np.nan,
                    "capdiff": np.nan,
                })

    td = pd.DataFrame(threshold_rows)
    td.to_csv(OUT / "Section37_fine_thresholds_reduced.csv", index=False)
    return df, td


def run_full_point(background, delta):
    """Run one full MP-SPMe verification point for common LLI + differential LAMn."""
    base = m.get_reference_params()
    p1 = m.make_mixed_degraded_cell(base, background, 0.0, 0.0)
    p2 = m.make_mixed_degraded_cell(base, background, delta, 0.0)
    q1 = m.lowrate_capacity(p1, CAP_RATE)
    q2 = m.lowrate_capacity(p2, CAP_RATE)
    y0, _ = m.init_parallel_at_common_ocv([p1, p2], 3.30, 0.4)
    sim = m.simulate_cc_halfcycle(
        [p1, p2], y0, C_RATE, "charge",
        max_step=12, rtol=1.5e-4, atol=1.5e-6
    )
    z = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
    return {
        "common_LLI": background,
        "dLAMn": delta,
        "M_peak": z["M_peak"],
        "M_rms": z["M_rms"],
        "qex_norm": z["Q_excess_Ahm2"] / (0.5 * (q1 + q2)),
        "capdiff": (q2 - q1) / (0.5 * (q1 + q2)),
        "Q1": q1,
        "Q2": q2,
        "duration_min": sim["t"][-1] / 60.0,
        "n_steps": len(sim["t"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-bg", type=float, default=None,
                        help="common LLI fraction for one full MP-SPMe verification point")
    parser.add_argument("--full-dlamn", type=float, default=None,
                        help="additional LAMn fraction in cell 2 for one full MP-SPMe verification point")
    args = parser.parse_args()

    if args.full_bg is not None or args.full_dlamn is not None:
        if args.full_bg is None or args.full_dlamn is None:
            parser.error("--full-bg and --full-dlamn must be supplied together")
        print(pd.Series(run_full_point(args.full_bg, args.full_dlamn)).to_string())
    else:
        _, thresholds = run_dense_reduced()
        print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
