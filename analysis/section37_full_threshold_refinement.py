"""Full MP-SPMe refinement of Section 3.7 degradation-heterogeneity thresholds.

For each aging background, this script finds the smallest branch-to-branch
heterogeneity for which M_peak reaches 0.80 and 0.95. Two mechanistic families
are considered:

1) common_LLI_dLAMn
   Cell 1: LLI = s, LAMn = 0
   Cell 2: LLI = s, LAMn = delta

2) same_LLI_LAMn_trajectory
   Cell 1: LLI = LAMn = s
   Cell 2: LLI = LAMn = s + delta

The search is performed with the full MP-SPMe at 1C charge, initialized at a
common 3.30 V. A coarse full-model scan brackets each threshold, then a 0.05
percentage-point refinement is performed inside the bracket.
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_section37"
OUT.mkdir(exist_ok=True)

BACKGROUNDS = (0.00, 0.05, 0.10, 0.15)
TARGETS = (0.80, 0.95)
COARSE_STEP = 0.005
FINE_STEP = 0.0005
MAX_DELTA = {
    "common_LLI_dLAMn": 0.06,
    "same_LLI_LAMn_trajectory": 0.07,
}


def build_pair(base, family, background, delta):
    if family == "common_LLI_dLAMn":
        p1 = m.make_mixed_degraded_cell(base, background, 0.0, 0.0, "decoupled")
        p2 = m.make_mixed_degraded_cell(base, background, delta, 0.0, "decoupled")
    elif family == "same_LLI_LAMn_trajectory":
        p1 = m.make_mixed_degraded_cell(base, background, background, 0.0, "decoupled")
        p2 = m.make_mixed_degraded_cell(
            base, background + delta, background + delta, 0.0, "decoupled"
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    return p1, p2


def run_point(base, family, background, delta):
    p1, p2 = build_pair(base, family, background, delta)
    y0, _ = m.init_parallel_at_common_ocv([p1, p2], 3.30, 0.4)
    sim = m.simulate_cc_halfcycle(
        [p1, p2], y0, 1.0, "charge",
        max_step=12, rtol=1.5e-4, atol=1.5e-6
    )
    z = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
    k = int(np.argmax(np.abs(sim["I"][:, 0] - sim["I"][:, 1])))
    recipient = int(np.argmax(np.abs(sim["I"][k]))) + 1
    return {
        "family": family,
        "background": background,
        "delta": delta,
        "M_peak": float(z["M_peak"]),
        "M_rms": float(z["M_rms"]),
        "peak_recipient": recipient,
        "duration_min": float(sim["t"][-1] / 60.0),
    }


def evaluate_cached(base, cache, family, background, delta):
    key = (family, round(background, 8), round(delta, 8))
    if key not in cache:
        cache[key] = run_point(base, family, background, delta)
        print(
            family, f"background={100*background:.1f}%",
            f"delta={100*delta:.3f}%",
            f"M_peak={cache[key]['M_peak']:.5f}",
            flush=True,
        )
    return cache[key]


def find_threshold(base, cache, family, background, target):
    max_delta = MAX_DELTA[family]

    previous = evaluate_cached(base, cache, family, background, 0.0)
    bracket = None
    for delta in np.arange(COARSE_STEP, max_delta + 0.5 * COARSE_STEP, COARSE_STEP):
        current = evaluate_cached(base, cache, family, background, float(delta))
        if current["M_peak"] >= target:
            bracket = (previous["delta"], current["delta"])
            break
        previous = current

    if bracket is None:
        return None

    lo, hi = bracket
    for delta in np.arange(lo + FINE_STEP, hi + 0.5 * FINE_STEP, FINE_STEP):
        current = evaluate_cached(base, cache, family, background, float(delta))
        if current["M_peak"] >= target:
            previous_candidates = [
                r for r in cache.values()
                if r["family"] == family
                and np.isclose(r["background"], background)
                and r["delta"] < current["delta"]
                and r["M_peak"] < target
            ]
            below = max(previous_candidates, key=lambda r: r["delta"])
            return below, current
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=["common_LLI_dLAMn", "same_LLI_LAMn_trajectory", "all"],
        default="all",
    )
    args = parser.parse_args()

    families = list(MAX_DELTA) if args.family == "all" else [args.family]
    base = m.get_reference_params()
    cache = {}
    thresholds = []

    for family in families:
        for background in BACKGROUNDS:
            for target in TARGETS:
                ans = find_threshold(base, cache, family, background, target)
                if ans is None:
                    thresholds.append({
                        "family": family, "background": background, "target": target,
                        "delta_below": np.nan, "M_below": np.nan,
                        "delta_crit_sampled": np.nan, "M_peak": np.nan,
                        "M_rms": np.nan, "capacity_difference_pct": np.nan,
                    })
                    continue

                below, hit = ans
                p1, p2 = build_pair(base, family, background, hit["delta"])
                q1 = m.lowrate_capacity(p1, 0.05)
                q2 = m.lowrate_capacity(p2, 0.05)
                thresholds.append({
                    "family": family,
                    "background": background,
                    "target": target,
                    "delta_below": below["delta"],
                    "M_below": below["M_peak"],
                    "delta_crit_sampled": hit["delta"],
                    "M_peak": hit["M_peak"],
                    "M_rms": hit["M_rms"],
                    "Q1": q1,
                    "Q2": q2,
                    "capacity_difference_pct": 100 * (q2 - q1) / (0.5 * (q1 + q2)),
                    "bracket_width_pctpoints": 100 * (hit["delta"] - below["delta"]),
                })

    all_points = pd.DataFrame(cache.values()).sort_values(
        ["family", "background", "delta"]
    )
    summary = pd.DataFrame(thresholds)
    all_points.to_csv(OUT / "Section37_full_refined_all_points.csv", index=False)
    summary.to_csv(OUT / "Section37_full_thresholds_final.csv", index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
