"""Tighter-tolerance verification of representative points in Figure 7."""

import numpy as np
import pandas as pd
from analysis.common import OUT, SUP, prepare_output
from analysis.np_sensitivity import base_at_np
from lfp_parallel import model as m

CASES = (
    (1.075, "LAMn", "charge", 3.30),
    (1.125, "LLI", "charge", 3.30),
    (1.20, "LAMp", "discharge", 3.35),
)


def comparison_table(tighter):
    reference = pd.read_csv(OUT / "Figure07_NP_design_sensitivity.csv")
    rows = []
    for case in tighter:
        match = reference[
            (reference["mode"] == case["mode"])
            & np.isclose(reference.NP_ratio, case["NP_ratio"])
        ]
        if len(match) != 1:
            raise RuntimeError("Complete Figure 7 data are required")
        row = dict(case)
        for key in ("M_peak", "qex_norm"):
            row[key + "_reference"] = float(match[key].iloc[0])
        row["qex_relative_difference"] = abs(
            row["qex_norm"] / row["qex_norm_reference"] - 1
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(SUP / "NP_solver_tolerance_check.csv", index=False)
    return result


def main():
    prepare_output()
    base = m.get_reference_params()
    rows = []
    for ratio, mode, direction, voltage in CASES:
        design = base_at_np(base, ratio)
        cells = [design, m.make_degraded_cell(design, mode, 0.20)]
        y, _ = m.init_parallel_at_common_ocv(
            cells, voltage, 0.4 if direction == "charge" else 0.75
        )
        sim = m.simulate_cc_halfcycle(
            cells, y, 1.0, direction, max_step=4, rtol=1e-5, atol=1e-7
        )
        met = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
        qbar = 0.5 * sum(m.lowrate_capacity(p) for p in cells)
        rows.append(
            dict(
                NP_ratio=ratio,
                mode=mode,
                M_peak=met["M_peak"],
                qex_norm=met["Q_excess_Ahm2"] / qbar,
            )
        )
        print(rows[-1], flush=True)
    print(comparison_table(rows).to_string(index=False))


if __name__ == "__main__":
    main()
