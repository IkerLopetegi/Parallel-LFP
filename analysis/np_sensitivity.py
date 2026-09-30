"""Main-text N/P study: change positive loading at fixed negative electrode/inventory.

N/P uses full 0--1 intercalation spans for LFP and graphite: Qn/Qp.
These are ideal model host capacities, not measured degradation-free windows.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m
from analysis.common import OUT, prepare_output


def design_np(p):
    return p["Qn"] / p["Qp"]


def base_at_np(base, target):
    if target <= 0:
        raise ValueError("N/P must be positive")
    p = copy.deepcopy(base)
    p["geom"]["L_pos"] *= design_np(base) / target
    p = m.update_derived_params(p)
    # Inventory, nominal LLI reference and negative geometry remain fixed.
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=list(np.linspace(0.90, 1.20, 13))
    )
    args = parser.parse_args()
    prepare_output()
    base = m.get_reference_params()
    protocol=dict(capacity_convention='full_0_1_intercalation_host_capacity',
                  graphite_span=[0.,1.], LFP_span=[0.,1.],
                  reference_NP=design_np(base), target_ratios=args.ratios,
                  loading_path='positive_thickness', fixed_negative_electrode=True,
                  fixed_lithium_inventory=True, fixed_nominal_capacity_reference=True,
                  degradation_fraction=.20, C_rate=1.0,
                  grid=base['disc'], rtol=8e-5, atol=7e-7,max_step_s=8,
                  model_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'lfp_parallel/model.py').read_bytes()).hexdigest())
    (OUT/'Figure07_protocol.json').write_text(json.dumps(protocol,indent=2))
    rows = []
    for ratio in args.ratios:
        p1 = base_at_np(base, ratio)
        fresh = m.lowrate_ocvr_characterization(p1)
        for mode, direction, voltage in [
            ("LLI", "charge", 3.30),
            ("LAMn", "charge", 3.30),
            ("LAMp", "discharge", 3.35),
        ]:
            p2 = m.make_degraded_cell(p1, mode, 0.20)
            y, _ = m.init_parallel_at_common_ocv(
                [p1, p2], voltage, 0.4 if direction == "charge" else 0.75
            )
            sim = m.simulate_cc_halfcycle(
                [p1, p2], y, 1.0, direction, max_step=8, rtol=8e-5, atol=7e-7
            )
            met = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
            row = dict(
                NP_ratio=design_np(p1),
                L_pos_um=p1["geom"]["L_pos"] * 1e6,
                Qn_host_Ahm2=p1["Qn"] / 3600,
                Qp_host_Ahm2=p1["Qp"] / 3600,
                mode=mode,
                M_peak=met["M_peak"],
                M_rms=met["M_rms"],
                qex_norm=met["Q_excess_Ahm2"]
                / (0.5 * (m.lowrate_capacity(p1) + m.lowrate_capacity(p2))),
                positive_utilization=fresh["xp_low"] - fresh["xp_high"],
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(
                OUT / "Figure07_NP_design_sensitivity.csv", index=False
            )
            print(row, flush=True)
    from analysis.publication_figures import render_np_figure
    render_np_figure()


if __name__ == "__main__":
    main()
