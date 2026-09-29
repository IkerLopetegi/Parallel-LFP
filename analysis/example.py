"""Small end-to-end reduced-model example (no full parameter sweep)."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfp_parallel import model as m
from analysis.common import OUT, prepare_output


def main():
    prepare_output()
    base = m.get_reference_params()
    cell1 = m.make_mixed_degraded_cell(base, lli=0.05)
    cell2 = m.make_mixed_degraded_cell(base, lli=0.05, lamn=0.05)
    sim = m.simulate_ocvr_pair_fast_stateR(cell1, cell2, 1.0, "charge", dq_frac=0.002)
    metrics = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
    pd.DataFrame(
        {
            "time_s": sim["t"],
            "voltage_V": sim["V"],
            "current1_Am2": sim["I"][:, 0],
            "current2_Am2": sim["I"][:, 1],
        }
    ).to_csv(OUT / "example.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
