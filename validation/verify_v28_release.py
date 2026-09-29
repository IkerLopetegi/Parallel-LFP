"""Check the final V28 numerical result matrix and manuscript figure inventory."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.robustness_v28 import PROTOCOL_HASH, points
from analysis.threshold_figures_v28 import verified_data
from validation.assemble_v28_figures import MAIN, SUPPLEMENT

OUT = ROOT / "results" / "recomputed"


def check():
    robust = pd.read_csv(OUT / "Figure06_parameter_robustness.csv")
    expected = points()
    assert len(robust) == len(expected) == 25
    assert all(row.Key == key and np.isclose(row.Value, value, rtol=1e-12, atol=0)
               for row, (_, key, value) in zip(robust.itertuples(), expected))
    assert set(robust.protocol_sha256) == {PROTOCOL_HASH}
    assert robust.matching_error_Ahm2.abs().max() < 3e-8
    assert (robust.M_peak >= .95).sum() == 23
    assert np.isclose(robust.loc[(robust.Key == "ds_neg") &
                                 (robust.Value == 3e-15), "M_peak"].iloc[0], .563460, atol=1e-6)

    data, summary = verified_data()
    assert len(data) == 1371 and len(summary) == 16
    assert ((summary.status == "not_reached").sum() == 1)
    missed = summary[summary.status == "not_reached"].iloc[0]
    assert (missed.family, missed.background, missed.target) == ("common_LLI_dLAMn", 0., .95)
    assert data[(data.family == missed.family) & (data.background == 0)].M_peak.max() < .95
    assert np.isclose(summary.grid_step_fraction.max(), .00025)

    hysteresis = pd.read_csv(OUT / "Figure02_metrics.csv")
    assert set(hysteresis["mode"]) == {"LLI", "LAMn", "LAMp"}
    assert (hysteresis.C_rate == .5).all() and (hysteresis.severity == .2).all()
    assert np.allclose(hysteresis.charge_endpoint_V, 3.65, atol=1e-8)
    assert np.allclose(hysteresis.discharge_endpoint_V, 2.5, atol=1e-8)
    assert (hysteresis.charge_capacity_Ahm2 -
            hysteresis.half_step_charge_capacity_Ahm2).abs().max() < .003

    crate = pd.read_csv(OUT / "supplementary" / "TableS01_Crate_sensitivity.csv")
    assert list(crate.C_rate) == [.1, .25, .5, 1.]
    assert crate[["M_peak", "M_rms", "qex_norm"]].notna().all().all()
    assert np.isclose(crate.M_peak.iloc[0], 1.046655, atol=1e-5)
    diagnostic = pd.read_csv(OUT / "supplementary" /
                             "TableS01_Crate_sensitivity_reduced_diagnostic.csv")
    assert diagnostic.status.iloc[0] == "electrode_bound_first"
    assert np.isnan(diagnostic.M_peak.iloc[0])

    maps = pd.read_csv(OUT / "FigureS06_LAMp_NP_maps.csv")
    assert len(maps) == 242
    invalid = maps[maps.status == "electrode_bound_first"]
    assert len(invalid) == 22 and invalid.M_peak.isna().all()
    assert maps[maps.status == "voltage_cutoff"].M_peak.notna().all()

    for name in MAIN:
        assert (OUT / f"{name}.pdf").stat().st_size > 1000
        assert (ROOT.parent / "final_paper" / f"{name}.pdf").read_bytes() == (
            OUT / f"{name}.pdf").read_bytes()
    for name in SUPPLEMENT:
        assert (OUT / "supplementary" / f"{name}.pdf").stat().st_size > 1000
        assert (ROOT.parent / "final_si" / "supplementary" / f"{name}.pdf").read_bytes() == (
            OUT / "supplementary" / f"{name}.pdf").read_bytes()
    assert (ROOT.parent / "final_paper" /
            "Parallel_LFP_Electrochimica_Acta_V23_submission.pdf").stat().st_size > 10000
    assert (ROOT.parent / "final_si" / "paper" /
            "Supplementary_Information.pdf").stat().st_size > 10000
    print("PASS: 25 variants, 1,371 threshold points, 242 map points, and all V28 figures")


if __name__ == "__main__":
    check()
