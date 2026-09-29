"""Map verified corrected outputs to the figure/table names used in V28."""

from pathlib import Path
import shutil
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import OUT, SUP, prepare_output


def copy_figure(old, new):
    for ext in ("pdf", "png"):
        source = SUP / f"{old}.{ext}"
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copyfile(source, SUP / f"{new}.{ext}")


def main():
    prepare_output()
    copy_figure("FigureS02_grid_convergence", "FigureS01_grid_convergence")
    copy_figure("FigureS01_ZK_verification", "FigureS02_ZK_verification")
    copy_figure("FigureS05_same_mixed_trajectory", "FigureS09_same_mixed_trajectory")
    # The corrected full MP-SPMe reaches all four voltage cutoffs. The
    # reduced-model 0.1C diagnostic has no cutoff-reaching metric.
    source = SUP / "TableS06_Crate_sensitivity.csv"
    df = pd.read_csv(source)
    assert list(df.C_rate) == [0.1, 0.25, 0.5, 1.0]
    assert df[["M_peak", "M_rms", "qex_norm"]].notna().all().all()
    reduced = SUP / "TableS01_Crate_sensitivity.csv"
    if reduced.exists() and "status" in pd.read_csv(reduced, nrows=0).columns:
        shutil.copyfile(reduced, SUP / "TableS01_Crate_sensitivity_reduced_diagnostic.csv")
    df.to_csv(reduced, index=False)
    copy_figure("FigureS06_Crate_sensitivity", "FigureS04_Crate_sensitivity")
    for name in ("FigureS01_grid_convergence", "FigureS02_ZK_verification",
                 "FigureS03_capacity_protocol_sensitivity", "FigureS04_Crate_sensitivity",
                 "FigureS05_graphite_diffusivity_sensitivity", "FigureS06_LAMp_NP_maps",
                 "FigureS07_threshold_summary", "FigureS08_integrated_heterogeneity",
                 "FigureS09_same_mixed_trajectory", "FigureS10_negative_electrode_potential"):
        assert (SUP / f"{name}.pdf").exists(), name
    print("Verified Figures S1--S10 and Table S1 naming")


if __name__ == "__main__":
    main()
