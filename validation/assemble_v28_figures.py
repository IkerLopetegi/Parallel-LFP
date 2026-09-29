"""Copy the verified corrected figures into the manuscript and SI source trees."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "recomputed"
PAPER = ROOT.parent / "final_paper"
SI = ROOT.parent / "final_si" / "supplementary"

MAIN = (
    "Figure02_degradation_hysteresis", "Figure03_lowrate_characterization",
    "Figure04_parallel_dynamics", "Figure05_model_fidelity",
    "Figure06_parameter_robustness", "Figure07_NP_design_sensitivity",
    "Figure08_threshold_heterogeneity", "Figure09_resistance_surface_area",
)
SUPPLEMENT = (
    "FigureS01_grid_convergence", "FigureS02_ZK_verification",
    "FigureS03_capacity_protocol_sensitivity", "FigureS04_Crate_sensitivity",
    "FigureS05_graphite_diffusivity_sensitivity", "FigureS06_LAMp_NP_maps",
    "FigureS07_threshold_summary", "FigureS08_integrated_heterogeneity",
    "FigureS09_same_mixed_trajectory", "FigureS10_negative_electrode_potential",
)


def main():
    for name in MAIN:
        source = OUT / f"{name}.pdf"
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copyfile(source, PAPER / source.name)
    for name in SUPPLEMENT:
        source = OUT / "supplementary" / f"{name}.pdf"
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copyfile(source, SI / source.name)
    print("Copied all eight computed main figures and S1--S10")


if __name__ == "__main__":
    main()
