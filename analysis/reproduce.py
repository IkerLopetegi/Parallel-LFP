"""Reproduce the final paper and SI; run from the repository root.

All numerical studies can be expensive. Existing full-model threshold and
robustness checkpoints resume only when the numerical configuration matches.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil
from analysis.common import prepare_output
from lfp_parallel import model as m

STAGES = (
    "verify",
    "figures",
    "saved",
    "core",
    "dynamics",
    "robustness",
    "np",
    "np_tolerance",
    "thresholds",
    "supplementary",
    "resistance",
    "diffusivity",
    "potential",
    "lamp_map",
    "checks",
    "all",
)


def run_module(name, *arguments):
    subprocess.run([sys.executable, "-m", "analysis." + name, *arguments], check=True)


def run_stage(stage):
    prepare_output()
    if stage == "all":
        for name in (
            "core",
            "robustness",
            "np",
            "np_tolerance",
            "thresholds",
            "supplementary",
            "resistance",
            "diffusivity",
            "potential",
            "checks",
            "saved",
            "verify",
        ):
            run_stage(name)
    elif stage == "verify":
        run_module("verify_results")
    elif stage == "figures":
        run_stage("saved")
    elif stage == "saved":
        for part in ("saved", "si", "graphical"):
            run_module("publication_figures", "--stage", part)
    elif stage == "core":
        run_module("publication_figures", "--stage", "core")
    elif stage == "dynamics":
        from analysis.core_studies import figure4_dynamics, figure5_model_fidelity

        base = m.get_reference_params()
        target = m.make_degraded_cell(base, "LLI", 0.10)
        _, severity, _ = m.match_capacity_lowrate_ocvr(
            base, target, "LAMn", bounds=(0, 0.25), C_rate=0.05
        )
        figure4_dynamics(base, severity)
        figure5_model_fidelity(base, severity)
    elif stage == "robustness":
        from analysis.robustness import run_all

        run_all()
    elif stage == "np":
        run_module("np_sensitivity")
    elif stage == "np_tolerance":
        run_module("np_tolerance_check")
    elif stage == "thresholds":
        from analysis.threshold_scan import FAMILIES, BACKGROUNDS

        for family in FAMILIES:
            for bg in BACKGROUNDS:
                run_module(
                    "threshold_scan", "--family", family, "--background", str(bg)
                )
        run_module("assemble_thresholds")
        run_module("threshold_figures")
    elif stage == "supplementary":
        from analysis.core_studies import supplementary_studies
        from analysis.supplementary_data import charge_family_maps, mixed_trajectory_map
        from analysis.supplementary_maps import lamp_map

        base = m.get_reference_params()
        lli = m.make_degraded_cell(base, "LLI", 0.10)
        _, sn, _ = m.match_capacity_lowrate_ocvr(
            base, lli, "LAMn", bounds=(0, 0.25), C_rate=0.05
        )
        supplementary_studies(base, sn)
        charge_family_maps(base)
        mixed_trajectory_map(base)
        lamp_map()
    elif stage == "lamp_map":
        run_module("fullmodel_lamp_map")
    elif stage == "resistance":
        from analysis.core_studies import figure9_resistance

        figure9_resistance(m.get_reference_params())
    elif stage == "diffusivity":
        run_module("diffusivity_sensitivity")
    elif stage == "potential":
        run_module("negative_electrode_potential_sensitivity")
    elif stage == "checks":
        run_module("supplementary_checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="verify")
    parser.add_argument(
        "--fresh-output",
        type=Path,
        help="Run in a new directory without supplied numerical results",
    )
    args = parser.parse_args()
    if args.fresh_output is not None:
        destination = args.fresh_output.resolve()
        destination.mkdir(parents=True, exist_ok=False)
        source = Path(__file__).resolve().parents[1]
        for name in ("analysis", "lfp_parallel", "figures"):
            shutil.copytree(
                source / name,
                destination / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        subprocess.run(
            [sys.executable, "-m", "analysis.reproduce", "--stage", args.stage],
            cwd=destination,
            check=True,
        )
    else:
        run_stage(args.stage)


if __name__ == "__main__":
    main()
