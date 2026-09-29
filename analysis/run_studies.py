"""Run a selected numerical study; use --study all for the historical collection."""

from pathlib import Path
import argparse, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import run_final_v8 as r
from analysis.common import prepare_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=[
            "capacities",
            "hysteresis",
            "dynamics",
            "fidelity",
            "robustness",
            "maps",
            "resistance",
            "supplementary",
            "all",
        ],
        default="capacities",
    )
    args = parser.parse_args()
    if args.study == "all":
        r.main()
        return
    prepare_output()
    base = r.m.get_reference_params()
    if args.study == "hysteresis":
        r.figure1_hysteresis(base)
    elif args.study == "capacities":
        r.figure3_lowrate(base)
    elif args.study == "maps":
        r.figure7_pure_maps(base)
        r.realistic_maps(base)
    elif args.study == "resistance":
        r.figure9_resistance(base)
    else:
        target = r.m.make_degraded_cell(base, "LLI", 0.10)
        _, severity, _ = r.m.match_capacity_lowrate_ocvr(base, target, "LAMn")
        if args.study == "supplementary":
            r.supplementary(base, severity, None)
        else:
            {
                "dynamics": r.figure4_dynamics,
                "fidelity": r.figure5_model_fidelity,
                "robustness": r.figure6_robustness,
            }[args.study](base, severity)


if __name__ == "__main__":
    main()
