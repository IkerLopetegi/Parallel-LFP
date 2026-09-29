"""Run the full reference-grid kinetic and transport sensitivity study."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import run_final_v8 as r
from analysis.common import prepare_output


def main():
    prepare_output()
    base = r.m.get_reference_params()
    target = r.m.make_degraded_cell(base, "LLI", 0.10)
    _, severity, _ = r.m.match_capacity_lowrate_ocvr(base, target, "LAMn")
    r.figure6_robustness(base, severity)


if __name__ == "__main__":
    main()
