"""Run a single full-model robustness point (contact ASR input in ohm m^2)."""

from pathlib import Path
import argparse, json, sys, time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import run_final_v8 as r
from analysis.common import OUT, prepare_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("key")
    parser.add_argument("value", type=float)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    prepare_output()
    base = r.m.get_reference_params()
    target = r.m.make_degraded_cell(base, "LLI", 0.10)
    _, severity, _ = r.m.match_capacity_lowrate_ocvr(base, target, "LAMn")
    varied = r.m.apply_parameter_variant(base, args.key, args.value)
    p1 = r.m.make_degraded_cell(varied, "LLI", 0.10)
    p2 = r.m.make_degraded_cell(varied, "LAMn", severity)
    y, _ = r.m.init_parallel_at_common_ocv([p1, p2], 3.30)
    start = time.perf_counter()
    sim = r.m.simulate_cc_halfcycle(
        [p1, p2], y, 1.0, "charge", max_step=9, rtol=1.1e-4, atol=1e-6
    )
    result = dict(
        parameter=args.label,
        key=args.key,
        value=args.value,
        runtime_s=time.perf_counter() - start,
        **r.qex_norm(sim, p1, p2),
    )
    out = OUT / "robustness"
    out.mkdir(exist_ok=True)
    (out / f"{args.index}.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
