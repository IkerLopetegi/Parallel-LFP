"""Run a single full-model robustness point (contact ASR input in ohm m^2)."""

from pathlib import Path
import argparse, json, sys, time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import robustness_v28 as r
from analysis.common import OUT, prepare_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("key")
    parser.add_argument("value", type=float)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    prepare_output()
    if not any(k == args.key and any(abs(v - args.value) <=
                                    1e-12 * max(abs(v), abs(args.value), 1e-20)
                                    for v in vals)
               for _, k, vals in r.VARIANTS):
        parser.error("Point is not in the V28 parameter grid")
    start = time.perf_counter()
    result = r.run_point(r.m.get_reference_params(), args.label, args.key, args.value)
    result = dict(
        runtime_s=time.perf_counter() - start,
        **result,
    )
    out = OUT / "robustness"
    out.mkdir(exist_ok=True)
    (out / f"{args.index}.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
