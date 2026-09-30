"""Find the first threshold crossing on a uniform full-model heterogeneity grid.

Unlike coarse-bracket-only refinement, every smaller grid point is checked.
The default spacing is 0.025 percentage points. This is a sampling resolution,
not an uncertainty bound on the physics or time integration. Runs checkpoint
one point at a time and only resume with exactly matching code/parameters.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m
from analysis.common import OUT, prepare_output

BACKGROUNDS = (0.0, 0.05, 0.10, 0.15)
TARGETS = (0.80, 0.95)
FAMILIES = ("common_LLI_dLAMn", "same_LLI_LAMn_trajectory")
FINE_STEP = 0.00025
MAX_DELTA = {"common_LLI_dLAMn": 0.06, "same_LLI_LAMn_trajectory": 0.07}


def build_pair(base, family, background, delta):
    if family == FAMILIES[0]:
        return (
            m.make_mixed_degraded_cell(base, background, 0, 0, "decoupled"),
            m.make_mixed_degraded_cell(base, background, delta, 0, "decoupled"),
        )
    if family == FAMILIES[1]:
        return (
            m.make_mixed_degraded_cell(base, background, background, 0, "decoupled"),
            m.make_mixed_degraded_cell(
                base, background + delta, background + delta, 0, "decoupled"
            ),
        )
    raise ValueError(f"Unknown degradation family: {family}")


def run_point(base, family, background, delta):
    p1, p2 = build_pair(base, family, background, delta)
    y, _ = m.init_parallel_at_common_ocv([p1, p2], 3.30, 0.4)
    sim = m.simulate_cc_halfcycle(
        [p1, p2], y, 1.0, "charge", max_step=4, rtol=1e-5, atol=1e-7
    )
    metrics = m.compute_current_metrics(sim["t"], sim["I"], sim["Iapp"])
    peak = np.argmax(np.abs(sim["I"][:, 0] - sim["I"][:, 1]))
    return dict(
        family=family,
        background=background,
        delta=delta,
        M_peak=metrics["M_peak"],
        M_rms=metrics["M_rms"],
        peak_recipient=int(np.argmax(np.abs(sim["I"][peak]))) + 1,
        duration_min=float(sim["t"][-1] / 60),
    )


def find_threshold(
    base,
    cache,
    family,
    background,
    target,
    step=FINE_STEP,
    max_delta=None,
    on_evaluate=None,
):
    """First sampled crossing; no monotonicity assumption between coarse points."""
    maximum = MAX_DELTA[family] if max_delta is None else max_delta
    if step <= 0 or maximum < 0 or not 0 < target:
        raise ValueError("Invalid threshold search bounds")
    previous = None
    for index in range(int(np.floor(maximum / step + 1e-9)) + 1):
        delta = round(index * step, 10)
        key = (family, round(background, 8), delta)
        if key not in cache:
            point = run_point(base, family, background, delta)
            if not np.isfinite(point["M_peak"]):
                raise m.NumericalError("Nonfinite threshold metric")
            cache[key] = point
            if on_evaluate is not None:
                on_evaluate(cache)
            print(
                f'{family}: background={background:.3f}, delta={delta:.5f}, M_peak={point["M_peak"]:.6f}',
                flush=True,
            )
        current = cache[key]
        if current["M_peak"] >= target:
            return previous, current
        previous = current
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument(
        "--background",
        type=float,
        default=None,
        help="Degradation fraction; default is all four backgrounds",
    )
    parser.add_argument(
        "--fine-step",
        type=float,
        default=FINE_STEP,
        help="Fraction, default 0.00025 = 0.025 percentage points",
    )
    args = parser.parse_args()
    if args.fine_step <= 0 or (
        args.background is not None and not 0 <= args.background <= 0.15
    ):
        parser.error("Require positive fine step and background in [0, .15]")
    prepare_output()
    output = OUT / "thresholds"
    output.mkdir(exist_ok=True)
    model_hash = hashlib.sha256(
        Path(m.__file__).read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    config = dict(
        model_and_script_sha256=model_hash, arguments=vars(args), numpy=np.__version__
    )
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    checkpoint = output / f"points_{tag}.json"
    cache = {}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved["config"] != config:
            raise RuntimeError("Checkpoint configuration mismatch")
        for r in saved["points"]:
            cache[(r["family"], round(r["background"], 8), round(r["delta"], 10))] = r

    def save(cache):
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dict(config=config, points=list(cache.values())), indent=2)
        )
        temporary.replace(checkpoint)

    base = m.get_reference_params()
    families = FAMILIES if args.family == "all" else (args.family,)
    backgrounds = BACKGROUNDS if args.background is None else (args.background,)
    rows = []
    for family in families:
        for background in backgrounds:
            for target in TARGETS:
                result = find_threshold(
                    base,
                    cache,
                    family,
                    background,
                    target,
                    args.fine_step,
                    on_evaluate=save,
                )
                row = dict(
                    family=family,
                    background=background,
                    target=target,
                    grid_step_fraction=args.fine_step,
                    status="not_reached",
                )
                if result is not None:
                    below, hit = result
                    p1, p2 = build_pair(base, family, background, hit["delta"])
                    q1, q2 = m.lowrate_capacity(p1), m.lowrate_capacity(p2)
                    row.update(
                        status="reached",
                        delta_below=below["delta"] if below else np.nan,
                        M_below=below["M_peak"] if below else np.nan,
                        delta_crit_sampled=hit["delta"],
                        M_peak=hit["M_peak"],
                        M_rms=hit["M_rms"],
                        Q1=q1,
                        Q2=q2,
                        capacity_difference_pct=100 * (q2 - q1) / (0.5 * (q1 + q2)),
                        bracket_width_pctpoints=(
                            100 * (hit["delta"] - below["delta"]) if below else 0.0
                        ),
                    )
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    output / f"Section37_full_thresholds_{tag}.csv", index=False
                )
    pd.DataFrame(cache.values()).to_csv(
        output / f"Section37_full_points_{tag}.csv", index=False
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
