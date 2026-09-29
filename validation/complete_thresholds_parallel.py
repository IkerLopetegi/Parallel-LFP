"""Resume the five long threshold cases using the unchanged point evaluator.

The source run's JSON configuration and previously evaluated points are kept.
Only missing uniform-grid points are evaluated; the summary is then rebuilt
using the same first-crossing and capacity formulas as the original script.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m
from analysis import section37_full_threshold_refinement as threshold


DIRECTORY = Path(__file__).resolve().parents[1] / "results/recomputed/thresholds"
STEP = 0.00025
CASES = [("common_LLI_dLAMn", 0.0)] + [
    ("same_LLI_LAMn_trajectory", bg) for bg in (0.0, 0.05, 0.10, 0.15)
]
BASE = None


def initialize():
    global BASE
    BASE = m.get_reference_params()


def compute(task):
    family, background, index = task
    return threshold.run_point(BASE, family, background, round(index * STEP, 10))


def select_checkpoint(family, background):
    matches = []
    for path in DIRECTORY.glob("points_*.json"):
        config = json.loads(path.read_text())["config"]["arguments"]
        exact = config["family"] == family and config["background"] == background
        first_all = family == "common_LLI_dLAMn" and background == 0 and config["family"] == "all"
        if exact or first_all:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint for {(family, background)}; got {matches}")
    return matches[0]


def save(path, source):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(source, indent=2))
    temporary.replace(path)


def run_batch(executor, sources, limit):
    futures = {}
    for case, (path, source) in sources.items():
        family, background = case
        present = {round(p["delta"], 10) for p in source["points"] if
                   p["family"] == family and round(p["background"], 8) == background}
        for index in range(limit[case] + 1):
            delta = round(index * STEP, 10)
            if delta not in present:
                future = executor.submit(compute, (family, background, index))
                futures[future] = (case, path)
    print(f"Evaluating {len(futures)} remaining points", flush=True)
    for count, future in enumerate(as_completed(futures), 1):
        case, path = futures[future]
        point = future.result()
        if not np.isfinite(point["M_peak"]):
            raise m.NumericalError(f"Nonfinite point: {point}")
        sources[case][1]["points"].append(point)
        save(path, sources[case][1])
        if count % 10 == 0 or count == len(futures):
            print(f"{count}/{len(futures)} completed: {case}, "
                  f"delta={point['delta']:.5f}, M_peak={point['M_peak']:.6f}", flush=True)


def first_crossing(source, family, background, target):
    points = {round(p["delta"], 10): p for p in source["points"] if
              p["family"] == family and round(p["background"], 8) == background}
    maximum = threshold.MAX_DELTA[family]
    previous = None
    for index in range(math.floor(maximum / STEP + 1e-9) + 1):
        delta = round(index * STEP, 10)
        if delta not in points:
            raise RuntimeError(f"Missing {family}, {background}, {delta}")
        hit = points[delta]
        if hit["M_peak"] >= target:
            return previous, hit
        previous = hit
    return None


def main():
    expected_hash = hashlib.sha256(
        Path(m.__file__).read_bytes() + Path(threshold.__file__).read_bytes()
    ).hexdigest()
    sources = {}
    for case in CASES:
        path = select_checkpoint(*case)
        source = json.loads(path.read_text())
        if source["config"]["model_and_script_sha256"] != expected_hash or not np.isclose(
            source["config"]["arguments"]["fine_step"], STEP
        ):
            raise RuntimeError(f"Checkpoint code/settings mismatch: {path}")
        sources[case] = (path, source)
    limits = {case: (240 if case[0] == "common_LLI_dLAMn" else 200) for case in CASES}
    with ProcessPoolExecutor(max_workers=8, initializer=initialize) as executor:
        run_batch(executor, sources, limits)
        # Only cases without a 0.95 crossing by 5% need the remaining 5--7% grid.
        additional = {}
        for case in CASES[1:]:
            source = sources[case][1]
            points = {round(p["delta"], 10): p for p in source["points"] if
                      p["family"] == case[0] and round(p["background"], 8) == case[1]}
            if all(points[round(i * STEP, 10)]["M_peak"] < 0.95 for i in range(201)):
                additional[case] = 280
        if additional:
            run_batch(executor, {c: sources[c] for c in additional}, additional)

    base = m.get_reference_params()
    for case, (path, source) in sources.items():
        family, background = case
        rows = []
        for target in threshold.TARGETS:
            result = first_crossing(source, family, background, target)
            row = dict(family=family, background=background, target=target,
                       grid_step_fraction=STEP, status="not_reached")
            if result is not None:
                below, hit = result
                p1, p2 = threshold.build_pair(base, family, background, hit["delta"])
                q1, q2 = m.lowrate_capacity(p1), m.lowrate_capacity(p2)
                row.update(status="reached", delta_below=below["delta"] if below else np.nan,
                           M_below=below["M_peak"] if below else np.nan,
                           delta_crit_sampled=hit["delta"], M_peak=hit["M_peak"],
                           M_rms=hit["M_rms"], Q1=q1, Q2=q2,
                           capacity_difference_pct=100*(q2-q1)/(0.5*(q1+q2)),
                           bracket_width_pctpoints=100*(hit["delta"]-below["delta"]) if below else 0.0)
            rows.append(row)
        tag = path.stem.rsplit("_", 1)[-1]
        pd.DataFrame(rows).to_csv(DIRECTORY / f"Section37_full_thresholds_{tag}.csv", index=False)
        pd.DataFrame(source["points"]).to_csv(DIRECTORY / f"Section37_full_points_{tag}.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
