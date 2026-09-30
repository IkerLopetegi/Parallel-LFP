"""Audit and combine independently checkpointed full threshold searches."""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = ROOT / "results" / "thresholds"
EXPECTED_FAMILIES = ("common_LLI_dLAMn", "same_LLI_LAMn_trajectory")
BACKGROUNDS = (0.0, 0.05, 0.10, 0.15)
TARGETS = (0.8, 0.95)
MAX_DELTA = {"common_LLI_dLAMn": 0.06, "same_LLI_LAMn_trajectory": 0.07}


def main():
    audited = {}
    provenance = []
    model_hashes = set()
    for path in sorted(THRESHOLDS.glob("Section37_full_thresholds_*.csv")):
        if path.name == "Section37_full_thresholds_combined.csv":
            continue
        tag = path.stem.rsplit("_", 1)[-1]
        checkpoint = THRESHOLDS / f"points_{tag}.json"
        if not checkpoint.exists():
            raise RuntimeError(f"Missing point checkpoint for {path.name}")
        source = json.loads(checkpoint.read_text())
        config = source["config"]
        model_hashes.add(config["model_and_script_sha256"])
        step = float(config["arguments"]["fine_step"])
        if not np.isclose(step, 0.00025):
            raise RuntimeError(f"Unexpected grid spacing in {path.name}")
        points = {
            (p["family"], round(p["background"], 8), round(p["delta"], 10)): p
            for p in source["points"]
        }
        table = pd.read_csv(path)
        for _, r in table.iterrows():
            family, background, target = r.family, round(float(r.background), 8), float(r.target)
            key = (family, background, target)
            if family not in EXPECTED_FAMILIES or background not in BACKGROUNDS or target not in TARGETS:
                raise RuntimeError(f"Unexpected threshold combination: {key}")
            if r.status == "reached":
                last = int(round(float(r.delta_crit_sampled) / step))
            elif r.status == "not_reached":
                last = math.floor(MAX_DELTA[family] / step + 1e-9)
            else:
                raise RuntimeError(f"Unknown status in {path.name}: {r.status}")
            series = []
            for i in range(last + 1):
                sample = (family, background, round(i * step, 10))
                if sample not in points:
                    raise RuntimeError(f"Missing point {sample} in {checkpoint.name}")
                series.append(points[sample]["M_peak"])
            if any(not np.isfinite(p) for p in series):
                raise RuntimeError(f"Nonfinite metric for {key}")
            if r.status == "reached":
                if any(p >= target for p in series[:-1]) or series[-1] < target:
                    raise RuntimeError(f"Incorrect first crossing for {key}")
                if not np.isclose(series[-1], r.M_peak, atol=1e-9):
                    raise RuntimeError(f"Point/table mismatch for {key}")
            elif any(p >= target for p in series):
                raise RuntimeError(f"Unreported crossing for {key}")
            if key in audited:
                prior = audited[key]
                if prior.status != r.status or (
                    r.status == "reached" and not np.isclose(prior.delta_crit_sampled, r.delta_crit_sampled)
                ):
                    raise RuntimeError(f"Conflicting rows for {key}")
            else:
                audited[key] = r
            provenance.append({"family": family, "background": background, "target": target,
                               "source": path.name, "points_checked": len(series)})
    expected = {(f, b, t) for f in EXPECTED_FAMILIES for b in BACKGROUNDS for t in TARGETS}
    if set(audited) != expected or len(model_hashes) != 1:
        raise RuntimeError(f"Incomplete or inconsistent thresholds: missing {sorted(expected - set(audited))}")
    table = pd.DataFrame([audited[key] for key in sorted(audited)])
    table.to_csv(THRESHOLDS / "Section37_full_thresholds_combined.csv", index=False)
    (THRESHOLDS / "reproduction_audit.json").write_text(json.dumps({
        "model_and_script_sha256": model_hashes.pop(),
        "grid_step_fraction": 0.00025,
        "rows": len(table),
        "sources": provenance,
    }, indent=2) + "\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
