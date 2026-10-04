# Parallel LFP model: final paper results

Simulation code and final numerical data for the manuscript and supplementary information.

## Installation and verification

Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -v
python -m analysis.reproduce --stage verify
python -m analysis.verify_results --manifest # exact release files before regeneration
```

The numerical model is in `lfp_parallel/model.py`. See `MODEL_GUIDE.md` for units, numerical methods and physical assumptions.

## Reproduction

Run commands from the repository root:

```bash
python -m analysis.reproduce --stage saved   # plots supported by saved tables
python -m analysis.reproduce --stage figures # all figure assets; includes representative simulations
python -m analysis.reproduce --stage all     # all numerical studies and final plots
```

Full-model simulations and threshold scans can take substantial time. Threshold and robustness studies reuse compatible saved checkpoints. Remove their outputs in a separate checkout to recompute them from scratch. Threshold JSON files describe the current numerical configuration and completed points; they are needed for safe resumption.

Run `python -m analysis.fullmodel_lamp_map --workers 6` to recompute Figure S6 using independent processes, or append `--plot-only` to render the retained map. Checkpoints are reused only for the same full-model protocol; the earlier OCV-R map is not reused. Every case must reach the 2.50 V cutoff; numerical failures stop the runner.

`--stage` also accepts `core`, `robustness`, `np`, `thresholds`, `supplementary`, `resistance`, `diffusivity`, `potential`, and `checks`. The default is `verify`, which checks data consistency without rerunning simulations. Use `analysis.verify_results --manifest` to check exact release checksums before regeneration; regenerated PDFs can have different metadata.

## Result map

Final vector PDFs and numerical tables are in `results/`; supplementary assets are in `results/supplementary/`. Figure 1 is the author-supplied schematic in `figures/`.

| Paper result | Numerical recipe / retained data |
|---|---|
| Figure 2 | `figure02_hysteresis.py`; full MP-SPMe trajectories, protocol and metrics |
| Figure 3 | `figure03_fullmodel.py`; full MP-SPMe capacity matching, trajectories, protocol and solver checks |
| Figures 4–5 | `core_studies.py`; dynamics and model-fidelity tables |
| Figure 6 | `robustness.py`; 25 variants and protocol |
| Figure 7 | `np_sensitivity.py`; N/P design table |
| Figure 8; SI S7 | `threshold_scan.py`, `assemble_thresholds.py`, `threshold_figures.py`; eight configurations and combined thresholds |
| Figure 9 | `core_studies.py`; contact-resistance and exchange-area maps |
| SI S2 | `publication_figures.py`; analytical particle-size-dependent regular-solution curves |
| SI S3 | `core_studies.py`; capacity-protocol sensitivity |
| SI S1, S4 | `core_studies.py`; grid-convergence and C-rate tables |
| SI S5 | `diffusivity_sensitivity.py`; graphite diffusivity sensitivity |
| SI S6 | `fullmodel_lamp_map.py`; 242 full MP-SPMe discharge cases, protocol, selected histories and solver checks |
| SI S8 | `supplementary_maps.py`, `supplementary_data.py`; reduced-model integrated maps |
| SI S9 | `supplementary_data.py`; mixed paths [s, 0.5s, 0.5s] in FigureS09 CSV |
| SI S10 | `negative_electrode_potential_sensitivity.py`; three full time histories and potential summary |
| SI S11 | `manuscript_revision_checks.py`; electrode trajectories |
| SI protocol, controlled N/P and cutoff tables | `core_studies.py`, `manuscript_revision_checks.py`; capacity-protocol, matched-N/P and cutoff CSVs |
