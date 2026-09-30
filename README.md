# Parallel LFP/graphite cells — corrected source package

Physics-based and reduced models accompanying *Degradation-Induced Electrode Misalignment and Current Redistribution in Parallel LFP/Graphite Cells*.

This package contains the corrected V28 source and regenerated results used in the revised manuscript and Supplementary Information. Historical CSVs are preserved in `results_published/`; final computed outputs and their source-bound threshold checkpoints are under `results/recomputed/`. Read `FINAL_RELEASE_REPORT.md`, `CORRECTIONS.md`, and `VALIDATION.md` for the numerical changes and scope of verification.

`FINAL_RELEASE_MANIFEST.sha256` records checksums of the final release files. Run `sha256sum -c FINAL_RELEASE_MANIFEST.sha256` from the repository root to check a downloaded copy.

## Quick start

Python 3.10 or newer:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis/example.py
```

The example runs a reduced-model charge step for common 5% LLI with 5% additional LAMn in cell 2. It prints current metrics and writes a time history. No LaTeX installation is required for figures. `requirements-tested.txt` records the versions used for validation; the broader minimum requirements are in `requirements.txt`.

To install the model as a package, use `python -m pip install -e .`.

## Source map

| File | Purpose |
|---|---|
| `lfp_parallel/model.py` | Parameters, thermodynamics, degradation, diffusion, full and reduced solvers |
| `MODEL_GUIDE.md` | Units, state layout, model choices and numerical contracts |
| `analysis/common.py` | Output paths and plotting configuration |
| `analysis/example.py` | Small, runnable starting example |
| `tests/test_model.py` | Conservation, symmetry, blocking, solver and workflow regressions |
| `results_published/` | Historical CSVs from the reviewed GitHub snapshot, unchanged |
| `results/recomputed/` | Results actually regenerated using corrected code |
| `validation/` | Audit summaries and reproducible validation driver |

All source is unpacked and directly browsable. The archive can be extracted into a clean directory; do not overlay it onto a previous result tree when making comparisons.

## Analyses

For a single study, for example: `python analysis/run_studies.py --study dynamics`. Run `python analysis/run_studies.py --help` for choices.

Run commands from the package root. Full-model studies are more expensive than the example. Threshold scans can require hundreds of integrations per family/background; they checkpoint each point and resume when code and settings match.

| Study | Command |
|---|---|
| Core numerical collection: capacities, dynamics, fidelity, resistance, supplementary studies | `python analysis/run_final_v8.py` |
| Figure 2, 20% degradation C/2 illustration | `python analysis/figure02_hysteresis.py` |
| Main N/P sensitivity: vary positive loading, fixed negative electrode/inventory | `python analysis/np_sensitivity.py` |
| Alternative N/P path: vary negative loading | `python analysis/np_sensitivity_anode_loading.py` |
| Graphite diffusivity comparison and separate reduced-model C-rate diagnostic | `python analysis/revision_v12_diffusivity.py` |
| Graphite potential diagnostic using full BV kinetics | `python analysis/negative_electrode_potential_sensitivity.py` |
| Dense reduced-model heterogeneity thresholds | `python analysis/section37_heterogeneity_thresholds.py` |
| One full verification point | `python analysis/section37_heterogeneity_thresholds.py --full-bg 0.05 --full-dlamn 0.05` |
| Full-model thresholds at 0.025 percentage-point spacing | `python analysis/section37_full_threshold_refinement.py` |
| One full-threshold family/background | `python analysis/section37_full_threshold_refinement.py --family common_LLI_dLAMn --background 0.10` |
| Full reference-grid robustness sweep | `python analysis/run_robustness.py` |
| One robustness point; contact resistance uses ohm m² | `python analysis/robust_worker.py contact rcontact 0.001 0` |
| Verify and plot full-model thresholds from source-bound checkpoints | `python analysis/threshold_figures_v28.py` |
| Supplementary LAMp/N/P and integrated maps | `python analysis/supplementary_maps_v28.py` |
| Map computed outputs to SI figure/table numbering | `python analysis/finalize_figures_v28.py` |
| Check final result inventory and numerical claims | `python validation/verify_v28_release.py` |

## September 2026 manuscript checks

The revised manuscript adds a controlled N/P loading comparison, selected cutoff sensitivity, and full-model electrode trajectories (Figure S11). Reproduce their CSVs and Figure S11 from the repository root:

```bash
python analysis/manuscript_revision_checks.py
python analysis/graphical_abstract.py
```

The first command writes `matched_NP_design.csv`, `cutoff_sensitivity.csv`, `full_model_electrode_trajectories.csv`, and Figure S11 under `results/recomputed/supplementary/`. The second command uses the trajectory CSV to generate the revised graphical abstract in the same directory. The three CSVs are committed alongside their generating scripts. These are focused full-model checks; the 0.80 and 0.95 current-sharing levels remain descriptive markers, and the cutoff comparison does not predict plating or temperature.

Threshold summaries use configuration-specific filenames to prevent partial runs overwriting another study. The full threshold search checks **every smaller point on the requested grid**, not just a coarse bracket. Grid spacing is not a confidence interval or a substitute for temporal convergence.

`run_final_v8.py` retains its historical filename for compatibility. Some intermediate names differ from final manuscript numbering; `analysis/finalize_figures_v28.py` and `validation/assemble_v28_figures.py` establish the final mapping. The main-text N/P recipe varies positive loading at fixed negative loading and lithium inventory; the alternative negative-loading path is reported separately. Figure S4 and Table S1 use the full MP-SPMe; the reduced-model 0.1C case reaches an electrode bound before voltage cutoff and has no valid completed-cycle metric.

The full-model thresholds comprise 1,371 sampled points across eight family/background trajectories at 0.025 percentage-point spacing. The 0% common-LLI trajectory does not reach the 0.95 peak marker through 6% extra LAMn. Figure S6 has 22 bound-before-cutoff reduced-model points in the high N/P design, displayed as missing values. The 25 robustness variants are each capacity re-matched at C/20 on the full reference grid. These non-crossings and invalid reduced-model points are retained as such in the CSVs.

The architecture schematic is an author-supplied asset and is not regenerated by code. The supplied current manuscript schematic is preserved under `figures/`.

## Scientific scope

These are isothermal mechanistic models, not a calibrated commercial-cell prediction. No lithium-plating, thermal or aging-rate submodel is included. Graphite BV potential diagnostics report a representative-particle potential, not deposited lithium. Model parameter values and the high-overpotential LFP constitutive choice were retained. Code verification does not establish experimental validity.

## Citation and license

See `CITATION.cff` and `LICENSE`. The original repository is https://github.com/IkerLopetegi/Parallel-LFP.
