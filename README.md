# Parallel LFP/graphite cells

Code and numerical data for *Degradation-Induced Electrode Misalignment and Current Redistribution in Parallel LFP/Graphite Cells*.

## Install

Run from the repository root. The tested environment is Python 3.12.14; the package supports Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
```

For Python 3.10 or 3.11, use `requirements.txt` instead of the tested pins.

On Windows, activate with `.venv\Scripts\activate`. No LaTeX installation is required to generate the figures.

## Reproduce the figures

```bash
python -m analysis.verify_results --manifest
python -m analysis.reproduce --stage saved
```

The first command checks the supplied files, protocols and numerical tables. The second renders every numerical manuscript figure (2–9) and every SI figure (S1–S11) from supplied histories, tables and analytical expressions. It runs no time-dependent simulations. Figure 1 is an author-supplied schematic at `figures/Figure01_model_architecture.pdf`.

PDFs are written to `results/` and `results/supplementary/`; PNG previews are generated alongside them. PDF metadata may differ between runs, so check the supplied manifest before regeneration.

## Recompute the simulations

```bash
python -m unittest discover -s tests -v
python -m analysis.reproduce --stage all --fresh-output ../parallel-lfp-recomputed
```

`--fresh-output` must name a directory that does not exist. It copies the simulation code and schematic there and recomputes results without reusing supplied checkpoints. Full recomputation includes the dense threshold searches and 242 full-model Figure S6 cases and can take substantial time. Without `--fresh-output`, compatible robustness, threshold and S6 checkpoints are reused.

For an individual study, use the commands below. Finish a recomputation with `python -m analysis.reproduce --stage saved` to apply the common publication layout. Run `python -m analysis.reproduce --help` for all stage names.

| Figure | Simulation command or analytical source |
|---|---|
| 1 | Supplied schematic; no simulation |
| 2 | `python -m analysis.figure02_hysteresis` |
| 3 | `python -m analysis.figure03_fullmodel` |
| 4–5 | `python -m analysis.reproduce --stage dynamics` |
| 6 | `python -m analysis.reproduce --stage robustness` |
| 7 | `python -m analysis.reproduce --stage np` |
| 8 and S7 | `python -m analysis.reproduce --stage thresholds` |
| 9 | `python -m analysis.reproduce --stage resistance` |
| S1, S3, S4, S8, S9 | `python -m analysis.reproduce --stage supplementary` |
| S2 | Analytical regular-solution expressions in `analysis/publication_figures.py` |
| S5 | `python -m analysis.reproduce --stage diffusivity` |
| S6 | `python -m analysis.fullmodel_lamp_map --workers 6` |
| S10 | `python -m analysis.reproduce --stage potential` |
| S11 and loading-path/cutoff tables | `python -m analysis.reproduce --stage checks` |

Figures 2, 3, S5 and S6 also accept `--plot-only` in their individual modules. To recompute the supplied Figure 3 cells at their specified severities, use:

```bash
python -m analysis.figure03_fullmodel --matched-severities 0.22775088039999603 0.3234038789057713
```

This checks the full-model capacity match and repeats the numerical refinement checks. Omitting `--matched-severities` performs the severity root searches.

## Repository structure

- `lfp_parallel/`: constitutive equations, transport discretization and current/voltage solvers.
- `analysis/`: figure-specific simulations, plotting and result verification.
- `tests/`: algebraic consistency, conservation, kinetics and numerical checks.
- `figures/`: the model architecture schematic.
- `results/`: manuscript PDFs, numerical tables, plotting histories and protocol JSON files.
- `results/supplementary/`: SI figures, histories and sensitivity tables.
- `results/thresholds/`: eight resumable threshold configurations, sampled points and combined results. Hashed filenames identify configurations, not paper versions.

Some CSV table identifiers differ from the displayed SI table numbers; the figure map above identifies their generating studies. `MODEL_GUIDE.md` explains units, model assumptions, kinetics and capacity conventions. This is an isothermal mechanistic simulation study; the negative-electrode potential diagnostic does not simulate lithium deposition.
