# V28 corrected-code results and manuscript release

29 September 2026. The accompanying revised manuscript and Supplementary Information were built from the regenerated figures. The original published CSVs are preserved under `results_published/` for comparison; the calculations described here are under `results/recomputed/`.

## Numerical changes

The representative capacity-matched 1C charge and discharge cases now both reach approximately complete transient handover (`M_peak` approximately 1.000). The original stored peaks were 0.98965 and 0.79783. The 25-variant robustness sweep re-matches LAMn to the 10% LLI target separately for each variant at C/20 and uses the full reference 10/5/10, 13-PSD-bin, 13-graphite-radial grid. Matching errors are below 2.9e-8 Ah m^-2. Of 25 variants, 23 reach `M_peak >= 0.95`; the low graphite diffusivity and low negative exchange-current variants yield 0.563460 and 0.715161. An earlier coarse-grid diagnostic yielded 0.3985 for the low-diffusivity case and is excluded from the manuscript.

Full MP-SPMe thresholds were searched at 0.025 percentage-point spacing. Values below are *additional severity* in percentage points; NR means no crossing anywhere on the sampled range. The 0% common-LLI case has a maximum sampled peak of 0.905 through 6% extra LAMn.

| Family | Background | `M_peak >= 0.80` | `M_peak >= 0.95` |
|---|---:|---:|---:|
| Common LLI + additional LAMn | 0% | 1.725 | NR (0–6%) |
| Common LLI + additional LAMn | 5% | 0.850 | 1.350 |
| Common LLI + additional LAMn | 10% | 0.825 | 1.350 |
| Common LLI + additional LAMn | 15% | 1.075 | 1.375 |
| Same LLI+LAMn trajectory | 0% | 4.225 | 6.500 |
| Same LLI+LAMn trajectory | 5% | 3.475 | 5.425 |
| Same LLI+LAMn trajectory | 10% | 3.350 | 5.000 |
| Same LLI+LAMn trajectory | 15% | 3.125 | 4.350 |

The positive-loading N/P sweep has 39 full-model cases. The separate negative-loading design path has 182 reduced-model cases and nine full-model checks. Figure S6 comprises 242 reduced-model LAMp/N/P map points; 22 of 121 in the higher-utilization design reach an electrode bound before the requested voltage cutoff and are explicitly missing. The full-model moderate C-rate table reaches all four cutoffs; its 0.1C peak is 1.046655, briefly indicating reverse branch current. The separate reduced OCV–R 0.1C diagnostic has no cutoff-reaching metric and is kept under a diagnostic filename.

## Verification and provenance

- All 16 model regression tests pass.
- The 1,371 threshold evaluations were audited for common model/search source SHA, complete sampled grids, and first crossing selection. `results/recomputed/thresholds/reproduction_audit.json` records the audit. The core model and threshold search script were not modified during regeneration.
- A tighter integration of selected threshold points agreed with the stored peaks to about 1e-5; the full-model 0.1C peak changed by 0.000004 under tighter time stepping and tolerances.
- `validation/verify_v28_release.py` verifies figure inventory, all 25 robustness variants and matches, threshold data and sampled crossings, Figure 2 cutoffs, the C-rate table, and the 22 explicitly invalid map cells.
- The manuscript and SI PDFs compile successfully; selected pages containing newly numbered figures and tables were rendered and visually inspected. Figure 1 and the graphical abstract remain author-supplied assets.

Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, and Matplotlib 3.10.8 were used. Full threshold grid spacing is a sampling resolution rather than a continuous-threshold confidence interval. These are isothermal mechanistic predictions; numerical verification does not establish experimental calibration or a lithium-plating prediction.
