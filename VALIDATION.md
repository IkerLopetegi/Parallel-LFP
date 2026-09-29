# Validation report

## Scope

Sixteen focused regression tests passed with the recorded environment. Representative full-model and reduced-model workflows were executed; this is not an exhaustive rerun of all manuscript sweeps.

## Representative full-model comparison

Same synthetic degradation families and 1C charge, using the default linear graphite kinetics. Original runs use max_step=12 s, rtol=1.5e-4, atol=1.5e-6. Corrected values below use max_step=4 s, rtol=1e-5, atol=1e-7. Both implementations were run; differences include the corrected zero-current initialization, not just solver precision.

| Pair | Original M_peak | Corrected M_peak | Original duration (min) | Corrected duration (min) |
|---|---:|---:|---:|---:|
| common5_dLAMn5 | 0.979238 | 1.000000 | 25.4875 | 19.9470 |
| trajectory5_vs10 | 0.951807 | 0.924280 | 26.5437 | 21.5178 |

## Conservation and convergence

Across three corrected representative full runs, maximum relative lithium drift was 1.76e-10; maximum relative electrolyte-salt drift was 7.28e-10. Maximum absolute parallel current residual was 2.12e-07 A/m². All reached 3.65 V. Reconstructed graphite surface fractions remained below one in these runs.

Tightening from 12 s / 1.5e-4 / 1.5e-6 to 4 s / 1e-5 / 1e-7 changed M_peak by:
- common5_dLAMn5: 1.11022e-15; excess charge changed by 0.051%.
- trajectory5_vs10: 0.000520101; excess charge changed by 0.165%.

Halving x_min from 2e-6 to 1e-6 in the common-5%-LLI/additional-5%-LAMn case changed excess charge by 0.0011%; M_peak remained effectively 1. This checks one endpoint-limited case, not every sensitivity point.

## Negative-electrode diagnostic

All three diagnostic runs used full symmetric graphite BV, Ecker diffusivity, max_step=8 s, rtol=1e-5 and atol=1e-7. These are actual regenerated outputs, with plots and histories under results/recomputed/supplementary/.

| Case | Rate | Cell | Published minimum (mV) | Corrected minimum (mV) |
|---|---:|---:|---:|---:|
| common5_dLAMn5 | 0.5C | 1 | 50.46 | 49.45 |
| common5_dLAMn5 | 0.5C | 2 | 13.60 | 11.23 |
| common5_dLAMn5 | 1.0C | 1 | 21.13 | 21.39 |
| common5_dLAMn5 | 1.0C | 2 | -26.27 | -32.93 |
| trajectory5_vs10 | 1.0C | 1 | -21.60 | -26.80 |
| trajectory5_vs10 | 1.0C | 2 | 16.50 | 15.56 |

The published column is read from the supplied historical CSV; the old BV script was not independently rerun for this table. Negative potential is a diagnostic, not a prediction of plated-lithium amount.

## Executed workflows

- Low-rate capacity matching and plotting; matched LAMn ≈22.9425%, LAMp ≈32.1216% for 10% LLI.
- Main capacity-matched full charge and discharge dynamics, including CSV and plot generation.
- Reduced OCV-R / single-radius / multiparticle fidelity comparison.
- All three graphite BV potential diagnostics and figure rendering.
- Legacy low-rate hysteresis illustration and figure rendering.
- Main positive-loading N/P=1.20 with 20% LAMp: full discharge smoke test reached 2.50 V.
- Reduced-model example and identical-cell charge-balance test.
- Regression tests for symmetry, cell swapping, lithium/salt conservation, unequal-width electrolyte interfaces, graphite surface consistency, conductance, fallback, parameter rejection, incomplete-run rejection, nonmonotone sampled-threshold search and safe imports.

## Not rerun in full

The full 0.025-percentage-point threshold grids, complete robustness grids, all N/P ratios, complete diffusivity matrix, and all 0–25% degradation maps were not regenerated. Their code is included, but results_published/ must not be treated as corrected output. Legacy figure numbering remains in some filenames; README maps studies to entry points.

## Reproduce

Run `python -m unittest discover -s tests -v` for the focused tests, `python validation/run_validation.py` for the three full representative runs, and the study commands in README for the larger workflows. Validation JSON files contain the measured values.
