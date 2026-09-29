# Corrections and their scientific implications

This document records the original source corrections relative to GitHub commit `6061e2078e11c3307166b456bd5ffcb9d3d0782a`. The subsequent V28 regeneration and manuscript changes are documented in `FINAL_RELEASE_REPORT.md`.

## Numerical changes

1. **Composition-limit flux control.** Both the core and diagnostic reused already-zeroed reaction currents when recomputing blocking, which could alternate the mask. A shared bounded-BV implementation now determines direction from the unmasked current. Outward current vanishes at the bound; inward current is allowed. A continuous ramp over the last `x_min` of composition avoids a discontinuous ODE at the boundary. Its width sensitivity was checked in a representative case.
2. **Current conservation.** Branch Newton iteration is residual-checked and has a bracketed fallback. The shared-voltage solver no longer returns the closest grid point when no root exists. Algebraic failures raise `NumericalError` rather than producing a plausible-looking trajectory.
3. **Graphite surface consistency.** The old model used the final shell-center concentration for OCP and kinetics but returned a reconstructed boundary concentration as `xn_surface`. The corrected default uses the same reconstructed surface value throughout and includes its current dependence in the differential conductance.
4. **Diffusion interfaces.** Electrolyte fluxes use half-cell-width-weighted harmonic averaging. Spherical diffusivity denominators no longer use dimensionless machine epsilon as a floor for dimensional diffusion coefficients, which distorted values below approximately 2.2e-16 m²/s.
5. **Initialization.** The old routine selected a nearby equilibrium-Maxwell grid point, without solving the dynamic model's zero-current equation. The new routine solves that equation at the specified common terminal voltage. This changes initial electrode states and therefore durations. Zero net current still permits internal exchange among PSD bins.
6. **Integration contracts.** Full simulations require successful integration and a voltage-cutoff event. The most restrictive branch voltage limit governs parallel operation. Reduced OCV-R trajectories now use adaptive integration and exact event location instead of fixed Euler stepping and silent state clipping.
7. **Parameters and edge cases.** Invalid severities, unknown coupling/kinetic/diffusivity modes and invalid metric input are rejected. Zero PSD width updates the true bin count. The one-dimensional PSD OCP path no longer attempts assignment into a NumPy scalar. Capacity matching rejects a residual mismatch rather than reporting a minimizer as an exact match. An incomplete parameter cache was removed.
8. **Metrics.** Time-weighted trapezoidal integration is retained and validated. The deprecated NumPy call is replaced with SciPy's supported trapezoidal integrator.

## Workflow and readability changes

- Shared graphite kinetics options replace global monkey-patching; import no longer starts simulations or changes global model behavior.
- The robustness entry points no longer depend on missing `analysis.run_all`.
- The diffusivity script actually calculates its C-rate table rather than writing embedded historical numbers.
- Full threshold search evaluates every smaller point on a uniform grid (default 0.025 percentage points). Checkpoints are tied to code/settings, outputs identify their configuration, and missing crossings are explicit. This supports the reported finer grid but does not claim the corrected thresholds equal V28.
- The main-text positive-loading N/P construction is supplied separately from the archive's alternative negative-loading path. The preserved nominal LLI reference and exact design ratio are documented.
- Source is directly exposed, formatted, and accompanied by a units/state guide, runnable example, package metadata, tested dependency versions and regression tests. A selected-study CLI avoids starting the whole collection unintentionally.
- Plotting uses Matplotlib's internal math rendering; no external TeX is required. Broken Python/LaTeX string escapes were repaired.
- Generated results use one output tree. Historical CSVs are preserved separately; they are not relabeled as verified outputs.
- Architecture redrawing was removed from the numerical workflow; the supplied author schematic is preserved.

## Follow-up completed for V28

The corrections change numerical predictions. The final thresholds, parameter/design sweeps and associated figures were regenerated and reconciled with the revised paper. `FINAL_RELEASE_REPORT.md` records the complete V28 result matrix and remaining scientific limitations. `VALIDATION.md` preserves the earlier focused solver comparison.

No electrochemical fit, experimental validation or replacement of the chosen LFP constitutive law was performed. The retained low-rate MP0D illustration routines are approximate fixed-step methods; they should not be interpreted as independent validation of the full model.
