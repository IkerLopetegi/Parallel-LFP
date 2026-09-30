# Model guide

## Units and signs

| Quantity | Units / convention |
|---|---|
| `I`, `Iapp` | A/m² geometric electrode area; positive means cell charge |
| `j0_ref`, `i_p` | A/m² active-material surface |
| `Qp`, `Qn`, `QLi`, `Q_nominal_ref` | C/m² geometric area |
| `lowrate_capacity()` | Ah/m² geometric area |
| `R_contact`, effective ASR | ohm m² |
| `cn`, `ce`, `cmax` | mol/m³ |
| `xp`, `xn` | dimensionless lithiation fractions |
| lengths, diffusion coefficients | m, m²/s |
| time, potentials | s, V |

Multiply geometric current density or areal capacity by electrode area to obtain A or Ah. `LLI=0.10` removes `0.10*Q_nominal_ref` from `QLi_fresh`; it is **not** 10% of total `QLi_fresh`. LAM fractions reduce active solid while keeping porosity fixed. Lost active solid is implicitly inactive solid. This is a chosen synthetic degradation construction; lithium trapped by LAM is not a separate state.

## State layout

`p['idx']` contains slices into each cell state. `ce` has `Nneg+Nsep+Npos` finite volumes. `xp` reshapes to `(Npos, Npsd)`. `cn` has `Nr_neg` spherical control volumes. Parallel states concatenate cell states.

PSD number weights are uniform quantiles of a lognormal distribution; lithium inventory uses volume weights, and reaction area uses `3*eps_s*w_volume/R`. Setting PSD CV to zero reduces the actual number of bins to one and updates state dimensions consistently.

## Main path

1. `get_reference_params()` constructs independent nested parameter dictionaries.
2. `make_degraded_cell()` or `make_mixed_degraded_cell()` applies validated synthetic degradation fractions.
3. `init_parallel_at_common_ocv()` solves for each branch's zero-net-current state at the requested dynamic terminal voltage.
4. `simulate_cc_halfcycle()` solves branch currents and shared voltage during a BDF integration.
5. `compute_current_metrics()` integrates over actual, nonuniform output times.

Zero net current at initialization does not require zero reaction in every PSD population. Homogeneous initial populations can redistribute lithium internally. This distinction matters for a phase-separating material; the initialization does not represent a fully relaxed multiphase equilibrium.

## Numerical corrections

- Spherical diffusion is conservative. The boundary concentration is reconstructed from the last volume-center value and imposed surface flux over half a radial cell. Both graphite OCP and kinetics use that reconstructed value.
- `neg.surface_method='outer_shell'` remains available as an explicit discretization sensitivity; `extrapolated` is the default. Both returned OCP and reported surface stoichiometry follow the selected method.
- `neg.kinetics='linear'` retains baseline linearized graphite charge transfer; `butler_volmer` selects the diagnostic relation. No analysis monkey-patches model functions.
- Positive BV flux is blocked only when it would drive a population outward at its composition bound. Inward flux remains allowed. The direction is determined from the unmasked trial current at each residual evaluation. A continuous endpoint ramp over the last `x_min` of composition avoids chattering from a discontinuous hard switch; it is unity elsewhere. This narrow numerical regularization is explicit and requires endpoint-width sensitivity checks for new limiting cases.
- Electrolyte interface resistance is the sum of the two half-cell diffusion resistances. This accounts for unequal neighboring widths and preserves salt conservation.
- Branch current is solved with residual-checked Newton/backtracking and Brent fallback. Its differential conductance includes the current dependence of graphite surface concentration and kinetics.
- The shared voltage solve raises if no current-balanced root is found. It never returns the nearest non-root voltage.
- Full and OCV-R simulations require a terminal cutoff; incomplete trajectories are errors. Shared limits use the most restrictive branch cutoff.
- Reduced OCV-R integration uses an adaptive solver and voltage event, with no clipping of accepted electrode states. `dq_frac` sets the maximum time step, not a fixed Euler increment.

## Retained approximations and limitations

The positive populations are internally uniform and use the regular-solution high-overpotential constitutive limit in the full dynamic model. Electrolyte concentration is resolved, but ionic potential is represented by the existing concentration/Ohmic approximation rather than an independently solved porous-electrode charge-conservation field. The graphite electrode has one representative radial particle. These are model assumptions, not corrected discretization defects.

Low-rate capacity matching uses the equilibrium Maxwell OCP plus state-dependent resistance; it is not a full MP-SPMe charge/discharge experiment. The legacy MP0D fixed-step routines are retained for low-overpotential hysteresis illustrations. Their time-step sensitivity and endpoint approximations should be checked before using them for new quantitative studies; the primary dynamic verification targets the MP-SPMe and adaptive OCV-R paths.

Composition and BV argument guards remain numerical protections. Returned `xn_surface_raw` exposes the un-clipped reconstructed graphite surface value for diagnostics. Validate state admissibility and convergence for new extreme parameter sets. Neither passing software tests nor a converged solver demonstrates that the chosen constitutive model is physically accurate for every rate.

## Electrode capacity and N/P

N/P is Qn/Qp using ideal 0–1 intercalation spans for both graphite and LFP. The baseline ratio is 0.883036; full-cell SOC reference spans do not define independent electrode capacity. `balancing_reference` initializes lithium inventory and the fixed nominal capacity scale only. Main designs use positive thickness L_pos = L_pos_ref*(Qn/Qp)_ref/target at fixed negative electrode and lithium inventory. The controlled SI comparison also varies negative thickness L_neg = L_neg_ref*target/(Qn/Qp)_ref. These nominal host capacities do not establish degradation-free windows or experimentally measured reversible capacities.
