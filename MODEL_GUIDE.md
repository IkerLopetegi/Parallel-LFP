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

Low-rate capacity matching uses the equilibrium Maxwell OCP plus state-dependent resistance; it is not a full MP-SPMe charge/discharge experiment. Figure 2 now uses the full MP-SPMe, including electrolyte transport and graphite diffusion, with the same constitutive laws as the main dynamic studies. The MP0D fixed-step routines remain available as reduced-model utilities and do not generate Figure 2. Their time-step sensitivity and endpoint approximations should be checked before using them for new quantitative studies; the primary dynamic verification targets the MP-SPMe and adaptive OCV-R paths.

Composition and BV argument guards remain numerical protections. Returned `xn_surface_raw` exposes the un-clipped reconstructed graphite surface value for diagnostics. Validate state admissibility and convergence for new extreme parameter sets. Neither passing software tests nor a converged solver demonstrates that the chosen constitutive model is physically accurate for every rate.

## Electrode capacity and N/P

N/P is Qn/Qp using ideal 0–1 intercalation spans for both graphite and LFP. The baseline ratio is 0.883036; full-cell SOC reference spans do not define independent electrode capacity. `balancing_reference` initializes lithium inventory and the fixed nominal capacity scale only. Main designs use positive thickness L_pos = L_pos_ref*(Qn/Qp)_ref/target at fixed negative electrode and lithium inventory. The controlled SI comparison also varies negative thickness L_neg = L_neg_ref*target/(Qn/Qp)_ref. These nominal host capacities do not establish degradation-free windows or experimentally measured reversible capacities.

## Figure 2 protocol and interpretation

`analysis.figure02_hysteresis` simulates single cells with the full production MP-SPMe: the fresh reference and isolated 20% LLI, LAMn and LAMp. The C/2 current is identical across cases and uses the fresh nominal capacity. Charge begins at the homogeneous zero-net-current state at the 2.5 V lower cutoff. Discharge continues immediately from the final charge state; neither electrolyte nor particle states are reset. The upper and lower loaded-voltage cutoffs are 3.65 and 2.5 V. Graphite lithiation is averaged with spherical shell volumes.

The retained CSV contains both half-cycles for all four cases, including the graphite surface lithiation and negative-electrode potential components. The LAMn cycle is repeated with tighter BDF tolerances and a smaller maximum step. Its voltage rebound persists in the full model: the diminishing graphite polarization near the start of discharge can outweigh the thermodynamic voltage decrease. This is a constitutive model prediction, not experimental validation. The fresh and 20% LAMp trajectories are nearly coincident because this reference balance retains positive-electrode capacity reserve.

Suggested manuscript caption:

```latex
\caption{C/2 charge/discharge trajectories obtained with the full MP-SPMe for isolated 20\% (a) LLI, (b) \LAMn{}, and (c) \LAMp{}, compared with the fresh cell. The current is referenced to the fresh nominal capacity. Charge starts from a homogeneous zero-net-current state at 2.5~V, and discharge starts immediately from the final charge state; the terminal-voltage limits are 2.5 and 3.65~V. Solid and dashed lines denote charge and discharge, respectively. The x-axis is volume-averaged graphite lithiation.}
```

In Section 3.1, describe Figure 2 as full MP-SPMe trajectories and remove the phrase "reduced-model illustration". Supplementary Fig. S11 supplies additional electrode trajectories for a capacity-matched parallel charge pair.

## Figure S6 protocol

`analysis.fullmodel_lamp_map` evaluates 242 full MP-SPMe discharge cases: an 11 by 11 common-LLI/additional-LAMp grid for each of the reference host N/P ratio and N/P=1.20. The latter changes positive thickness while retaining negative geometry and lithium inventory. Initialization is at a common zero-net-current terminal voltage of 3.35 V. The applied 1C current uses the sum of the two C/20 OCV-R usable capacities; the dynamic trajectory uses the full MP-SPMe and terminates at 2.50 V. LAM kinetic coupling is decoupled, consistent with the thermodynamic comparisons.

The full model suppresses outward reaction flux near LFP population composition bounds and allows current transfer to the other branch. The reduced OCV-R boundary termination and its clipped LFP endpoint OCP are not used to determine S6 map completion. For the high-N/P panel, `analysis.bounded_fullmodel` rejects trial BDF steps outside an electrode composition domain and restarts from the last accepted state with a smaller first step. This preserves the production RHS and lithium conservation without clipping the state. Failed integrations raise an error rather than produce blank cells. Saved summaries contain the endpoint states, conservation residual, current-sharing metrics and protocol hash; four refined histories and their solver comparisons are retained. The figure can be rendered directly from the final numerical table.
