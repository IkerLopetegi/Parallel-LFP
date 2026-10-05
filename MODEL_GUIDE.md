# Model guide

## Units and signs

| Quantity | Units / convention |
|---|---|
| `I`, `Iapp` | A/m² geometric electrode area; positive means cell charge |
| `j0_ref`, positive reaction current | A/m² active-material surface |
| LFP molar flux | Reaction current divided by Faraday constant, mol/m²/s |
| `Qp`, `Qn`, `QLi`, `Q_nominal_ref` | C/m² geometric area |
| `lowrate_capacity()` | Ah/m² geometric area |
| `R_contact`, effective ASR | Ω m² |
| `cn`, `ce`, `cmax` | mol/m³ |
| `xp`, `xn` | Dimensionless lithiation fractions |
| Lengths, diffusion coefficients | m, m²/s |
| Time, potentials | s, V |

Multiply geometric current density or areal capacity by electrode area to obtain A or Ah.

## Model hierarchy and kinetics

The MP-SPMe contains one representative graphite particle with radial solid diffusion, one-dimensional electrolyte concentration transport, and a distribution of internally uniform LFP particles at each positive-electrode location. The single-radius model uses the same transport equations with one LFP radius. The OCV–R model evolves electrode lithium balance using equilibrium full-cell OCV and a state-dependent lumped resistance; it resolves no concentration gradients.

Positive-electrode reactions use symmetric Butler–Volmer kinetics. `neg.kinetics='linear'` selects linearized graphite Butler–Volmer kinetics, the default for parallel-current studies. `neg.kinetics='butler_volmer'` selects the full relation used by Figures 2 and 3 and the Figure S10 potential diagnostic. Charge-transfer resistance is the small-overpotential kinetic slope expressed per geometric electrode area. It is distinct from added contact resistance and from the electrolyte Ohmic contribution.

The OCV–R resistance contains linearized negative- and positive-electrode charge transfer, Bruggeman-corrected electrolyte Ohmic resistance, and imposed contact resistance. Its exchange-current densities depend on electrode composition. `lowrate_ocvr_characterization()` uses this relation at C/20 to determine cutoff-limited capacity.

## Thermodynamics and capacity

The MP-SPMe uses the homogeneous regular-solution LFP relation corresponding to the Zelič–Katrašnik high-overpotential limit. Every particle-size bin has its own interaction parameter. Equilibrium balancing and OCV–R use a Maxwell coexistence plateau computed with the volume-weighted interaction parameter. The direction-dependent low-overpotential curves are shown only as analytical illustrations in Figure S2.

PSD number weights are uniform quantiles of a lognormal distribution. Lithium inventory uses volume weights; reaction area uses `3*eps_s*w_volume/R`. Zero PSD variation gives one radius and consistent state dimensions.

N/P is `Qn/Qp` using ideal 0–1 host intercalation spans for graphite and LFP. The reference ratio is 0.883036. `balancing_reference` initializes cyclable-lithium inventory and the fixed nominal capacity scale; those coordinates do not impose cycle endpoints. Operating windows depend on lithium inventory, thermodynamic functions and voltage limits. These host capacities do not establish degradation-free or experimentally measured reversible windows.

The main N/P sweep changes positive thickness at fixed negative electrode and lithium inventory. The controlled SI comparison also considers changing negative thickness at fixed positive electrode and inventory.

`LLI=0.10` removes `0.10*Q_nominal_ref` from `QLi_fresh`. LAM reduces active-material volume while keeping porosity fixed; the inactive volume replaces the lost active solid. In the kinetically decoupled case, the reference exchange-current coefficient is rescaled to preserve its product with reaction area. In the coupled case this compensation is absent. Lithium trapped by LAM is not a separate state.

## State layout and numerical solution

`p['idx']` provides the state slices. Electrolyte concentration has `Nneg+Nsep+Npos` finite volumes. LFP stoichiometry reshapes to `(Npos, Npsd)`; graphite has `Nr_neg` spherical control volumes. Parallel states concatenate cell states.

Solid and electrolyte diffusion use conservative finite volumes. The graphite surface concentration is reconstructed from the outer volume center and imposed flux. Electrolyte interface flux uses the two adjacent half-volume diffusion resistances. Positive reaction flux is suppressed only when it would drive a population outward at a composition bound; a narrow endpoint ramp regularizes this condition.

`cell_at_voltage()` solves branch current with residual-checked Newton iteration and Brent fallback. `solve_parallel_voltage()` imposes the total applied current at the common terminal voltage. After eliminating these algebraic variables, `simulate_cc_halfcycle()` integrates states with BDF and a terminal-voltage event. OCV–R also uses adaptive integration and voltage termination. Incomplete trajectories raise errors.

For prescribed-current single-cell calculations, `current_controlled.evaluate_at_current()` evaluates graphite polarization at the known current and solves the positive reaction balance for voltage. It uses the same constitutive and transport functions. Tests compare its voltage and complete state derivative with `cell_at_voltage()` and check lithium and salt conservation.

## Figure protocols

**Figure 2:** four single-cell C/2 charge/discharge cycles, with identical fresh-reference current, full graphite Butler–Volmer kinetics, homogeneous zero-net-current initialization at 2.50 V, and immediate discharge from the final charge state. Cutoffs are 3.65 and 2.50 V. The LAMn case has a tighter-solver repeat.

**Figure 3:** full graphite Butler–Volmer kinetics and full-model C/20 discharge-capacity matching. Each cell starts from the homogeneous midpoint of its lithium-conserving admissible composition interval, undergoes a conditioning discharge, and then charges and discharges without rest or state reset. Each plotted half-cycle uses its own normalized passed charge. The graphite composition guard is 1e-10 and the LFP guard is 2e-6. Solver and graphite-guard refinements are retained.

**Figures 4–9 and parallel-current SI studies:** C/20 OCV–R capacity characterization with linearized graphite kinetics in the electrochemical simulations. Figure S10 uses full graphite Butler–Volmer and Ecker concentration-dependent graphite diffusivity. Its negative-electrode potential is `Un + eta_n` versus Li/Li+.

**Figure S6:** 242 full MP-SPMe discharge cases across two host N/P designs. The 1C current uses the sum of branch C/20 OCV–R capacities. Initialization is at a common zero-net-current voltage of 3.35 V; termination is at 2.50 V. The high-N/P integrator rejects out-of-domain trial states and restarts with a smaller first step. Every retained case reaches its cutoff; selected cases have solver-refinement histories.

All solver settings and numerical provenance are recorded in the figure protocols or generating scripts. Checkpoint hashes prevent resuming with incompatible numerical configurations.

## Physical scope

The model is isothermal, uses internally uniform LFP populations and one representative graphite particle, and approximates ionic potential from concentration and Ohmic terms.
