"""Figure 2: full MP-SPMe C/2 cycles for isolated 20% degradation.

Run this module to simulate and render; use --plot-only to render retained data.
The current is referenced to the fresh nominal capacity for every cell. Charge
starts from a homogeneous, lithium-conserving zero-net-current state at Vmin;
discharge starts immediately from the charged state, without reinitialization.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import BDF
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfp_parallel import model as m
from analysis.common import OUT, ROOT, prepare_output

RATE = 0.5
SEVERITY = 0.20
SETTINGS = dict(max_step=5.0, rtol=4e-5, atol=3e-7)
TIGHT = dict(max_step=2.5, rtol=1e-5, atol=1e-7)
MODES = ('fresh', 'LLI', 'LAMn', 'LAMp')


def history(sim, p, case, direction):
    # Exact finite-volume shell volumes, rather than an arithmetic radial mean.
    faces = np.linspace(0., 1., p['disc']['Nr_neg'] + 1)
    weights = np.diff(faces**3)
    rows = []
    for t, y, voltage in zip(sim['t'], sim['y'], sim['V']):
        state = m.unpack_cell_state(y, p)
        if (state['xp'].min() < -1e-8 or state['xp'].max() > 1+1e-8
                or state['cn'].min() < 0 or state['cn'].max() > p['neg']['cmax']
                or state['ce'].min() <= 0):
            raise m.NumericalError('Figure 2 accepted state outside composition bounds')
        xn = float(weights @ state['cn'] / p['neg']['cmax'])
        xp = float(np.mean(state['xp'] @ p['lfp']['psd']['w_volume']))
        lithium = p['Qn'] * xn + p['Qp'] * xp
        evaluation = m.cell_at_voltage(y, p, float(voltage))
        rows.append(dict(case=case, direction=direction, t_s=t,
                         capacity_Ahm2=abs(sim['Iapp'])*t/3600.,
                         V=voltage, xn=xn, xp=xp,
                         xn_surface=evaluation['xn_surface'],
                         I_Apm2=sim['Iapp'], Un=evaluation['Un'],
                         eta_n=evaluation['eta_n'], Eneg=evaluation['Eneg'],
                         lithium_error_relative=(lithium-p['QLi'])/p['QLi']))
    data = pd.DataFrame(rows)
    if not np.isfinite(data.select_dtypes('number').to_numpy()).all():
        raise m.NumericalError('Nonfinite Figure 2 trajectory')
    if data.lithium_error_relative.abs().max() > 1e-6:
        raise m.NumericalError('Figure 2 lithium conservation failed')
    if data.xn.min() < 0 or data.xn.max() > 1:
        raise m.NumericalError('Figure 2 bulk graphite composition outside [0,1]')
    return data


def full_halfcycle(p, y0, direction, settings):
    """Integrate the production MP-SPMe kernel, resolving boundary trial steps.

    BDF may propose a state beyond a composition boundary before locating the
    voltage event. Restart from the last accepted state with a smaller step if
    that trial state has no algebraic current solution. No state clipping or
    model/kinetics substitution is introduced here.
    """
    sign = 1 if direction == 'charge' else -1
    current = sign * RATE * p['Q_nominal_ref'] / 3600.
    cutoff = p['Vmax'] if sign > 0 else p['Vmin']
    guess = [3.3]

    def evaluate(t, y):
        result = m.solve_parallel_voltage(t, y, [p], lambda _: current, guess[0])
        guess[0] = result[0]
        return result

    def rhs(t, y):
        return evaluate(t, y)[2][0]['dy']

    voltage = evaluate(0., y0)[0]
    if sign * (voltage-cutoff) >= 0:
        raise m.NumericalError('Initial Figure 2 loaded voltage outside cutoff')
    times, states, voltages = [0.], [y0.copy()], [voltage]
    end_time = 2. * 3600. / RATE
    max_step = settings['max_step']
    retries = 0
    solver = BDF(rhs, 0., y0, end_time, **settings)
    while solver.status == 'running':
        try:
            solver.step()
        except m.NumericalError:
            retries += 1
            if retries > 6:
                raise
            max_step *= .1
            print('endpoint trial-state retry:', direction,
                  't_s', times[-1], 'max_step_s', max_step, flush=True)
            solver = BDF(rhs, times[-1], states[-1], end_time,
                         max_step=max_step, rtol=settings['rtol'], atol=settings['atol'])
            continue
        if solver.status == 'failed':
            raise m.NumericalError('Figure 2 BDF integration failed')
        voltage = evaluate(solver.t, solver.y)[0]
        if sign * (voltage-cutoff) >= 0:
            dense = solver.dense_output()
            tcut = brentq(lambda t: evaluate(t, dense(t))[0]-cutoff,
                          times[-1], solver.t, xtol=1e-9)
            times.append(tcut)
            states.append(dense(tcut))
            voltages.append(evaluate(tcut, states[-1])[0])
            return dict(t=np.asarray(times), y=np.asarray(states),
                        V=np.asarray(voltages), Iapp=current,
                        reached_cutoff=True, endpoint_retries=retries)
        times.append(solver.t)
        states.append(solver.y.copy())
        voltages.append(voltage)
    raise m.NumericalError('Figure 2 half-cycle did not reach cutoff')


def run_case(p, case, settings=SETTINGS):
    y, _ = m.init_cell_at_ocv(p, p['Vmin'], soc_hint=0.)
    frames = []
    for direction in ('charge', 'discharge'):
        sim = full_halfcycle(p, y, direction, settings)
        cutoff = p['Vmax'] if direction == 'charge' else p['Vmin']
        if not sim['reached_cutoff'] or abs(sim['V'][-1]-cutoff) > 1e-7:
            raise m.NumericalError('Figure 2 half-cycle missed voltage cutoff')
        frames.append(history(sim, p, case, direction))
        y = sim['y'][-1].copy()
        print(case, direction, 'duration_s', sim['t'][-1],
              'initial_V', sim['V'][0], 'final_V', sim['V'][-1], flush=True)
    return pd.concat(frames, ignore_index=True)


def render(data):
    colors = {'LLI': '#0072B2', 'LAMn': '#D55E00', 'LAMp': '#E69F00'}
    fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.92), sharey=True,
                            constrained_layout=True)
    for ax, case, label in zip(axs, colors, ('LLI', r'LAM$_n$', r'LAM$_p$')):
        for name, color, caption in [('fresh', '#202020', 'Fresh'),
                                     (case, colors[case], f'20% {label}')]:
            for direction, ls in [('charge', '-'), ('discharge', '--')]:
                curve = data[(data.case == name) & (data.direction == direction)]
                ax.plot(curve.xn, curve.V, ls, color=color, lw=1.25,
                        label=f'{caption}, {direction}')
        ax.set(xlabel=r'Bulk graphite lithiation, $x_n$', xlim=(0, 1),
               ylim=(2.48, 3.69))
        # Keep legends above the data, with identical placement across panels.
        ax.legend(frameon=False, fontsize=5.9, loc='upper left')
    axs[0].set_ylabel('Terminal voltage (V)')
    for ax, label in zip(axs, 'abc'):
        ax.text(.015, 1.025, label, transform=ax.transAxes,
                va='bottom', fontweight='bold', clip_on=False)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'Figure02_degradation_hysteresis.{ext}',
                    bbox_inches='tight', dpi=320)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args(argv)
    prepare_output()
    if args.plot_only:
        render(pd.read_csv(OUT / 'Figure02_fullmodel_trajectories.csv'))
        return
    base = m.get_reference_params()
    frames = []
    for case in MODES:
        p = m.make_degraded_cell(base, case, 0. if case == 'fresh' else SEVERITY)
        frames.append(run_case(p, case))
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUT / 'Figure02_fullmodel_trajectories.csv', index=False)
    # The near-saturation LAMn trajectory receives a stricter full-cycle check.
    tight = run_case(m.make_degraded_cell(base, 'LAMn', SEVERITY), 'LAMn', TIGHT)
    tight.to_csv(OUT / 'Figure02_LAMn_tolerance_trajectories.csv', index=False)
    rows = []
    for (case, direction), curve in data.groupby(['case', 'direction'], sort=False):
        row = dict(case=case, direction=direction, model='MP-SPMe', C_rate=RATE,
                   capacity_Ahm2=curve.capacity_Ahm2.iloc[-1],
                   initial_V=curve.V.iloc[0], endpoint_V=curve.V.iloc[-1],
                   initial_xn=curve.xn.iloc[0], endpoint_xn=curve.xn.iloc[-1],
                   peak_V=curve.V.max(), lithium_error_relative=curve.lithium_error_relative.abs().max())
        if case == 'LAMn':
            refined = tight[tight.direction == direction]
            row['tight_capacity_Ahm2'] = refined.capacity_Ahm2.iloc[-1]
            row['capacity_relative_difference'] = abs(row['capacity_Ahm2']/row['tight_capacity_Ahm2']-1)
            # Compare trajectories at equal passed charge, excluding the longer
            # run's unmatched endpoint. Retain the curve difference explicitly.
            q = np.linspace(0, min(row['capacity_Ahm2'], row['tight_capacity_Ahm2']), 2001)
            row['max_curve_voltage_difference_V'] = np.max(np.abs(
                np.interp(q, curve.capacity_Ahm2, curve.V)-
                np.interp(q, refined.capacity_Ahm2, refined.V)))
            if (row['capacity_relative_difference'] > .001
                    or row['max_curve_voltage_difference_V'] > .002):
                raise m.NumericalError('Figure 2 LAMn trajectory failed tolerance check')
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'Figure02_metrics.csv', index=False)
    protocol = dict(model='MP-SPMe', positive_kinetics='high-overpotential regular-solution relation',
                    graphite_kinetics='linear', C_rate=RATE,
                    current_reference='fresh Q_nominal_ref / 3600',
                    current_magnitude_Apm2=RATE*base['Q_nominal_ref']/3600.,
                    degradation_fraction=SEVERITY, cases=list(MODES),
                    initialization='homogeneous zero-net-current state at Vmin; lowest-SOC root',
                    discharge_initialization='final charge state; no rest or reinitialization',
                    bulk_graphite_average='spherical finite-volume shell weights',
                    grid=base['disc'], solver=SETTINGS, LAMn_tolerance_solver=TIGHT,
                    integrator='BDF; retry invalid trial states from last accepted state with 10x smaller max_step',
                    model_sha256=hashlib.sha256((ROOT/'lfp_parallel/model.py').read_bytes()).hexdigest())
    (OUT/'Figure02_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    render(data)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
