"""Figure 3: capacity-matched C/20 MP-SPMe cycles with graphite Butler-Volmer.

No OCV-R calculation is used for this figure or its capacity matching.
One initial discharge conditions each cell before the plotted charge/discharge
cycle. The applied current is fixed to C/20 of the fresh nominal reference.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import BDF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfp_parallel import model as m
from lfp_parallel.current_controlled import evaluate_at_current
from analysis.common import ROOT, OUT, prepare_output
from analysis.figure02_hysteresis import history

RATE = 0.05
SETTINGS = dict(max_step=30., rtol=1e-7, atol=1e-9)
TIGHT = dict(max_step=15., rtol=2e-8, atol=2e-10)
KINETICS = 'butler_volmer'
MATCH_RTOL = 1e-4
COMPOSITION_FLOOR = 1e-10


def full_halfcycle(p, y0, direction, settings, rate=RATE):
    """Reject invalid BDF trials; locate cutoff before accepting an endpoint."""
    sign=1 if direction=='charge' else -1
    current=sign*rate*p['Q_nominal_ref']/3600.
    cutoff=p['Vmax'] if sign>0 else p['Vmin']
    guess=[3.3]
    def evaluate(t,y):
        e=evaluate_at_current(y,p,current,guess[0])
        guess[0]=e['V']
        return e['V'],np.array([current]),[e]
    shell_weights=np.diff(np.linspace(0,1,p['disc']['Nr_neg']+1)**3)
    def valid(y,e):
        state=m.unpack_cell_state(y,p)
        lithium=p['Qn']*(state['cn']@shell_weights)/p['neg']['cmax']+p['Qp']*np.mean(state['xp']@p['lfp']['psd']['w_volume'])
        return (state['xp'].min()>=-1e-8 and state['xp'].max()<=1+1e-8
                and state['cn'].min()>=0 and state['cn'].max()<=p['neg']['cmax']
                and state['ce'].min()>=p['num']['ce_min']
                and 0<=e['xn_surface_raw']<=1
                and abs(lithium/p['QLi']-1)<=1e-6)
    def rhs(t,y):
        return evaluate(t,y)[2][0]['dy']
    voltage,_,e=evaluate(0.,y0)
    if not valid(y0,e[0]) or sign*(voltage-cutoff)>=0:
        raise m.NumericalError('Invalid Figure 3 initial loaded state')
    times=[0.];states=[y0.copy()];voltages=[voltage]
    end_time=2*3600/rate
    solver=BDF(rhs,0.,y0,end_time,**settings)
    rejected=0;consecutive=0
    while solver.status=='running':
        trial_step=min(solver.h_abs,solver.max_step)
        bad=False
        try:
            solver.step()
            if solver.status=='failed': raise m.NumericalError('Figure 3 BDF failed')
            voltage,_,e=evaluate(solver.t,solver.y)
            if sign*(voltage-cutoff)>=0:
                dense=solver.dense_output()
                tcut=brentq(lambda t:evaluate(t,dense(t))[0]-cutoff,
                            times[-1],solver.t,xtol=1e-12)
                ycut=dense(tcut);vcut,_,ecut=evaluate(tcut,ycut)
                if valid(ycut,ecut[0]):
                    times.append(tcut);states.append(ycut);voltages.append(vcut)
                    return dict(t=np.asarray(times),y=np.asarray(states),
                                V=np.asarray(voltages),Iapp=current,reached_cutoff=True,
                                endpoint_retries=rejected)
                bad=True
            else:
                bad=not valid(solver.y,e[0])
        except m.NumericalError:
            bad=True
        if bad:
            rejected+=1;consecutive+=1
            if consecutive>25 or rejected>1000:
                raise m.NumericalError('Figure 3 trial rejection limit exceeded')
            first_step=max(1e-10,.1*trial_step)
            solver=BDF(rhs,times[-1],states[-1],end_time,
                       first_step=min(first_step,end_time-times[-1]),**settings)
            continue
        consecutive=0
        times.append(solver.t);states.append(solver.y.copy());voltages.append(voltage)
    raise m.NumericalError('Figure 3 did not reach voltage cutoff')


def cycle(p, settings=SETTINGS, retain=False):
    """Condition by discharge, then charge and discharge without state resets."""
    eps=p['num']['x_min']
    lo=max(eps,(p['QLi']-p['Qn']*(1-eps))/p['Qp'])
    hi=min(1-eps,(p['QLi']-p['Qn']*eps)/p['Qp'])
    if lo>=hi: raise ValueError('No admissible lithium-conserving initial state')
    y=m.state_from_xp(p,.5*(lo+hi))
    initial = full_halfcycle(p, y, 'discharge', settings, rate=RATE)
    y = initial['y'][-1].copy()
    frames=[]; capacities={}
    for direction in ('charge','discharge'):
        sim = full_halfcycle(p, y, direction, settings, rate=RATE)
        cutoff = p['Vmax'] if direction=='charge' else p['Vmin']
        if abs(sim['V'][-1]-cutoff)>1e-6:
            raise m.NumericalError('Figure 3 missed cutoff')
        capacities[direction]=abs(sim['Iapp'])*sim['t'][-1]/3600.
        if capacities[direction]>min(p['Qn'],p['Qp'])/3600.*(1+1e-6):
            raise m.NumericalError('Half-cycle exceeds an electrode host capacity')
        if retain:
            frames.append(history(sim,p,p['deg']['mode'],direction,current_evaluator=evaluate_at_current))
        y = sim['y'][-1].copy()
    return capacities, pd.concat(frames,ignore_index=True) if retain else None


def match_cell(base, target, mode, bounds):
    """Match full-model discharge capacity; all objective evaluations simulate."""
    cache={}
    def residual(severity):
        key=float(severity)
        if key not in cache:
            p=m.make_degraded_cell(base,mode,key)
            capacity,_=cycle(p)
            cache[key]=capacity['discharge']
            print('capacity root',mode,'severity',key,'Q',cache[key],flush=True)
        return cache[key]-target
    severity=float(brentq(residual,*bounds,xtol=2e-6,rtol=1e-7))
    error=residual(severity)
    if abs(error)>MATCH_RTOL*target:
        raise m.NumericalError('Full-model capacity matching failed')
    return m.make_degraded_cell(base,mode,severity), severity, error


def check_composition_guard(p, q, data):
    """Repeat the limiting LAMn cell with a tenfold smaller composition guard."""
    import copy
    refined=copy.deepcopy(p)
    refined['num']['graphite_x_min']=COMPOSITION_FLOOR/10
    q_refined,d_refined=cycle(refined,TIGHT,retain=True)
    checks=[]
    for direction in ('charge','discharge'):
        a=data[data.direction==direction]
        b=d_refined[d_refined.direction==direction]
        axis=np.linspace(0,1,2001)
        v_error=float(np.max(abs(np.interp(axis,a.capacity_Ahm2/q[direction],a.V)-
                                 np.interp(axis,b.capacity_Ahm2/q_refined[direction],b.V))))
        q_error=abs(q[direction]/q_refined[direction]-1)
        checks.append(dict(case=p['deg']['mode'],direction=direction,
                           graphite_composition_floor=COMPOSITION_FLOOR,
                           refined_graphite_composition_floor=refined['num']['graphite_x_min'],
                           capacity_relative_difference=q_error,
                           max_voltage_difference_V=v_error))
        if q_error>.001 or v_error>.002:
            raise m.NumericalError('Figure 3 composition guard refinement failed')
    d_refined.to_csv(OUT/'Figure03_LAMN_guard_trajectories.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'Figure03_composition_guard_check.csv',index=False)
    return checks


def render(data, metrics):
    fig, axs=plt.subplots(2,1,figsize=(3.55,5.25),constrained_layout=True)
    colors={'LLI':'#0072B2','LAMN':'#D55E00','LAMP':'#E69F00'}
    for case,color in colors.items():
        row=metrics[metrics.case==case].iloc[0]
        mode={'LLI':'LLI','LAMN':r'LAM$_n$','LAMP':r'LAM$_p$'}[case]
        label=f'{100*row.severity:.2f}% {mode}'
        for ax,direction in zip(axs,('charge','discharge')):
            z=data[(data.case==case)&(data.direction==direction)]
            ax.plot(z.capacity_Ahm2/z.capacity_Ahm2.iloc[-1],z.V,color=color,lw=1.3,
                    label=label)
    for ax,direction,letter in zip(axs,('charged','discharged'),'ab'):
        ax.set(xlabel=f'Normalized {direction} capacity',ylabel='C/20 terminal voltage (V)',
               xlim=(0,1),ylim=(2.47,3.69))
        ax.axhline(2.5,color='0.85',lw=.7)
        ax.axhline(3.65,color='0.85',lw=.7)
        ax.text(.025,.975,letter,transform=ax.transAxes,va='top',fontweight='bold')
    axs[0].legend(frameon=False,fontsize=7.3,loc='lower center',ncol=1)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'Figure03_lowrate_characterization.{ext}',bbox_inches='tight',dpi=320)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--matched-severities',type=float,nargs=2,metavar=('LAMN','LAMP'),
                        help='Reuse candidate severities; recheck full-model capacity matching')
    args=parser.parse_args(argv)
    prepare_output()
    if args.plot_only:
        protocol=json.loads((OUT/'Figure03_protocol.json').read_text())
        if (protocol['model_sha256']!=hashlib.sha256((ROOT/'lfp_parallel/model.py').read_bytes()).hexdigest()
                or protocol['current_controlled_sha256']!=hashlib.sha256((ROOT/'lfp_parallel/current_controlled.py').read_bytes()).hexdigest()
                or protocol['graphite_kinetics']!=KINETICS):
            raise m.NumericalError('Figure 3 saved protocol mismatch')
        render(pd.read_csv(OUT/'Figure03_fullmodel_trajectories.csv'),pd.read_csv(OUT/'Figure03_lowrate.csv'))
        return
    base=m.get_reference_params();base['neg']['kinetics']=KINETICS
    base['num']['graphite_x_min']=COMPOSITION_FLOOR
    target=m.make_degraded_cell(base,'LLI',.10)
    q_target,target_data=cycle(target,retain=True)
    print('Target LLI discharge capacity',q_target['discharge'],flush=True)
    pairs=[(target,.10,0.,q_target,target_data)]
    for index,(mode,bounds) in enumerate([('LAMn',(0.,.25)),('LAMp',(0.,.50))]):
        if args.matched_severities is None:
            p,severity,error=match_cell(base,q_target['discharge'],mode,bounds)
        else:
            severity=args.matched_severities[index]
            if not bounds[0]<=severity<=bounds[1]: raise ValueError('Severity outside matching bounds')
            p=m.make_degraded_cell(base,mode,severity)
        q,data=cycle(p,retain=True)
        error=q['discharge']-q_target['discharge']
        if abs(error)>MATCH_RTOL*q_target['discharge']:
            raise m.NumericalError('Candidate severity failed full-model capacity matching')
        pairs.append((p,severity,error,q,data))
    frames=[];rows=[];checks=[]
    for p,severity,error,q,data in pairs:
        frames.append(data)
        rows.append(dict(case=p['deg']['mode'],severity=severity,model='MP-SPMe',
                         graphite_kinetics=KINETICS,C_rate=RATE,Capacity_Ahm2=q['discharge'],
                         charge_capacity_Ahm2=q['charge'],matching_error_Ahm2=error,
                         negative_host_capacity_Ahm2=p['Qn']/3600.,
                         positive_host_capacity_Ahm2=p['Qp']/3600.,
                         lithium_error_relative=data.lithium_error_relative.abs().max(),
                         min_raw_surface=data.xn_surface_raw.min(),max_raw_surface=data.xn_surface_raw.max()))
        q_tight,data_tight=cycle(p,TIGHT,retain=True)
        for direction in ('charge','discharge'):
            a=data[data.direction==direction];b=data_tight[data_tight.direction==direction]
            axis=np.linspace(0,1,2001)
            v_error=float(np.max(abs(np.interp(axis,a.capacity_Ahm2/q[direction],a.V)-np.interp(axis,b.capacity_Ahm2/q_tight[direction],b.V))))
            q_error=abs(q[direction]/q_tight[direction]-1)
            checks.append(dict(case=p['deg']['mode'],direction=direction,
                               capacity_relative_difference=q_error,max_voltage_difference_V=v_error))
            print('refinement',checks[-1],flush=True)
            if q_error>.001 or v_error>.002:
                raise m.NumericalError('Figure 3 solver refinement failed')
        data_tight.to_csv(OUT/f'Figure03_{p["deg"]["mode"]}_tolerance_trajectories.csv',index=False)
    for p,severity,error,q,case_data in pairs:
        if p['deg']['mode']=='LAMN':
            check_composition_guard(p,q,case_data)
    data=pd.concat(frames,ignore_index=True);metrics=pd.DataFrame(rows)
    data.to_csv(OUT/'Figure03_fullmodel_trajectories.csv',index=False)
    metrics.to_csv(OUT/'Figure03_lowrate.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'Figure03_solver_tolerance.csv',index=False)
    protocol=dict(model='MP-SPMe',graphite_kinetics=KINETICS,C_rate=RATE,
                  current_reference='fresh Q_nominal_ref / 3600',
                  current_magnitude_Apm2=RATE*base['Q_nominal_ref']/3600.,
                  initialization='homogeneous midpoint of lithium-conserving admissible xp interval',
                  conditioning='one C/20 discharge to 2.5 V before plotted cycle',
                  state_continuation='no rest or reset between half-cycles',
                  matching='full-model discharge capacity after conditioning and charge',
                  matching_relative_tolerance=MATCH_RTOL,
                  curve_refinement_comparison='own-halfcycle normalized capacity; capacities checked separately',
                  reused_candidate_severities=args.matched_severities,normalized_axes='each half-cycle own passed charge',
                  kinetic_coupling='decoupled',positive_composition_floor=base['num']['x_min'],
                  graphite_composition_floor=COMPOSITION_FLOOR,
                  refined_composition_floor=COMPOSITION_FLOOR/10,
                  integrator='BDF with rejection of out-of-domain trials; smaller-first-step restart',
                  grid=base['disc'],solver=SETTINGS,tolerance_solver=TIGHT,
                  model_sha256=hashlib.sha256((ROOT/'lfp_parallel/model.py').read_bytes()).hexdigest(),
                  current_controlled_sha256=hashlib.sha256((ROOT/'lfp_parallel/current_controlled.py').read_bytes()).hexdigest())
    (OUT/'Figure03_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    render(data,metrics)
    print(metrics.to_string(index=False),flush=True)
    return float(metrics[metrics.case=='LAMN'].severity.iloc[0])


if __name__=='__main__':main()
