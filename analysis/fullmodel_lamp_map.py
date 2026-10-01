"""Full MP-SPMe Figure S6: LLI/LAMp discharge maps at two host N/P ratios.

Run --workers N for independent process workers; --plot-only reuses saved data.
The low-rate OCV-R capacity only defines the applied 1C current. Every map
trajectory is integrated with the production MP-SPMe to the voltage cutoff.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfp_parallel import model as m
from analysis.common import OUT, SUP, ROOT, prepare_output
from analysis.np_sensitivity import base_at_np, design_np
from analysis.bounded_fullmodel import simulate_discharge

GRID = np.linspace(0., .25, 11)
SOLVER = {'reference':dict(max_step=5., rtol=4e-5, atol=3e-7),
          'NP_1.20':dict(max_step=2.5, rtol=1e-5, atol=1e-7)}
TIGHT = {'reference':dict(max_step=2.5, rtol=1e-5, atol=1e-7),
         'NP_1.20':dict(max_step=1.25, rtol=2.5e-6, atol=2.5e-8)}
TABLE = OUT/'FigureS06_LAMp_NP_maps.csv'
PROTOCOL = OUT/'FigureS06_protocol.json'
CHECKPOINT = OUT/'FigureS06_fullmodel_checkpoint.csv'
TOLERANCE = SUP/'FigureS06_solver_tolerance.csv'
HISTORY = SUP/'FigureS06_selected_trajectories.csv'


def protocol():
    base = m.get_reference_params()
    return dict(model='MP-SPMe', reference_NP=design_np(base), high_NP=1.2,
                capacity_convention='full_0_1_intercalation_host_capacity',
                loading_path='positive_thickness; fixed negative electrode and lithium inventory',
                grid=base['disc'], common_LLI=GRID.tolist(), additional_LAMp=GRID.tolist(),
                degradation_kinetic_coupling='decoupled', C_rate=1.,
                current_reference='sum of branch C/20 OCV-R usable capacities',
                direction='discharge', initial_OCV_V=3.35, SOC_root_hint=.75,
                cutoff_V=2.5, solver=SOLVER, tolerance_solver=TIGHT,
                integration={'reference':'production solve_ivp BDF',
                             'NP_1.20':'BDF with out-of-domain trial rejection; smaller-first-step restart'},
                positive_kinetics='production high-overpotential regular-solution relation',
                graphite_kinetics='production linear relation',
                model_sha256=hashlib.sha256((ROOT/'lfp_parallel/model.py').read_bytes()).hexdigest())


def protocol_hash():
    return hashlib.sha256(json.dumps(protocol(),sort_keys=True).encode()).hexdigest()


def simulate_case(job):
    name, lli, lamp, tight, retain = job
    base = m.get_reference_params()
    design = base if name=='reference' else base_at_np(base,1.2)
    cells = [m.make_mixed_degraded_cell(design,lli=lli,lamp=loss,
                                      kinetic_coupling='decoupled') for loss in (0.,lamp)]
    caps = [m.lowrate_capacity(p) for p in cells]
    y,_ = m.init_parallel_at_common_ocv(cells,3.35,.75)
    settings = (TIGHT if tight else SOLVER)[name]
    sim = (m.simulate_cc_halfcycle(cells,y,1.,'discharge',**settings) if name=='reference'
           else simulate_discharge(cells,y,1.,**settings))
    if not sim['reached_cutoff'] or abs(sim['V'][-1]-2.5)>1e-7:
        raise m.NumericalError('S6 full-model discharge missed cutoff')
    if not np.allclose(sim['I'].sum(axis=1),sim['Iapp'],rtol=1e-8,atol=1e-7):
        raise m.NumericalError('S6 branch currents do not sum to applied current')
    metrics = m.compute_current_metrics(sim['t'],sim['I'],sim['Iapp'])
    averages = []
    errors = []
    offsets = np.cumsum([0]+[p['Nstate'] for p in cells])
    for k,p in enumerate(cells):
        states = sim['y'][:,offsets[k]:offsets[k+1]]
        xp = states[:,p['idx']['xp']].reshape(-1,p['disc']['Npos'],p['disc']['Npsd'])
        cn = states[:,p['idx']['cn']]
        ce = states[:,p['idx']['ce']]
        xpbar = np.mean(xp @ p['lfp']['psd']['w_volume'],axis=1)
        shell_weights = np.diff(np.linspace(0.,1.,p['disc']['Nr_neg']+1)**3)
        xnbar = cn @ shell_weights / p['neg']['cmax']
        error = np.max(abs(p['Qp']*xpbar+p['Qn']*xnbar-p['QLi']))/p['QLi']
        if (error>1e-6 or xp.min() < -1e-7 or xp.max()>1+1e-7
                or cn.min()<0 or cn.max()>p['neg']['cmax'] or ce.min()<=0):
            raise m.NumericalError(f'S6 {name} LLI={lli} LAMp={lamp} cell={k+1}: lithium_error={error:.3g}, xp=[{xp.min():.12g},{xp.max():.12g}], cn=[{cn.min():.12g},{cn.max():.12g}] cmax={p["neg"]["cmax"]:.12g}, ce_min={ce.min():.12g}')
        averages.append((xpbar,xnbar));errors.append(error)
    peak_index = int(np.argmax(abs(sim['I'][:,0]-sim['I'][:,1])))
    row = dict(design=name,design_NP=design_np(design),LLI=lli,dLAMp=lamp,
               model='MP-SPMe',rtol=settings['rtol'],atol=settings['atol'],max_step_s=settings['max_step'],
               M_peak=metrics['M_peak'],M_rms=metrics['M_rms'],
               qex_norm=metrics['Q_excess_Ahm2']/(.5*sum(caps)),
               peak_recipient=int(np.argmax(abs(sim['I'][peak_index])))+1,
               status='voltage_cutoff',endpoint_V=sim['V'][-1],duration_s=sim['t'][-1],
               applied_current_Apm2=sim['Iapp'],cap1_Ahm2=caps[0],cap2_Ahm2=caps[1],
               endpoint_I1_Apm2=sim['I'][-1,0],endpoint_I2_Apm2=sim['I'][-1,1],
               endpoint_xp1=averages[0][0][-1],endpoint_xp2=averages[1][0][-1],
               endpoint_xn1=averages[0][1][-1],endpoint_xn2=averages[1][1][-1],
               lithium_error_relative=max(errors),rejected_steps=sim.get('rejected_steps',0),
               protocol_sha256=protocol_hash())
    history = None
    if retain:
        history = pd.DataFrame(dict(design=name,LLI=lli,dLAMp=lamp,t_s=sim['t'],V=sim['V'],
                                    I1=sim['I'][:,0],I2=sim['I'][:,1],Iapp=sim['Iapp'],
                                    xp1=averages[0][0],xp2=averages[1][0],
                                    xn1=averages[0][1],xn2=averages[1][1]))
    return row, history


def render(data):
    prepare_output()
    fig,axs=plt.subplots(1,2,figsize=(7.,3.),constrained_layout=True)
    grid=np.round(GRID,6)
    data=data.copy()
    data['LLI']=data.LLI.round(6);data['dLAMp']=data.dLAMp.round(6)
    maximum=max(1.05,float(data.M_peak.max()))
    for ax,name in zip(axs,('reference','NP_1.20')):
        z=data[data.design==name].pivot(index='LLI',columns='dLAMp',values='M_peak')
        # Grid points are at 0,2.5,...25%; cell edges extend by half a spacing.
        image=ax.imshow(z.loc[grid,grid].to_numpy(),origin='lower',extent=[-1.25,26.25,-1.25,26.25],
                        vmin=0,vmax=maximum,cmap='viridis',aspect='auto',interpolation='nearest')
        ax.set(xlabel='Additional LAM$_p$ (%)',ylabel='Common LLI (%)',
               xticks=np.arange(0,26,5),yticks=np.arange(0,26,5))
    fig.colorbar(image,ax=axs,label=r'$M_{\rm peak}$')
    for ax,label in zip(axs,'ab'):
        ax.text(.02,1.025,label,transform=ax.transAxes,va='bottom',fontweight='bold',clip_on=False)
    for ext in ('pdf','png'):
        fig.savefig(SUP/f'FigureS06_LAMp_NP_maps.{ext}',bbox_inches='tight',dpi=320)
    plt.close(fig)


def run_map(workers=1,plot_only=False):
    prepare_output()
    if plot_only:
        data=pd.read_csv(TABLE)
        if set(data.model)!={'MP-SPMe'} or set(data.protocol_sha256)!={protocol_hash()}:
            raise RuntimeError('S6 saved data uses a different model or protocol')
        render(data);return
    phash=protocol_hash()
    rows=[]
    for path in (TABLE,CHECKPOINT):
        if path.exists():
            old=pd.read_csv(path)
            if 'protocol_sha256' in old and set(old.protocol_sha256)=={phash}:
                rows=old.to_dict('records')
    PROTOCOL.write_text(json.dumps(protocol(),indent=2)+'\n')
    def key(row):return row['design'],round(row['LLI'],6),round(row['dLAMp'],6)
    completed={key(row) for row in rows}
    jobs=[(name,float(lli),float(lamp),False,False) for name in ('reference','NP_1.20')
          for lli in GRID for lamp in GRID if (name,round(lli,6),round(lamp,6)) not in completed]
    failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(simulate_case,job):job for job in jobs}
        for future in as_completed(futures):
            try:
                row,_=future.result()
            except Exception as exc:
                failures.append((futures[future],str(exc)))
                print('CASE FAILED',futures[future],str(exc),flush=True)
                continue
            rows.append(row)
            pd.DataFrame(rows).sort_values(['design','LLI','dLAMp']).to_csv(CHECKPOINT,index=False)
            print(f'{len(rows)}/242',row['design'],row['LLI'],row['dLAMp'],
                  'M_peak',row['M_peak'],'qex',row['qex_norm'],flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} S6 full-model cases failed: {failures}')
    data=pd.DataFrame(rows).sort_values(['design','LLI','dLAMp'])
    if len(data)!=242 or data.duplicated(['design','LLI','dLAMp']).any():
        raise RuntimeError('Incomplete S6 map')
    data.to_csv(TABLE,index=False)
    if CHECKPOINT.exists():CHECKPOINT.unlink()
    selected=[('NP_1.20',0.,.25),('NP_1.20',0.,.05),('NP_1.20',.25,.25),('reference',.25,.25)]
    checks=[];histories=[]
    for name,lli,lamp in selected:
        row,history=simulate_case((name,lli,lamp,True,True))
        base=data[(data.design==name)&np.isclose(data.LLI,lli)&np.isclose(data.dLAMp,lamp)].iloc[0]
        peak_difference=abs(row['M_peak']-base.M_peak)
        q_difference=abs(row['qex_norm']-base.qex_norm)
        checks.append(dict(design=name,LLI=lli,dLAMp=lamp,M_peak_reference=base.M_peak,
                           M_peak_tight=row['M_peak'],M_peak_difference=peak_difference,
                           qex_reference=base.qex_norm,qex_tight=row['qex_norm'],
                           qex_absolute_difference=q_difference,
                           qex_relative_difference=q_difference/max(abs(base.qex_norm),1e-5),
                           duration_s_reference=base.duration_s,duration_s_tight=row['duration_s']))
        histories.append(history)
        if peak_difference>.01 or q_difference>.002:
            raise m.NumericalError('S6 solver refinement failed')
        print('tolerance check',checks[-1],flush=True)
    pd.DataFrame(checks).to_csv(TOLERANCE,index=False)
    pd.concat(histories,ignore_index=True).to_csv(HISTORY,index=False)
    render(data)
    print(data.groupby('design')[['M_peak','M_rms','qex_norm']].agg(['min','max']).to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    run_map(args.workers,args.plot_only)


if __name__=='__main__':main()
