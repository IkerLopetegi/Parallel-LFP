"""MP-SPMe BDF integration with rejection of out-of-domain trial steps.

Use the production algebraic/RHS kernel and retain the state exactly. A failed
trial is discarded; BDF restarts from the last accepted state with a smaller
first step. There is no state clipping or change to the constitutive model.
"""
import numpy as np
from scipy.integrate import BDF
from scipy.optimize import brentq
from lfp_parallel import model as m


def simulate_discharge(cells,y0,C_rate=1.,max_step=2.5,rtol=1e-5,atol=1e-7):
    offsets=np.cumsum([0]+[p['Nstate'] for p in cells])
    applied=-C_rate*sum(m.lowrate_capacity(p) for p in cells)
    cutoff=max(p['Vmin'] for p in cells)
    guess=[3.3]

    def in_domain(y):
        for k,p in enumerate(cells):
            state=m.unpack_cell_state(y[offsets[k]:offsets[k+1]],p)
            if (state['xp'].min()<0 or state['xp'].max()>1
                    or state['cn'].min()<0 or state['cn'].max()>p['neg']['cmax']
                    or state['ce'].min()<=0):
                return False
        return True

    def evaluate(t,y):
        result=m.solve_parallel_voltage(t,y,cells,lambda _:applied,guess[0])
        guess[0]=result[0]
        return result

    def rhs(t,y):
        return np.concatenate([e['dy'] for e in evaluate(t,y)[2]])

    voltage,current,_=evaluate(0.,y0)
    if not in_domain(y0) or voltage<=cutoff:
        raise m.NumericalError('Invalid bounded MP-SPMe discharge initialization')
    times,states,voltages,currents=[0.],[y0.copy()],[voltage],[current]
    end_time=1.6*3600./C_rate
    settings=dict(max_step=max_step,rtol=rtol,atol=atol)
    solver=BDF(rhs,0.,y0,end_time,**settings)
    rejected=0
    consecutive=0
    while solver.status=='running':
        previous_t=times[-1]
        previous_y=states[-1]
        trial_step=min(solver.h_abs,solver.max_step)
        bad=False
        try:
            solver.step()
            if solver.status=='failed':
                raise m.NumericalError('Bounded MP-SPMe BDF integration failed')
            bad=not in_domain(solver.y)
            if not bad:
                voltage,current,_=evaluate(solver.t,solver.y)
                if voltage<=cutoff:
                    dense=solver.dense_output()
                    tcut=brentq(lambda t:evaluate(t,dense(t))[0]-cutoff,
                                 previous_t,solver.t,xtol=1e-9)
                    ycut=dense(tcut)
                    bad=not in_domain(ycut)
                    if not bad:
                        vcut,icut,_=evaluate(tcut,ycut)
                        times.append(tcut);states.append(ycut)
                        voltages.append(vcut);currents.append(icut)
                        return dict(t=np.asarray(times),y=np.asarray(states),V=np.asarray(voltages),
                                    I=np.asarray(currents),Iapp=applied,success=True,
                                    reached_cutoff=True,rejected_steps=rejected)
        except m.NumericalError:
            bad=True
        if bad:
            rejected+=1;consecutive+=1
            if consecutive>20 or rejected>1000:
                raise m.NumericalError('Bounded MP-SPMe trial rejection limit exceeded')
            first_step=max(1e-10,.1*trial_step)
            solver=BDF(rhs,previous_t,previous_y,end_time,
                       first_step=min(first_step,end_time-previous_t),**settings)
            continue
        consecutive=0
        times.append(solver.t);states.append(solver.y.copy())
        voltages.append(voltage);currents.append(current)
    raise m.NumericalError('Bounded MP-SPMe discharge did not reach voltage cutoff')
