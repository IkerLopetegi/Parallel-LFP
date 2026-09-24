from pathlib import Path
import sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfp_parallel.model as m

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / 'results_v13' / 'supplementary'
SUP.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':'serif','mathtext.fontset':'cm','font.size':8.5,'text.usetex':True,
    'axes.labelsize':9,'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,
    'lines.linewidth':1.45,'axes.spines.top':False,'axes.spines.right':False,
    'figure.dpi':170,'savefig.dpi':320,'pdf.fonttype':42,'ps.fonttype':42})

# Safety-oriented diagnostic variant: use the full symmetric Butler-Volmer
# relation for graphite rather than the linearized R_ct,n term in the production
# MP-SPMe. Positive-electrode and electrolyte equations are unchanged.
def cell_at_voltage_bvneg(y,p,Vcell):
    s=m.unpack_cell_state(y,p); RT=p['R']*p['T']
    ce_n=max(float(np.mean(s['ce_neg'])),p['num']['ce_min'])
    xn_s=float(np.clip(s['cn'][-1]/p['neg']['cmax'],p['num']['x_min'],1-p['num']['x_min']))
    Un=float(m.graphite_ocp_verbrugge(xn_s))
    j0n=p['neg']['j0_ref']*2*math.sqrt(max(p['num']['x_min'],xn_s*(1-xn_s)))*math.sqrt(ce_n/p['elec']['c_ref'])
    Kneg=max(j0n*p['neg']['a_s']*p['geom']['L_neg'],1e-16)
    Rctn0=RT/(p['F']*Kneg)
    kn=p['elec']['kappa']*p['neg']['eps_e']**p['elec']['brugg']
    ks=p['elec']['kappa']*p['sep']['eps_e']**p['elec']['brugg']
    kp=p['elec']['kappa']*p['pos']['eps_e']**p['elec']['brugg']
    Rbase=p['geom']['L_neg']/(2*kn)+p['geom']['L_sep']/ks
    Rohm=Rbase+p['z_pos']/kp
    Rother=Rohm+p['R_contact']
    ce_p=np.maximum(s['ce_pos'],p['num']['ce_min'])
    dPhi=2*RT/p['F']*(1-p['elec']['t_plus'])*np.log(ce_p/ce_n)
    Npos,Npsd=p['disc']['Npos'],p['disc']['Npsd']
    ce_mat=np.repeat(ce_p[:,None],Npsd,axis=1)
    Rmat=np.repeat(Rother[:,None],Npsd,axis=1)
    dpm=np.repeat(dPhi[:,None],Npsd,axis=1)
    Aweight=p['dx_pos']*np.repeat(p['pos']['a_s_bins'][None,:],Npos,axis=0)
    x=np.clip(s['xp'],p['num']['x_min'],1-p['num']['x_min'])
    j0p=p['pos']['j0_ref']*2*np.sqrt(np.maximum(p['num']['x_min'],x*(1-x)))*np.sqrt(ce_mat/p['elec']['c_ref'])
    blocked=np.zeros_like(x,dtype=bool); I=0.0

    def eta_n(Ig):
        return -(2*RT/p['F'])*np.arcsinh(Ig/(2*Kneg))
    def deta_n_dI(Ig):
        return -(RT/p['F'])/(Kneg*np.sqrt(1+(Ig/(2*Kneg))**2))
    def solve_fixed(Iguess,eta0,blocked):
        for _ in range(p['num']['newton_max']+8):
            etan=eta_n(Iguess)
            eta=eta0+etan-Iguess*Rmat
            arg=np.clip(p['F']*eta/(2*RT),-18,18)
            ia=2*j0p*np.sinh(arg); ia[blocked]=0
            Icalc=float(np.sum(Aweight*ia))
            did=j0p*(p['F']/RT)*np.cosh(arg); did[blocked]=0
            de_dI=deta_n_dI(Iguess)-Rmat
            df=1-float(np.sum(Aweight*did*de_dI))
            Inew=Iguess-(Iguess-Icalc)/max(df,1e-12)
            if not np.isfinite(Inew): Inew=0.5*Iguess
            if abs(Inew-Iguess)<=p['num']['newton_tol']*max(1,abs(Inew)):
                Iguess=Inew; break
            Iguess=.6*Iguess+.4*Inew
        etan=eta_n(Iguess); eta=eta0+etan-Iguess*Rmat
        arg=np.clip(p['F']*eta/(2*RT),-18,18)
        ia=2*j0p*np.sinh(arg); ia[blocked]=0
        did=j0p*(p['F']/RT)*np.cosh(arg); did[blocked]=0
        return Iguess,ia,did,eta,float(etan)

    for _ in range(p['num']['branch_max']):
        Up=m.lfp_ocp_zk_high(x,p); eta0=Vcell+Un-Up-dpm
        I,ia,did,eta,etan=solve_fixed(I,eta0,blocked)
        new_block=((x<=2*p['num']['x_min'])&(ia>0)) | ((x>=1-2*p['num']['x_min'])&(ia<0))
        if np.array_equal(new_block,blocked): break
        blocked=new_block
    Up=m.lfp_ocp_zk_high(x,p); eta0=Vcell+Un-Up-dpm
    I,ia,did,eta,etan=solve_fixed(I,eta0,blocked)
    de_dI=deta_n_dI(I)-Rmat
    num=float(np.sum(Aweight*did)); den=1-float(np.sum(Aweight*did*de_dI))
    dIdV=num/max(den,1e-12)
    jmol=ia/p['F']
    dxp=-3*jmol/np.repeat(p['lfp']['psd']['R'][None,:],Npos,axis=0)/p['pos']['cmax']
    Jneg=-I/(p['F']*p['neg']['a_s']*p['geom']['L_neg'])
    Dnodes=m.graphite_diffusivity_ecker2015(np.clip(s['cn']/p['neg']['cmax'],p['num']['x_min'],1-p['num']['x_min']),p)
    dcn,cs=m.spherical_diffusion_rhs(s['cn'],p['neg']['Ds'],p['neg']['R'],Jneg,D_nodes=Dnodes)
    srcn=(1-p['elec']['t_plus'])*p['neg']['a_s']*Jneg*np.ones(p['disc']['Nneg'])
    srcs=np.zeros(p['disc']['Nsep'])
    srcp=(1-p['elec']['t_plus'])*np.sum(np.repeat(p['pos']['a_s_bins'][None,:],Npos,axis=0)*jmol,axis=1)
    dce=m.electrolyte_rhs(s['ce'],np.r_[srcn,srcs,srcp],p)
    dy=np.zeros(p['Nstate']); dy[p['idx']['ce']]=dce; dy[p['idx']['xp']]=dxp.ravel(); dy[p['idx']['cn']]=dcn
    return dict(dy=dy,I=I,dIdV=dIdV,Up=Up,Un=Un,eta_p=eta,i_p=ia,dir_p=np.sign(ia),
                xn_surface=float(np.clip(cs/p['neg']['cmax'],p['num']['x_min'],1-p['num']['x_min'])),
                Rct_n=Rctn0,Rohm=Rohm,eta_n=etan,Eneg=float(Un+etan))

m.cell_at_voltage=cell_at_voltage_bvneg
base=m.get_reference_params(); base['neg']['Ds_model']='ecker2015'

cases={
    'common5_dLAMn5': (
        m.make_mixed_degraded_cell(base,.05,0,0,'decoupled'),
        m.make_mixed_degraded_cell(base,.05,.05,0,'decoupled'),
        'Common 5\\% LLI; cell 2 +5\\% LAM$_n$'),
    'trajectory5_vs10': (
        m.make_mixed_degraded_cell(base,.05,.05,0,'decoupled'),
        m.make_mixed_degraded_cell(base,.10,.10,0,'decoupled'),
        '5\\% LLI+LAM$_n$ vs 10\\% LLI+LAM$_n$')
}

runs=[('common5_dLAMn5',.5),('common5_dLAMn5',1.0),('trajectory5_vs10',1.0)]
summary=[]; histories={}
for key,cr in runs:
    p1,p2,label=cases[key]
    y,_=m.init_parallel_at_common_ocv([p1,p2],3.30,.4)
    sim=m.simulate_cc_halfcycle([p1,p2],y,cr,'charge',max_step=25 if cr<1 else 20,rtol=3e-4,atol=2e-6)
    mismatch=np.abs(sim['I'][:,0]-sim['I'][:,1])/abs(sim['Iapp'])
    kp=int(np.argmax(mismatch)); rec=int(np.argmax(np.abs(sim['I'][kp])))+1
    slices=[]; k=0
    for p in [p1,p2]: slices.append(slice(k,k+p['Nstate'])); k+=p['Nstate']
    long=[]
    for it,(tt,yy,V) in enumerate(zip(sim['t'],sim['y'],sim['V'])):
        for cell,(sl,p) in enumerate(zip(slices,[p1,p2]),1):
            e=cell_at_voltage_bvneg(yy[sl],p,float(V))
            long.append(dict(t_s=tt,cell=cell,V_cell=V,I=sim['I'][it,cell-1],I_fraction=sim['I'][it,cell-1]/abs(sim['Iapp']),
                             Un=e['Un'],eta_n=e['eta_n'],Eneg=e['Eneg'],xn_surface=e['xn_surface']))
    df=pd.DataFrame(long); histories[(key,cr)]=(sim,df,label)
    for cell in [1,2]:
        z=df[df.cell==cell]
        summary.append(dict(case=key,C_rate=cr,cell=cell,M_peak=float(mismatch.max()),peak_recipient=rec,
                            min_Eneg_V=float(z.Eneg.min()),min_eta_n_V=float(z.eta_n.min()),
                            max_xn_surface=float(z.xn_surface.max()),duration_min=float(sim['t'][-1]/60)))
    df.to_csv(SUP/f'FigureS11_{key}_{cr:.1f}C.csv',index=False)

S=pd.DataFrame(summary)
S.to_csv(SUP/'TableS11_negative_electrode_potential.csv',index=False)

fig,axs=plt.subplots(2,2,figsize=(7.05,5.0),constrained_layout=True,sharex='col')
for col,key in enumerate(['common5_dLAMn5','trajectory5_vs10']):
    sim,df,label=histories[(key,1.0)]
    ax=axs[0,col]
    for cell in [1,2]:
        z=df[df.cell==cell]
        ax.plot(z.t_s/60,z.I_fraction,label=f'Cell {cell}')
    ax.axhline(.5,ls=':',lw=1)
    ax.set_ylabel('$I_i/|I_{app}|$')
    ax.set_title(label,fontsize=8.4)
    ax.legend(frameon=False)
    ax.text(.015,.98,'ab'[col],transform=ax.transAxes,va='top',ha='left',fontweight='bold',fontsize=9.3)
    ax=axs[1,col]
    for cell in [1,2]:
        z=df[df.cell==cell]
        ax.plot(z.t_s/60,1000*z.Eneg,label=f'Cell {cell}')
    ax.axhline(0,ls='--',lw=1)
    ax.set(xlabel='Time (min)',ylabel='$\\phi_{s,n}-\\phi_{e,n}$ (mV)')
    ax.legend(frameon=False)
    ax.text(.015,.98,'cd'[col],transform=ax.transAxes,va='top',ha='left',fontweight='bold',fontsize=9.3)
fig.savefig(SUP/'FigureS11_negative_electrode_potential.pdf',bbox_inches='tight')
fig.savefig(SUP/'FigureS11_negative_electrode_potential.png',bbox_inches='tight',dpi=320)
plt.close(fig)
print(S.to_string(index=False))
