"""Verify the current result inventory, numerical provenance and data mappings."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.common import ROOT, OUT, SUP
from analysis.robustness import PROTOCOL_HASH, points
from analysis.threshold_figures import verified_data

MAIN=('Figure02_degradation_hysteresis','Figure03_lowrate_characterization',
      'Figure04_parallel_dynamics','Figure05_model_fidelity','Figure06_parameter_robustness',
      'Figure07_NP_design_sensitivity','Figure08_threshold_heterogeneity','Figure09_resistance_surface_area')
SI=('FigureS01_grid_convergence','FigureS02_ZK_verification','FigureS03_capacity_protocol_sensitivity',
    'FigureS04_Crate_sensitivity','FigureS05_graphite_diffusivity_sensitivity','FigureS06_LAMp_NP_maps',
    'FigureS07_threshold_summary','FigureS08_integrated_heterogeneity','FigureS09_same_mixed_trajectory',
    'FigureS10_negative_electrode_potential','FigureS11_fullmodel_electrode_trajectories')


def verify(check_manifest=False):
    fig2=pd.read_csv(OUT/'Figure02_fullmodel_trajectories.csv')
    fig2protocol=json.loads((OUT/'Figure02_protocol.json').read_text())
    fig2metrics=pd.read_csv(OUT/'Figure02_metrics.csv')
    assert fig2protocol['model']=='MP-SPMe'
    assert fig2protocol['graphite_kinetics']=='butler_volmer'
    assert fig2.xn_surface_raw.between(-1e-8,1+1e-8).all()
    assert fig2protocol['model_sha256']==hashlib.sha256((ROOT/'lfp_parallel/model.py').read_bytes()).hexdigest()
    assert set(fig2.case)=={'fresh','LLI','LAMn','LAMp'}
    assert len(fig2metrics)==8 and set(fig2metrics.model)=={'MP-SPMe'}
    assert fig2.lithium_error_relative.abs().max()<1e-6
    assert fig2.xn.between(0,1).all()
    assert np.allclose(fig2.Eneg,fig2.Un+fig2.eta_n,atol=1e-10)
    for (case,direction),g in fig2.groupby(['case','direction']):
        assert (np.diff(g.t_s)>0).all()
        assert np.allclose(abs(g.I_Apm2),fig2protocol['current_magnitude_Apm2'])
        assert np.isclose(g.V.iloc[-1],3.65 if direction=='charge' else 2.5,atol=1e-7)
        assert np.allclose(g.capacity_Ahm2,abs(g.I_Apm2)*g.t_s/3600)
        expected=fig2metrics[(fig2metrics.case==case)&(fig2metrics.direction==direction)]
        assert len(expected)==1 and np.isclose(g.capacity_Ahm2.iloc[-1],expected.capacity_Ahm2.iloc[0])
    assert fig2metrics.capacity_relative_difference.dropna().max()<.001
    assert fig2metrics.max_curve_voltage_difference_V.dropna().max()<.002
    fig3=pd.read_csv(OUT/'Figure03_fullmodel_trajectories.csv')
    protocol=json.loads((OUT/'Figure03_protocol.json').read_text())
    metrics=pd.read_csv(OUT/'Figure03_lowrate.csv')
    checks=pd.read_csv(OUT/'Figure03_solver_tolerance.csv')
    assert protocol['model']=='MP-SPMe' and protocol['graphite_kinetics']=='butler_volmer'
    assert protocol['model_sha256']==fig2protocol['model_sha256']
    assert protocol['current_controlled_sha256']==hashlib.sha256((ROOT/'lfp_parallel/current_controlled.py').read_bytes()).hexdigest()
    guard=pd.read_csv(OUT/'Figure03_composition_guard_check.csv')
    assert len(guard)==2 and guard.capacity_relative_difference.max()<.001
    assert guard.max_voltage_difference_V.max()<.002
    assert (metrics.Capacity_Ahm2<=metrics.negative_host_capacity_Ahm2*(1+1e-6)).all()
    assert (metrics.Capacity_Ahm2<=metrics.positive_host_capacity_Ahm2*(1+1e-6)).all()
    assert len(metrics)==3 and len(checks)==6
    assert set(metrics.case)==set(fig3.case)=={'LLI','LAMN','LAMP'}
    assert set(metrics.graphite_kinetics)=={'butler_volmer'}
    assert fig3.ce_min_molm3.min()>=1
    assert fig3.branch_current_residual_Apm2.abs().max()<1e-7
    assert metrics.Capacity_Ahm2.max()/metrics.Capacity_Ahm2.min()-1<protocol['matching_relative_tolerance']
    assert fig3.xn_surface_raw.between(-1e-8,1+1e-8).all()
    assert fig3.lithium_error_relative.abs().max()<1e-6
    assert checks.capacity_relative_difference.max()<.001
    assert checks.max_voltage_difference_V.max()<.002
    for (case,direction),g in fig3.groupby(['case','direction']):
        assert (np.diff(g.t_s)>0).all()
        assert np.allclose(abs(g.I_Apm2),protocol['current_magnitude_Apm2'])
        assert abs(g.V.iloc[-1]-(3.65 if direction=='charge' else 2.5))<1e-6
        assert np.allclose(g.capacity_Ahm2,abs(g.I_Apm2)*g.t_s/3600)
    robust=pd.read_csv(OUT/'Figure06_parameter_robustness.csv')
    assert len(robust)==len(points())==25
    for row,(_,key,value) in zip(robust.itertuples(),points()):
        assert row.Key==key and np.isclose(row.Value,value,rtol=1e-12,atol=0)
    assert set(robust.protocol_sha256)=={PROTOCOL_HASH}
    assert robust.matching_error_Ahm2.abs().max()<1e-4
    assert (robust.M_peak>=.95).sum()==23
    data,summary=verified_data()
    assert len(data)==1371 and len(summary)==16 and (summary.status=='not_reached').sum()==1
    maps=pd.read_csv(OUT/'FigureS06_LAMp_NP_maps.csv')
    assert len(maps)==242
    assert set(maps.model)=={'MP-SPMe'} and set(maps.status)=={'voltage_cutoff'}
    assert maps[['M_peak','M_rms','qex_norm']].notna().all().all()
    assert np.allclose(maps.endpoint_V,2.5,atol=1e-7)
    assert maps.lithium_error_relative.max()<1e-6
    assert not maps.duplicated(['design','LLI','dLAMp']).any()
    from analysis.fullmodel_lamp_map import protocol_hash, SOLVER
    assert set(maps.protocol_sha256)=={protocol_hash()}
    for name,g in maps.groupby('design'):
        assert np.allclose(g.rtol,SOLVER[name]['rtol'])
        assert np.allclose(g.atol,SOLVER[name]['atol'])
        assert np.allclose(g.max_step_s,SOLVER[name]['max_step'])
    s6protocol=json.loads((OUT/'FigureS06_protocol.json').read_text())
    assert s6protocol['model']=='MP-SPMe'
    assert s6protocol['model_sha256']==hashlib.sha256((ROOT/'lfp_parallel/model.py').read_bytes()).hexdigest()
    s6checks=pd.read_csv(SUP/'FigureS06_solver_tolerance.csv')
    assert len(s6checks)==4 and s6checks.M_peak_difference.max()<.01
    assert s6checks.qex_absolute_difference.max()<.002
    for row in s6checks.itertuples():
        match=maps[(maps.design==row.design)&np.isclose(maps.LLI,row.LLI)&np.isclose(maps.dLAMp,row.dLAMp)]
        assert len(match)==1 and np.isclose(match.M_peak.iloc[0],row.M_peak_reference)
        assert np.isclose(match.qex_norm.iloc[0],row.qex_reference)
    s6history=pd.read_csv(SUP/'FigureS06_selected_trajectories.csv')
    assert np.allclose(s6history.I1+s6history.I2,s6history.Iapp,rtol=1e-8,atol=1e-7)
    assert len(s6history.groupby(['design','LLI','dLAMp']))==4
    for _,g in s6history.groupby(['design','LLI','dLAMp']):
        assert np.isclose(g.V.iloc[-1],2.5,atol=1e-7)
        assert (np.diff(g.t_s)>0).all()
    from analysis.np_sensitivity import design_np
    from lfp_parallel import model as m
    assert np.isclose(maps[maps.design=='reference'].design_NP, design_np(m.get_reference_params())).all()
    assert np.isclose(maps[maps.design=='NP_1.20'].design_NP,1.2).all()
    npdata=pd.read_csv(OUT/'Figure07_NP_design_sensitivity.csv')
    assert len(npdata)==39
    assert np.allclose(npdata.NP_ratio,npdata.Qn_host_Ahm2/npdata.Qp_host_Ahm2)
    for _,g in npdata.groupby('mode'):
        assert np.allclose(g.sort_values('NP_ratio').NP_ratio,np.linspace(.9,1.2,13))
    protocol=json.loads((OUT/'Figure07_protocol.json').read_text())
    assert protocol['capacity_convention']=='full_0_1_intercalation_host_capacity'
    assert protocol['graphite_span']==protocol['LFP_span']==[0.,1.]
    assert protocol['model_sha256']==hashlib.sha256((ROOT/'lfp_parallel/model.py').read_bytes()).hexdigest()
    tolerance=pd.read_csv(SUP/'NP_solver_tolerance_check.csv')
    assert len(tolerance)==3 and tolerance.qex_relative_difference.max()<.004
    for row in tolerance.itertuples():
        match=npdata[(npdata['mode']==row.mode)&np.isclose(npdata.NP_ratio,row.NP_ratio)]
        assert len(match)==1 and np.isclose(match.qex_norm.iloc[0],row.qex_norm_reference)
    controlled=pd.read_csv(SUP/'matched_NP_design.csv')
    assert set(np.round(controlled.NP,8))=={1.,1.2}
    assert controlled[['M_peak','M_rms','q_excess']].notna().all().all()
    crate=pd.read_csv(SUP/'TableS06_Crate_sensitivity.csv')
    assert list(crate.C_rate)==[.1,.25,.5,1.]
    assert crate[['M_peak','M_rms','qex_norm']].notna().all().all()
    diffusion=pd.read_csv(SUP/'TableS03_graphite_diffusivity_sensitivity.csv')
    assert len(diffusion)==8 and diffusion[['M_peak','M_rms','qex_norm']].notna().all().all()
    mixed=pd.read_csv(SUP/'FigureS09_same_mixed_trajectory.csv')
    assert len(mixed)==121 and not mixed.duplicated(['s1','s2']).any()
    assert set(mixed.family)=={'same_mixed_trajectory'}
    for k in (1,2):
        assert np.allclose(mixed[f'LLI{k}'],mixed[f's{k}'])
        assert np.allclose(mixed[f'LAMn{k}'],.5*mixed[f's{k}'])
        assert np.allclose(mixed[f'LAMp{k}'],.5*mixed[f's{k}'])
    margin=pd.read_csv(SUP/'TableS06_negative_electrode_potential.csv')
    for case,rate in (('common5_dLAMn5',.5),('common5_dLAMn5',1.),('trajectory5_vs10',1.)):
        history=pd.read_csv(SUP/f'FigureS10_{case}_{rate:.1f}C.csv')
        assert set(history.cell)=={1,2}
        assert np.allclose(history.Eneg,history.Un+history.eta_n,atol=1e-10)
        fractions=history.pivot(index='t_s',columns='cell',values='I_fraction')
        assert np.allclose(fractions.sum(axis=1),1,atol=1e-7)
        for cell in (1,2):
            expected=margin[(margin.case==case)&(margin.C_rate==rate)&(margin.cell==cell)]
            assert len(expected)==1
            assert np.isclose(history[history.cell==cell].Eneg.min(),expected.min_Eneg_V.iloc[0],atol=1e-9)
    trajectories=pd.read_csv(SUP/'full_model_electrode_trajectories.csv')
    assert np.allclose(trajectories.I1+trajectories.I2,(trajectories.I1+trajectories.I2).iloc[0],atol=1e-7)
    assert len(pd.read_csv(SUP/'matched_NP_design.csv'))==4
    assert len(pd.read_csv(SUP/'cutoff_sensitivity.csv'))==3
    pdfs=[OUT/f'{name}.pdf' for name in MAIN]+[SUP/f'{name}.pdf' for name in SI]
    pdfs += [ROOT/'figures/Figure01_model_architecture.pdf',SUP/'Graphical_Abstract_revised.pdf']
    for path in pdfs:
        b=path.read_bytes()
        assert len(b)>1000 and b.startswith(b'%PDF') and b.rstrip().endswith(b'%%EOF'),path
    manifest=ROOT/'MANIFEST.sha256'
    if check_manifest and manifest.exists():
        for line in manifest.read_text().splitlines():
            digest,path=line.split('  ',1)
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    print('PASS: model/result provenance, 25 robustness variants, 1,371 threshold points,')
    print('39 N/P cases, 3 tolerance checks, 242 full-model electrode-balance map points,')
    print('121 mixed-path points, all S10 histories,')
    print('and all 21 manuscript/SI/graphical figure assets.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',action='store_true',help='also verify exact release bytes (before regenerating outputs)')
    verify(parser.parse_args().manifest)
