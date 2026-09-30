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
    invalid=maps.status=='electrode_bound_first'
    assert invalid.sum()==22 and maps.loc[invalid,'M_peak'].isna().all()
    assert maps.loc[~invalid,'M_peak'].notna().all()
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
    print('242 electrode-balance map points, 121 mixed-path points, all S10 histories,')
    print('and all 21 manuscript/SI/graphical figure assets.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',action='store_true',help='also verify exact release bytes (before regenerating outputs)')
    verify(parser.parse_args().manifest)
