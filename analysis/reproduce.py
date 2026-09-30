"""Reproduce the final paper and SI; run from the repository root.

All numerical studies can be expensive. Existing full-model threshold and
robustness checkpoints resume only when the numerical configuration matches.
"""
import argparse
import subprocess
import sys
from analysis.common import prepare_output
from lfp_parallel import model as m

STAGES = ('verify','figures','saved','core','robustness','np','thresholds',
          'supplementary','resistance','diffusivity','potential','checks','all')


def run_module(name, *arguments):
    subprocess.run([sys.executable, '-m', 'analysis.'+name, *arguments], check=True)


def run_stage(stage):
    prepare_output()
    if stage == 'all':
        for name in ('core','robustness','np','thresholds','supplementary',
                     'resistance','diffusivity','potential','checks','saved','verify'):
            run_stage(name)
    elif stage == 'verify':
        run_module('verify_results')
    elif stage == 'figures':
        run_module('publication_figures', '--stage','all')
    elif stage == 'saved':
        for part in ('saved','si','graphical'):
            run_module('publication_figures','--stage',part)
    elif stage == 'core':
        run_module('publication_figures','--stage','core')
    elif stage == 'robustness':
        from analysis.robustness import run_all
        run_all()
    elif stage == 'np':
        run_module('np_sensitivity')
    elif stage == 'thresholds':
        from analysis.threshold_scan import FAMILIES, BACKGROUNDS
        for family in FAMILIES:
            for bg in BACKGROUNDS:
                run_module('threshold_scan','--family',family,'--background',str(bg))
        run_module('assemble_thresholds')
        run_module('threshold_figures')
    elif stage == 'supplementary':
        from analysis.core_studies import supplementary_studies
        from analysis.supplementary_data import charge_family_maps, mixed_trajectory_map
        from analysis.supplementary_maps import lamp_map
        base=m.get_reference_params()
        lli=m.make_degraded_cell(base,'LLI',.10)
        _,sn,_=m.match_capacity_lowrate_ocvr(base,lli,'LAMn',bounds=(0,.25),C_rate=.05)
        supplementary_studies(base,sn)
        charge_family_maps(base)
        mixed_trajectory_map(base)
        lamp_map()
    elif stage == 'resistance':
        from analysis.core_studies import figure9_resistance
        figure9_resistance(m.get_reference_params())
    elif stage == 'diffusivity':
        run_module('diffusivity_sensitivity')
    elif stage == 'potential':
        run_module('negative_electrode_potential_sensitivity')
    elif stage == 'checks':
        run_module('manuscript_revision_checks')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, default='verify')
    run_stage(parser.parse_args().stage)


if __name__ == '__main__':
    main()
