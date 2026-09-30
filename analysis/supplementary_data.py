"""Generate the reduced-model map data used in SI Figures S8 and S9."""
import pandas as pd
from analysis.common import OUT, SUP, prepare_output
from analysis.core_studies import GRID, get_cell_cached, fast_pair_metrics
from lfp_parallel import model as m


def charge_family_maps(base=None):
    prepare_output()
    base = m.get_reference_params() if base is None else base
    cache, rows = {}, []
    for s1 in GRID:
        for s2 in GRID:
            for family, a, b in (
                ('common_LLI_plus_dLAMn', (s1, 0, 0), (s1, s2, 0)),
                ('same_LLI_LAMn_trajectory', (s1, s1, 0), (s2, s2, 0)),
            ):
                p1, q1 = get_cell_cached(cache, base, *a)
                p2, q2 = get_cell_cached(cache, base, *b)
                z = fast_pair_metrics(p1, p2, 'charge', 3.30, q1, q2)
                rows.append(dict(family=family, s1=s1, s2=s2,
                                 **{k:z[k] for k in ('M_peak','M_rms','qex_norm','capdiff')}))
        pd.DataFrame(rows).to_csv(OUT / 'Figure08_realistic_maps.csv', index=False)
        print('charge-family maps:', s1, flush=True)


def mixed_trajectory_map(base=None):
    prepare_output()
    base = m.get_reference_params() if base is None else base
    cache, rows = {}, []
    for s1 in GRID:
        p1, q1 = get_cell_cached(cache, base, s1, .5*s1, .5*s1)
        for s2 in GRID:
            p2, q2 = get_cell_cached(cache, base, s2, .5*s2, .5*s2)
            z = fast_pair_metrics(p1, p2, 'charge', 3.30, q1, q2)
            rows.append(dict(family='same_mixed_trajectory', s1=s1, s2=s2,
                             LLI1=s1, LAMn1=.5*s1, LAMp1=.5*s1,
                             LLI2=s2, LAMn2=.5*s2, LAMp2=.5*s2,
                             **{k:z[k] for k in ('M_peak','M_rms','qex_norm','capdiff')}))
        pd.DataFrame(rows).to_csv(SUP / 'FigureS09_same_mixed_trajectory.csv', index=False)
        print('mixed-trajectory map:', s1, flush=True)


if __name__ == '__main__':
    charge_family_maps()
    mixed_trajectory_map()
