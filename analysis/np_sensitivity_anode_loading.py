"""Negative-loading N/P construction used by the controlled SI comparison."""
from pathlib import Path
import sys, copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lfp_parallel import model as m

from analysis.np_sensitivity import design_np


def base_at_np(base, target):
    p = copy.deepcopy(base)
    np0 = design_np(p)
    # Vary negative coating loading via thickness, a common cell-design lever.
    p["geom"]["L_neg"] = base["geom"]["L_neg"] * target / np0
    p = m.update_derived_params(p)
    # Cathode loading/lithium inventory is fixed as N/P is changed by anode loading.
    p["QLi_fresh"] = base["QLi_fresh"]
    p["QLi"] = p["QLi_fresh"]
    p["Q_nominal_ref"] = base["Q_nominal_ref"]
    return p
