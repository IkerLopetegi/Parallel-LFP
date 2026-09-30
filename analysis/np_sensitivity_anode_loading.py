"""Negative-loading N/P construction used by the controlled SI comparison."""
from pathlib import Path
import sys, copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lfp_parallel import model as m

def design_np(p):
    a = p["balancing_reference"]
    qn = p["Qn"] * (a["xn_100"] - a["xn_0"])
    qp = p["Qp"] * (a["xp_0"] - a["xp_100"])
    return qn / qp


def base_at_np(base, target):
    p = copy.deepcopy(base)
    np0 = design_np(p)
    # Vary negative coating loading via thickness, a common cell-design lever.
    p["geom"]["L_neg"] = base["geom"]["L_neg"] * target / np0
    p = m.update_derived_params(p)
    # Cathode loading/lithium inventory is fixed as N/P is changed by anode loading.
    p["QLi_fresh"] = base["QLi_fresh"]
    p["QLi"] = p["QLi_fresh"]
    # Define C rate and percentage LLI against each design's own fresh C/20 usable capacity.
    p["Q_nominal_ref"] = base["Q_nominal_ref"]
    for _ in range(2):
        q = m.lowrate_capacity(p, 0.05)
        p["Q_nominal_ref"] = q * 3600.0
    return p

