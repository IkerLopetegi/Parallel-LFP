"""Single-cell MP-SPMe evaluation at an imposed current.

Uses the constitutive and transport functions in model.py. For a single cell,
current is known, so only the positive reaction balance is solved for voltage.
This avoids an ill-conditioned nested current root near graphite saturation.
The optional num.graphite_x_min separates the graphite numerical guard from
the LFP endpoint guard; absent this option both use num.x_min.
It is mathematically equivalent to cell_at_voltage followed by voltage balance;
it is not a reduced model and is not a parallel-cell solver.
"""

import numpy as np
from scipy.optimize import brentq
from lfp_parallel import model as m


def evaluate_at_current(y, p, current, voltage_guess=3.3):
    """Return full RHS, loaded voltage and diagnostics for prescribed A/m²."""
    if not np.isfinite(current):
        raise ValueError("Finite imposed current required")
    state = m.unpack_cell_state(y, p)
    thermal = p["R"] * p["T"] / p["F"]
    eps = p["num"]["x_min"]
    ce_n = max(float(np.mean(state["ce_neg"])), p["num"]["ce_min"])
    ce_p = np.maximum(state["ce_pos"], p["num"]["ce_min"])
    xp = np.clip(state["xp"], eps, 1 - eps)
    up = m.lfp_ocp_zk_high(xp, p)
    area = p["dx_pos"] * p["pos"]["a_s_bins"][None, :]
    exchange_p = (
        2
        * p["pos"]["j0_ref"]
        * np.sqrt(np.maximum(eps, xp * (1 - xp)))
        * np.sqrt(ce_p[:, None] / p["elec"]["c_ref"])
    )
    conductivity = [
        p["elec"]["kappa"] * p[e]["eps_e"] ** p["elec"]["brugg"]
        for e in ("neg", "sep", "pos")
    ]
    rohms = (
        p["geom"]["L_neg"] / (2 * conductivity[0])
        + p["geom"]["L_sep"] / conductivity[1]
        + p["z_pos"] / conductivity[2]
    )
    series = (rohms + p["R_contact"])[:, None]
    concentration = (2 * thermal * (1 - p["elec"]["t_plus"]) * np.log(ce_p / ce_n))[
        :, None
    ]
    diffusivities = m._graphite_diffusivity(state["cn"], p)
    flux_per_current = -1 / (p["F"] * p["neg"]["a_s"] * p["geom"]["L_neg"])
    surface = state["cn"][-1]
    method = p["neg"].get("surface_method", "extrapolated")
    if method == "extrapolated":
        surface -= (
            current
            * flux_per_current
            * p["neg"]["R"]
            / (2 * len(state["cn"]))
            / diffusivities[-1]
        )
    elif method != "outer_shell":
        raise ValueError("Unknown graphite surface method")
    raw = float(surface / p["neg"]["cmax"])
    graphite_eps = p["num"].get("graphite_x_min", eps)
    xn = float(np.clip(raw, graphite_eps, 1 - graphite_eps))
    un = float(m.graphite_ocp_verbrugge(xn))
    exchange_n = (
        2
        * p["neg"]["j0_ref"]
        * np.sqrt(max(graphite_eps, xn * (1 - xn)))
        * np.sqrt(ce_n / p["elec"]["c_ref"])
    )
    exchange_area = exchange_n * p["neg"]["a_s"] * p["geom"]["L_neg"]
    kinetics = p["neg"].get("kinetics", "linear")
    if kinetics == "butler_volmer":
        eta_n = float(-2 * thermal * np.arcsinh(current / (2 * exchange_area)))
    elif kinetics == "linear":
        eta_n = -current * thermal / exchange_area
    else:
        raise ValueError("Unknown graphite kinetics")

    def balance(voltage):
        eta = voltage + un + eta_n - up - concentration - current * series
        reaction, slope = m._bounded_butler_volmer(eta, exchange_p, xp, thermal, eps)
        return (
            float(np.sum(area * reaction) - current),
            reaction,
            eta,
            float(np.sum(area * slope)),
        )

    lo, hi = p["Vmin"] - 0.5, p["Vmax"] + 0.5
    voltage = float(np.clip(voltage_guess, lo, hi))
    tolerance = p["num"]["newton_tol"] * max(1, abs(current))
    for _ in range(p["num"]["newton_max"]):
        residual, reaction, eta_p, slope = balance(voltage)
        if abs(residual) <= tolerance:
            break
        if not np.isfinite(slope) or slope <= 0:
            break
        voltage = float(np.clip(voltage - residual / slope, lo, hi))
    if abs(balance(voltage)[0]) > tolerance:
        if balance(lo)[0] * balance(hi)[0] > 0:
            raise m.NumericalError("Imposed current cannot be bracketed in voltage")
        voltage = brentq(lambda v: balance(v)[0], lo, hi, xtol=1e-12)
    residual, reaction, eta_p, _ = balance(voltage)
    if not np.isfinite(residual) or abs(residual) > 10 * tolerance:
        raise m.NumericalError("Current-controlled positive balance failed")
    molar_flux = reaction / p["F"]
    dxp = -3 * molar_flux / (p["lfp"]["psd"]["R"][None, :] * p["pos"]["cmax"])
    j_neg = current * flux_per_current
    dcn, _ = m.spherical_diffusion_rhs(
        state["cn"], p["neg"]["Ds"], p["neg"]["R"], j_neg, D_nodes=diffusivities
    )
    source_n = np.full(
        p["disc"]["Nneg"], (1 - p["elec"]["t_plus"]) * p["neg"]["a_s"] * j_neg
    )
    source_p = (1 - p["elec"]["t_plus"]) * np.sum(
        p["pos"]["a_s_bins"][None, :] * molar_flux, axis=1
    )
    dce = m.electrolyte_rhs(
        state["ce"], np.r_[source_n, np.zeros(p["disc"]["Nsep"]), source_p], p
    )
    dy = np.zeros(p["Nstate"])
    dy[p["idx"]["ce"]] = dce
    dy[p["idx"]["xp"]] = dxp.ravel()
    dy[p["idx"]["cn"]] = dcn
    return dict(
        dy=dy,
        V=voltage,
        I=current,
        Up=up,
        Un=un,
        eta_p=eta_p,
        i_p=reaction,
        dir_p=np.sign(reaction),
        xn_surface=xn,
        xn_surface_raw=raw,
        Rct_n=thermal / exchange_area,
        Rohm=rohms,
        eta_n=eta_n,
        Eneg=un + eta_n,
        current_residual=-residual,
    )
