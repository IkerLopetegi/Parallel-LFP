"""LFP/graphite models with charge-positive, area-normalized currents.

Units: I [A m^-2], Qp/Qn/QLi [C m^-2], returned capacities [Ah m^-2],
concentrations [mol m^-3], lengths [m], potentials [V], ASR [ohm m^2].
See MODEL_GUIDE.md for state layout, model approximations and entry points.
"""

from __future__ import annotations
import copy, math
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp, trapezoid
from scipy.special import erfinv

F = 96485.33212
R_GAS = 8.314462618


class NumericalError(RuntimeError):
    """A numerical solve did not satisfy its physical or algebraic contract."""


def _mode_sign(mode):
    if mode not in ("charge", "discharge"):
        raise ValueError("mode must be 'charge' or 'discharge'")
    return 1 if mode == "charge" else -1


def _validate_fraction(value, name, upper=1.0):
    if not np.isfinite(value) or not 0 <= value < upper:
        raise ValueError(f"{name} must be finite and in [0, {upper})")
    return float(value)


# ---------- Thermodynamics and particle distribution ----------


def compute_regular_solution_limits(omega_rt: float):
    """Spinodal and binodal compositions of a symmetric regular solution."""
    if omega_rt <= 2:
        return dict(xs1=np.nan, xs2=np.nan, xalpha=np.nan, xbeta=np.nan)
    disc = math.sqrt(1 - 2 / omega_rt)
    xs1, xs2 = (1 - disc) / 2, (1 + disc) / 2
    f = lambda x: math.log(x / (1 - x)) + omega_rt * (1 - 2 * x)
    eps = 1e-12
    xa = brentq(f, eps, xs1 - eps)
    xb = brentq(f, xs2 + eps, 1 - eps)
    return dict(xs1=xs1, xs2=xs2, xalpha=xa, xbeta=xb)


def omega_rt_from_particle_radius(radius_m):
    """Particle-size-dependent Omega/(kBT), following Zelic & Katrasnik (2019).

    Zelic & Katrasnik define the characteristic C3 particle length
        L = 3.6338 V_p/A_p.
    The present model stores an equivalent spherical radius R; for a sphere
    V/A=R/3, hence L=1.2112667 R. Their Eqs. (16)-(17) relate the nucleation
    barrier Delta phi/kBT = -31.4435 nm/L + 1.42925 to the regular-solution
    barrier. Particles with L < 22 nm are assigned Omega=0, as in the paper.

    The conversion from equivalent spherical R to C3 L is a modeling mapping,
    not a measured particle-shape parameter.
    """
    R = np.atleast_1d(np.asarray(radius_m, dtype=float))
    out = np.zeros_like(R)
    for k, r in enumerate(R):
        L_nm = 1.2112666666666667 * r * 1e9
        if L_nm < 22.0:
            out[k] = 0.0
            continue
        target = -31.4435 / L_nm + 1.42925
        if target <= 0:
            out[k] = 0.0
            continue

        # Half the difference between symmetric spinodal chemical potentials.
        def barrier(w):
            d = math.sqrt(1 - 2 / w)
            xs1 = (1 - d) / 2
            return math.log(xs1 / (1 - xs1)) + w * (1 - 2 * xs1)

        out[k] = brentq(lambda w: barrier(w) - target, 2 + 1e-10, 10.0)
    return float(out[0]) if np.ndim(radius_m) == 0 else out


def build_psd(r_mean: float, cv: float, n: int):
    """Deterministic lognormal number distribution and volume weights."""
    if (
        not np.isfinite(r_mean)
        or r_mean <= 0
        or cv < 0
        or not np.isfinite(cv)
        or int(n) != n
        or n < 1
    ):
        raise ValueError(
            "PSD requires positive radius, nonnegative CV and positive integer bin count"
        )
    if n == 1 or cv == 0:
        return dict(
            R=np.array([r_mean]),
            w_number=np.array([1.0]),
            w_volume=np.array([1.0]),
            R_number_mean=r_mean,
            R_volume_mean=r_mean,
            cv=0.0,
        )
    sigma_ln = math.sqrt(math.log(1 + cv**2))
    mu_ln = math.log(r_mean) - 0.5 * sigma_ln**2
    q = (np.arange(1, n + 1) - 0.5) / n
    z = math.sqrt(2) * erfinv(2 * q - 1)
    radii = np.exp(mu_ln + sigma_ln * z)
    wn = np.ones(n) / n
    wv = wn * radii**3
    wv /= wv.sum()
    return dict(
        R=radii,
        w_number=wn,
        w_volume=wv,
        R_number_mean=float(np.sum(wn * radii)),
        R_volume_mean=float(np.sum(wv * radii)),
        cv=cv,
    )


def graphite_ocp_verbrugge(x):
    """Graphite OCP fit used in the original manuscript (Verbrugge et al.)."""
    x = np.clip(np.asarray(x, dtype=float), 1e-10, 1 - 1e-10)
    c = np.array(
        [
            0.783536091577375,
            -58.461699380293744,
            0.148099389333928,
            -0.040045422267324,
            18.275366641133953,
            -0.169250626374014,
            -0.003688396507869,
            24.422298845122281,
            -0.324145214809420,
            -5.714288636362947,
            20.257336075941385,
            -0.580842687886653,
            5.692875412845519,
            20.340933501171019,
            -0.580914902015057,
        ]
    )
    return (
        c[0] * np.exp(c[1] * x)
        + c[2]
        + c[3] * np.tanh(c[4] * (x + c[5]))
        + c[6] * np.tanh(c[7] * (x + c[8]))
        + c[9] * np.tanh(c[10] * (x + c[11]))
        + c[12] * np.tanh(c[13] * (x + c[14]))
    )


def graphite_diffusivity_baker_verbrugge(x, p):
    """Thermodynamic-factor graphite diffusivity sensitivity.

    Uses the Baker-Verbrugge form D(x) proportional to
    -x(1-x) F/(RT) dU/dx, normalized so D(0.5)=the reference Ds.
    It is used only as a sensitivity case, not as a cell-specific fit.
    """
    x = np.clip(np.asarray(x, dtype=float), 1e-5, 1 - 1e-5)
    h = 2e-5
    xp = np.clip(x + h, 1e-5, 1 - 1e-5)
    xm = np.clip(x - h, 1e-5, 1 - 1e-5)
    dU = (graphite_ocp_verbrugge(xp) - graphite_ocp_verbrugge(xm)) / (xp - xm)
    fac = -x * (1 - x) * p["F"] / (p["R"] * p["T"]) * dU
    x0 = 0.5
    xp0 = x0 + h
    xm0 = x0 - h
    dU0 = (float(graphite_ocp_verbrugge(xp0)) - float(graphite_ocp_verbrugge(xm0))) / (
        xp0 - xm0
    )
    fac0 = -x0 * (1 - x0) * p["F"] / (p["R"] * p["T"]) * dU0
    ratio = np.clip(fac / max(fac0, 1e-8), 0.05, 20.0)
    return p["neg"]["Ds"] * ratio


def graphite_diffusivity_ecker2015(x, p=None):
    """Graphite solid diffusivity from the Ecker et al. parameterization.

    Reference-temperature (25 degC) fit:
        D_s(x) = 8.4e-13 exp(-11.3 x) + 8.2e-15  [m^2 s^-1].
    The present study is isothermal at approximately room temperature, so no
    additional Arrhenius temperature correction is applied in this sensitivity.
    """
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return 8.4e-13 * np.exp(-11.3 * x) + 8.2e-15


def lfp_ocp_regular(x, p, omega_rt=None):
    """Homogeneous regular-solution LFP equilibrium potential."""
    x = np.clip(
        np.asarray(x, dtype=float), p["num"]["thermo_min"], 1 - p["num"]["thermo_min"]
    )
    w = p["lfp"]["Omega_RT_eff"] if omega_rt is None else np.asarray(omega_rt)
    RT = p["R"] * p["T"]
    return p["lfp"]["U0"] + RT / p["F"] * (np.log((1 - x) / x) + w * (2 * x - 1))


def lfp_ocp_equilibrium(x, p):
    """Reference equilibrium LFP OCP with Maxwell plateau for balancing only."""
    x = np.clip(
        np.asarray(x, dtype=float), p["num"]["thermo_min"], 1 - p["num"]["thermo_min"]
    )
    u = lfp_ocp_regular(x, p).copy()
    lim = p["lfp"]["limits_eff"]
    if p["lfp"]["Omega_RT_eff"] > 2:
        mask = (x > lim["xalpha"]) & (x < lim["xbeta"])
        u = np.where(mask, p["lfp"]["U0"], u)
    return u


def lfp_ocp_zk0d(x, direction, p):
    """Low-overpotential 0D phase-separation branch of Zelic & Katrasnik.

    direction > 0: LFP delithiation (cell charge)
    direction < 0: LFP lithiation (cell discharge)
    The final dimension may correspond to PSD bins, each with its own Omega(L).
    """
    x = np.clip(np.asarray(x, dtype=float), p["num"]["x_min"], 1 - p["num"]["x_min"])
    d = np.sign(np.broadcast_to(direction, x.shape)).astype(float)
    d = np.where(d == 0, 1.0, d)
    if x.ndim >= 1 and x.shape[-1] == p["disc"]["Npsd"]:
        w = np.broadcast_to(np.asarray(p["lfp"]["Omega_RT_bins"]), x.shape)
        u = lfp_ocp_regular(x, p, w)
        for j, wj in enumerate(p["lfp"]["Omega_RT_bins"]):
            if wj <= 2:
                continue
            lim = p["lfp"]["limits_bins"][j]
            xx = x[..., j]
            dd = d[..., j]
            # Eq. 15: low-overpotential charge plateau x_alpha < x <= x_s2
            charge_plateau = (dd > 0) & (xx > lim["xalpha"]) & (xx <= lim["xs2"])
            # Eq. 14: low-overpotential discharge plateau x_s1 < x <= x_beta
            discharge_plateau = (dd < 0) & (xx > lim["xs1"]) & (xx <= lim["xbeta"])
            uj = u[..., j]
            uj = np.where(charge_plateau | discharge_plateau, p["lfp"]["U0"], uj)
            u[..., j] = uj
        return u
    # Scalar/reference version, used only for diagnostics.
    u = lfp_ocp_regular(x, p).copy()
    lim = p["lfp"]["limits_eff"]
    if p["lfp"]["Omega_RT_eff"] > 2:
        charge_plateau = (d > 0) & (x > lim["xalpha"]) & (x <= lim["xs2"])
        discharge_plateau = (d < 0) & (x > lim["xs1"]) & (x <= lim["xbeta"])
        u = np.where(charge_plateau | discharge_plateau, p["lfp"]["U0"], u)
    return u


def lfp_ocp_zk_high(x, p):
    """High-overpotential Zelic-Katrasnik limit (their Eq. 11).

    The average particle chemical potential follows the regular-solution
    spinodal relation and is independent of charge/discharge direction.  This
    is the appropriate limiting constitutive relation for the finite-current
    dynamic simulations in this study.  Each PSD bin uses its own Omega(L).
    """
    x = np.clip(np.asarray(x, dtype=float), p["num"]["x_min"], 1 - p["num"]["x_min"])
    if x.ndim >= 1 and x.shape[-1] == p["disc"]["Npsd"]:
        w = np.broadcast_to(np.asarray(p["lfp"]["Omega_RT_bins"]), x.shape)
        return lfp_ocp_regular(x, p, w)
    return lfp_ocp_regular(x, p, p["lfp"]["Omega_RT_eff"])


# ---------- Parameters and degradation ----------


def get_reference_params():
    """Representative literature-based LFP/graphite parameter set.

    Geometry/capacity parameters are mainly based on Prada et al. (2012).
    The LFP phase-separation relation is from Zelic & Katrasnik (2019), while
    graphite OCP follows Verbrugge et al. (2017). The set is deliberately a
    representative composite parameterization, not a parameterization of one
    named commercial cell.
    """
    p = dict(F=F, R=R_GAS, T=298.15, Vmin=2.50, Vmax=3.65)
    p["geom"] = dict(L_pos=80e-6, L_sep=25e-6, L_neg=34e-6)
    # Geometry and phase fractions follow the A123/ANR26650-oriented
    # parameterization reported by Prada et al. The active-material fraction
    # is the electrochemically active fraction; LAM creates an inactive-solid
    # fraction rather than artificially converting solid volume into pore volume.
    p["pos"] = dict(eps_s=0.374, eps_f=0.200, eps_e=0.426, cmax=22806.0, j0_ref=0.05)
    p["sep"] = dict(eps_e=0.55)
    p["neg"] = dict(
        eps_s=0.58,
        eps_f=0.060,
        eps_e=0.360,
        cmax=30555.0,
        R=5e-6,
        j0_ref=0.5,
        Ds=1e-14,
        Ds_model="constant",
        kinetics="linear",
        surface_method="extrapolated",
    )
    p["lfp"] = dict(R_mean=50e-9, psd_cv=0.25, U0=3.422)
    p["elec"] = dict(
        c_ref=1200.0, c_init=1200.0, De=1.2e-10, kappa=0.78, t_plus=0.36, brugg=1.5
    )
    p["R_contact"] = 0.0
    p["disc"] = dict(Nneg=10, Nsep=5, Npos=10, Npsd=13, Nr_neg=13)
    p["num"] = dict(
        x_min=2e-6,
        thermo_min=1e-12,
        balance_min=1e-10,
        ce_min=1.0,
        newton_tol=1e-9,
        newton_max=18,
    )
    # Literature coordinates are used only to initialize cyclable Li inventory
    # and the approximate 1C reference. They are NOT treated as fixed operating
    # limits once a different set of OCP functions is combined.
    p["balancing_reference"] = dict(
        xp_0=0.7400, xn_0=0.0132, xp_100=0.0350, xn_100=0.8110
    )
    p = update_derived_params(p)
    a = p["balancing_reference"]
    q0 = p["Qp"] * a["xp_0"] + p["Qn"] * a["xn_0"]
    q100 = p["Qp"] * a["xp_100"] + p["Qn"] * a["xn_100"]
    p["QLi_fresh"] = 0.5 * (q0 + q100)
    p["QLi"] = p["QLi_fresh"]
    p["Q_nominal_ref"] = 0.5 * (
        p["Qp"] * (a["xp_0"] - a["xp_100"]) + p["Qn"] * (a["xn_100"] - a["xn_0"])
    )
    p["deg"] = dict(mode="FRESH", severity=0.0, kinetic_coupling="decoupled")
    return p


def update_derived_params(p):
    p = copy.deepcopy(p)
    p["lfp"]["psd"] = build_psd(
        p["lfp"]["R_mean"], p["lfp"]["psd_cv"], p["disc"]["Npsd"]
    )
    p["disc"]["Npsd"] = len(p["lfp"]["psd"]["R"])
    p["lfp"]["Omega_RT_bins"] = omega_rt_from_particle_radius(p["lfp"]["psd"]["R"])
    p["lfp"]["limits_bins"] = [
        (
            compute_regular_solution_limits(float(w))
            if w > 2
            else dict(xs1=np.nan, xs2=np.nan, xalpha=np.nan, xbeta=np.nan)
        )
        for w in p["lfp"]["Omega_RT_bins"]
    ]
    # A smooth electrode-level equilibrium OCP is needed only for balancing and
    # low-rate capacity matching.  Rather than importing a second arbitrary
    # regular-solution parameter, use the volume-weighted interaction parameter
    # implied by the same particle-size distribution.  Dynamic particles retain
    # their individual Omega(L) values.
    p["lfp"]["Omega_RT_eff"] = float(
        np.sum(p["lfp"]["psd"]["w_volume"] * p["lfp"]["Omega_RT_bins"])
    )
    p["lfp"]["limits_eff"] = compute_regular_solution_limits(p["lfp"]["Omega_RT_eff"])
    gn, gs, gp = p["geom"]["L_neg"], p["geom"]["L_sep"], p["geom"]["L_pos"]
    nn, ns, np_ = p["disc"]["Nneg"], p["disc"]["Nsep"], p["disc"]["Npos"]
    p["dx_neg"], p["dx_sep"], p["dx_pos"] = gn / nn, gs / ns, gp / np_
    p["dx_vec"] = np.r_[
        np.full(nn, p["dx_neg"]), np.full(ns, p["dx_sep"]), np.full(np_, p["dx_pos"])
    ]
    p["eps_e_vec"] = np.r_[
        np.full(nn, p["neg"]["eps_e"]),
        np.full(ns, p["sep"]["eps_e"]),
        np.full(np_, p["pos"]["eps_e"]),
    ]
    p["Nelec"] = len(p["dx_vec"])
    p["Qp"] = p["F"] * p["pos"]["eps_s"] * gp * p["pos"]["cmax"]
    p["Qn"] = p["F"] * p["neg"]["eps_s"] * gn * p["neg"]["cmax"]
    Rb = p["lfp"]["psd"]["R"]
    wv = p["lfp"]["psd"]["w_volume"]
    p["pos"]["a_s_bins"] = 3 * p["pos"]["eps_s"] * (wv / Rb)
    p["pos"]["a_s_total"] = float(np.sum(p["pos"]["a_s_bins"]))
    p["neg"]["a_s"] = 3 * p["neg"]["eps_s"] / p["neg"]["R"]
    nce = p["Nelec"]
    nx = np_ * p["disc"]["Npsd"]
    nr = p["disc"]["Nr_neg"]
    p["idx"] = dict(
        ce=slice(0, nce), xp=slice(nce, nce + nx), cn=slice(nce + nx, nce + nx + nr)
    )
    p["Nstate"] = nce + nx + nr
    p["z_pos"] = (np.arange(np_) + 0.5) * p["dx_pos"]
    return p


def make_degraded_cell(base, mode, severity, kinetic_coupling="decoupled"):
    """Synthetic degradation perturbations used to isolate causality.

    LLI removes cyclable lithium. LAM scales electrode active material volume.
    In the default 'decoupled' setting, j0_ref is rescaled to compensate the
    leading change in total exchange area, so the main comparison isolates
    thermodynamic/capacity effects from resistance growth.
    """
    p = copy.deepcopy(base)
    m = mode.upper().replace(" ", "").replace("_", "")
    s = _validate_fraction(float(severity), "severity")
    if kinetic_coupling not in ("coupled", "decoupled"):
        raise ValueError("kinetic_coupling must be coupled or decoupled")
    if m in ("FRESH", "NONE"):
        pass
    elif m == "LLI":
        p["QLi"] = base["QLi_fresh"] - s * base["Q_nominal_ref"]
    elif m in ("LAMP", "LAMPE", "LAMPOS"):
        a0 = base["pos"]["a_s_total"]
        p["pos"]["eps_s"] = base["pos"]["eps_s"] * (1 - s)
        # Lost active material is treated as electrochemically inactive solid;
        # electrolyte porosity is held fixed so the default LAM perturbation does
        # not introduce an unintended transport change.
        p["pos"]["eps_e"] = base["pos"]["eps_e"]
        p = update_derived_params(p)
        p["QLi"] = base["QLi_fresh"]
        if kinetic_coupling == "decoupled":
            p["pos"]["j0_ref"] = base["pos"]["j0_ref"] * a0 / p["pos"]["a_s_total"]
    elif m in ("LAMN", "LAMNE", "LAMNEG"):
        a0 = base["neg"]["a_s"]
        p["neg"]["eps_s"] = base["neg"]["eps_s"] * (1 - s)
        p["neg"]["eps_e"] = base["neg"]["eps_e"]
        p = update_derived_params(p)
        p["QLi"] = base["QLi_fresh"]
        if kinetic_coupling == "decoupled":
            p["neg"]["j0_ref"] = base["neg"]["j0_ref"] * a0 / p["neg"]["a_s"]
    else:
        raise ValueError(mode)
    if not 0 < p["QLi"] < p["Qn"] + p["Qp"]:
        raise ValueError("Degradation leaves no admissible lithium inventory")
    p["deg"] = dict(mode=m, severity=s, kinetic_coupling=kinetic_coupling)
    return p


def estimate_effective_asr(p):
    RT = p["R"] * p["T"]
    Rctn = RT / (p["F"] * p["neg"]["j0_ref"] * p["neg"]["a_s"] * p["geom"]["L_neg"])
    Rctp = RT / (
        p["F"] * p["pos"]["j0_ref"] * p["pos"]["a_s_total"] * p["geom"]["L_pos"]
    )
    kn = p["elec"]["kappa"] * p["neg"]["eps_e"] ** p["elec"]["brugg"]
    ks = p["elec"]["kappa"] * p["sep"]["eps_e"] ** p["elec"]["brugg"]
    kp = p["elec"]["kappa"] * p["pos"]["eps_e"] ** p["elec"]["brugg"]
    Re = (
        p["geom"]["L_neg"] / (2 * kn)
        + p["geom"]["L_sep"] / ks
        + p["geom"]["L_pos"] / (2 * kp)
    )
    return Rctn + Rctp + Re + p["R_contact"]


# ---------- Equilibrium balancing helpers (internal, not a paper result) ----------


def fullcell_equilibrium_ocv(p, xp):
    scalar = np.ndim(xp) == 0
    xp = np.atleast_1d(np.asarray(xp, dtype=float))
    xn = (p["QLi"] - p["Qp"] * xp) / p["Qn"]
    valid = (
        (xp > p["num"]["balance_min"])
        & (xp < 1 - p["num"]["balance_min"])
        & (xn > p["num"]["balance_min"])
        & (xn < 1 - p["num"]["balance_min"])
    )
    V = np.full_like(xp, np.nan)
    Up = V.copy()
    Un = V.copy()
    if np.any(valid):
        Up[valid] = lfp_ocp_equilibrium(xp[valid], p)
        Un[valid] = graphite_ocp_verbrugge(xn[valid])
        V[valid] = Up[valid] - Un[valid]
    if scalar:
        return float(V[0]), float(xn[0]), float(Up[0]), float(Un[0])
    return V, xn, Up, Un


def equilibrium_xp_at_voltage(p, Vtarget, side="low"):
    eps = p["num"]["balance_min"]
    xmin = max(eps, (p["QLi"] - p["Qn"] * (1 - eps)) / p["Qp"])
    xmax = min(1 - eps, (p["QLi"] - p["Qn"] * eps) / p["Qp"])
    if xmax <= xmin:
        raise RuntimeError("No lithium-conserving interval")
    lin = np.linspace(xmin, xmax, 4000)
    vals = fullcell_equilibrium_ocv(p, lin)[0] - Vtarget
    ids = np.where(
        np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (vals[:-1] * vals[1:] <= 0)
    )[0]
    if len(ids) == 0:
        # nearest internally valid point. This is used only to initialize a low-rate
        # characterization; the terminal cutoff itself is enforced dynamically.
        k = int(np.nanargmin(np.abs(vals)))
        return float(lin[k])
    j = ids[-1] if side == "low" else ids[0]
    return brentq(
        lambda z: fullcell_equilibrium_ocv(p, z)[0] - Vtarget, lin[j], lin[j + 1]
    )


# ---------- Fast 0D multiparticle characterization and parallel model ----------


def init_mp0d_at_voltage(p, Vtarget, side="low"):
    xp = equilibrium_xp_at_voltage(p, Vtarget, side)
    return np.full(p["disc"]["Npsd"], xp, dtype=float)


def mp0d_xn_from_xp(xp, p):
    xbar = float(np.sum(p["lfp"]["psd"]["w_volume"] * np.asarray(xp)))
    return (p["QLi"] - p["Qp"] * xbar) / p["Qn"]


def _mp0d_positive_potential(xp, p, I):
    RT = p["R"] * p["T"]
    direction = 1.0 if I >= 0 else -1.0
    Up = lfp_ocp_zk0d(xp, np.full_like(xp, direction), p)
    j0p = p["pos"]["j0_ref"] * 2 * np.sqrt(np.maximum(p["num"]["x_min"], xp * (1 - xp)))
    A = p["pos"]["a_s_bins"] * p["geom"]["L_pos"]
    blocked = ((I > 0) & (xp <= 2 * p["num"]["x_min"])) | (
        (I < 0) & (xp >= 1 - 2 * p["num"]["x_min"])
    )
    if np.all(blocked):
        phi = float(np.nanmax(Up) + 5.0) if I > 0 else float(np.nanmin(Up) - 5.0)
        return phi, Up, np.zeros_like(xp), A

    def f(phi):
        arg = np.clip(p["F"] * (phi - Up) / (2 * RT), -30, 30)
        ib = 2 * j0p * np.sinh(arg)
        ib = np.where(blocked, 0.0, ib)
        return float(np.sum(A * ib) - I)

    lo = float(np.nanmin(Up) - 1.5)
    hi = float(np.nanmax(Up) + 1.5)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        lo = float(np.nanmin(Up) - 5)
        hi = float(np.nanmax(Up) + 5)
    phi = brentq(f, lo, hi, maxiter=80)
    arg = np.clip(p["F"] * (phi - Up) / (2 * RT), -30, 30)
    ibin = 2 * j0p * np.sinh(arg)
    ibin = np.where(blocked, 0.0, ibin)
    return phi, Up, ibin, A


def mp0d_voltage_and_rates(state, p, Iapp, R_extra=0.0):
    xp = np.clip(
        np.asarray(state, dtype=float), p["num"]["x_min"], 1 - p["num"]["x_min"]
    )
    xn = float(
        np.clip(mp0d_xn_from_xp(xp, p), p["num"]["x_min"], 1 - p["num"]["x_min"])
    )
    phi, Up, ibin, A = _mp0d_positive_potential(xp, p, Iapp)
    Un = float(graphite_ocp_verbrugge(xn))
    RT = p["R"] * p["T"]
    j0n = p["neg"]["j0_ref"] * 2 * math.sqrt(max(p["num"]["x_min"], xn * (1 - xn)))
    Rctn = RT / (p["F"] * max(j0n, 1e-15) * p["neg"]["a_s"] * p["geom"]["L_neg"])
    kn = p["elec"]["kappa"] * p["neg"]["eps_e"] ** p["elec"]["brugg"]
    ks = p["elec"]["kappa"] * p["sep"]["eps_e"] ** p["elec"]["brugg"]
    kp = p["elec"]["kappa"] * p["pos"]["eps_e"] ** p["elec"]["brugg"]
    Rohm = (
        p["geom"]["L_neg"] / (2 * kn)
        + p["geom"]["L_sep"] / ks
        + p["geom"]["L_pos"] / (2 * kp)
        + p["R_contact"]
        + R_extra
    )
    V = phi - Un + Iapp * (Rctn + Rohm)
    dxp = -3 * (ibin / p["F"]) / (p["lfp"]["psd"]["R"] * p["pos"]["cmax"])
    # Frozen-state zero-net-current voltage, retaining the current-direction branch.
    direction = 1.0 if Iapp >= 0 else -1.0
    Up0 = lfp_ocp_zk0d(xp, np.full_like(xp, direction), p)
    j0p = p["pos"]["j0_ref"] * 2 * np.sqrt(np.maximum(p["num"]["x_min"], xp * (1 - xp)))

    def f0(ph):
        ar = np.clip(p["F"] * (ph - Up0) / (2 * RT), -25, 25)
        return float(np.sum(A * 2 * j0p * np.sinh(ar)))

    phi0 = brentq(f0, float(np.nanmin(Up0) - 0.5), float(np.nanmax(Up0) + 0.5))
    Uzero = phi0 - Un
    return (
        V,
        dxp,
        dict(
            Up=Up, Un=Un, ibin=ibin, phi_p=phi, Rctn=Rctn, Rohm=Rohm, Uzero=Uzero, xn=xn
        ),
    )


def simulate_mp0d_halfcycle_fixed(
    p,
    C_rate=0.05,
    mode="charge",
    state0=None,
    dq_frac=5e-4,
    max_steps=12000,
    R_extra=0.0,
):
    sgn = _mode_sign(mode)
    if not np.isfinite(C_rate) or C_rate <= 0:
        raise ValueError("C_rate must be positive and finite")
    I1C = p["Q_nominal_ref"] / 3600.0
    Iapp = sgn * C_rate * I1C
    if state0 is None:
        state0 = init_mp0d_at_voltage(
            p, p["Vmin"] if sgn > 0 else p["Vmax"], "low" if sgn > 0 else "high"
        )
    x = np.array(state0, dtype=float)
    Vcut = p["Vmax"] if sgn > 0 else p["Vmin"]
    dQ = dq_frac * (p["Q_nominal_ref"] / 3600.0)
    dt = dQ * 3600.0 / max(abs(Iapp), 1e-12)
    ts = [0.0]
    states = [x.copy()]
    vals = [mp0d_voltage_and_rates(x, p, Iapp, R_extra)]
    reached = False
    for _ in range(max_steps):
        V, dx, _ = vals[-1]
        if (sgn > 0 and V >= Vcut) or (sgn < 0 and V <= Vcut):
            reached = True
            break
        pred = np.clip(x + dt * dx, p["num"]["x_min"], 1 - p["num"]["x_min"])
        _, dx2, _ = mp0d_voltage_and_rates(pred, p, Iapp, R_extra)
        xnew = np.clip(
            x + 0.5 * dt * (dx + dx2), p["num"]["x_min"], 1 - p["num"]["x_min"]
        )
        vnew = mp0d_voltage_and_rates(xnew, p, Iapp, R_extra)
        crossed = (sgn > 0 and vnew[0] >= Vcut > V) or (sgn < 0 and vnew[0] <= Vcut < V)
        if crossed:
            delta = xnew - x
            fun = (
                lambda ff: mp0d_voltage_and_rates(
                    np.clip(x + ff * delta, p["num"]["x_min"], 1 - p["num"]["x_min"]),
                    p,
                    Iapp,
                    R_extra,
                )[0]
                - Vcut
            )
            try:
                frac = float(brentq(fun, 0, 1))
            except ValueError:
                frac = float(np.clip((Vcut - V) / (vnew[0] - V), 0, 1))
            xcut = np.clip(x + frac * delta, p["num"]["x_min"], 1 - p["num"]["x_min"])
            ts.append(ts[-1] + frac * dt)
            states.append(xcut.copy())
            vals.append(mp0d_voltage_and_rates(xcut, p, Iapp, R_extra))
            reached = True
            break
        x = xnew
        ts.append(ts[-1] + dt)
        states.append(x.copy())
        vals.append(vnew)
    V = np.array([v[0] for v in vals])
    U0 = np.array([v[2]["Uzero"] for v in vals])
    xn = np.array([v[2]["xn"] for v in vals])
    t = np.array(ts)
    st = np.array(states)
    cap = abs(Iapp) * t[-1] / 3600.0
    return dict(
        t=t,
        state=st,
        V=V,
        Uzero=U0,
        xn=xn,
        Iapp=Iapp,
        capacity_Ahm2=float(cap),
        reached_cutoff=reached,
        mode=mode,
        C_rate=C_rate,
    )


def simulate_mp0d_cycle_fixed(p, C_rate=0.05, dq_frac=5e-4, R_extra=0.0):
    ch = simulate_mp0d_halfcycle_fixed(
        p, C_rate, "charge", None, dq_frac=dq_frac, R_extra=R_extra
    )
    ds = simulate_mp0d_halfcycle_fixed(
        p, C_rate, "discharge", ch["state"][-1], dq_frac=dq_frac, R_extra=R_extra
    )
    return ch, ds


def apparent_capacity(p, C_rate=0.05, dq_frac=1e-3):
    ch, ds = simulate_mp0d_cycle_fixed(p, C_rate=C_rate, dq_frac=dq_frac)
    if not ch["reached_cutoff"] or not ds["reached_cutoff"]:
        raise NumericalError(
            "MP0D capacity characterization did not reach both cutoffs"
        )
    return ds["capacity_Ahm2"], ch, ds


def apparent_capacity_cached(p, C_rate=0.05, dq_frac=1e-3):
    """Compatibility entry point; no incomplete global parameter cache is used."""
    return apparent_capacity(p, C_rate, dq_frac)[0]


def match_capacity_lowrate(
    base, p_target, target_mode, bounds=(0, 0.40), C_rate=0.05, dq_frac=1e-3
):
    target = apparent_capacity_cached(p_target, C_rate, dq_frac)

    def f(s):
        return (
            apparent_capacity_cached(
                make_degraded_cell(base, target_mode, s), C_rate, dq_frac
            )
            - target
        )

    lo, hi = bounds
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        r = minimize_scalar(
            lambda z: abs(f(z)),
            bounds=bounds,
            method="bounded",
            options={"xatol": 2e-4},
        )
        sev = float(r.x)
    else:
        sev = float(brentq(f, lo, hi, xtol=2e-4))
    pm = make_degraded_cell(base, target_mode, sev)
    cm = apparent_capacity_cached(pm, C_rate, dq_frac)
    if abs(cm - target) > 1e-4 * max(target, 1e-12):
        raise NumericalError(
            "Capacity target cannot be matched within the requested severity bounds"
        )
    return (
        pm,
        sev,
        dict(target_capacity=target, matched_capacity=cm, error_Ahm2=cm - target),
    )


def init_mp0d_at_common_voltage(p, Vtarget, soc_hint=0.35):
    xp = equilibrium_xp_at_voltage(p, Vtarget, "low" if soc_hint < 0.5 else "high")
    return np.full(p["disc"]["Npsd"], xp, dtype=float)


def parallel_mp0d_split(state1, state2, p1, p2, Iapp, Rextra1=0.0, Rextra2=0.0):
    def residual(I1):
        I2 = Iapp - I1
        return (
            mp0d_voltage_and_rates(state1, p1, I1, Rextra1)[0]
            - mp0d_voltage_and_rates(state2, p2, I2, Rextra2)[0]
        )

    scale = max(abs(Iapp), 1.0)
    lo, hi = -1.5 * scale, 2.5 * scale
    flo, fhi = residual(lo), residual(hi)
    for _ in range(5):
        if flo * fhi <= 0:
            break
        lo -= 2 * scale
        hi += 2 * scale
        flo, fhi = residual(lo), residual(hi)
    if flo * fhi > 0:
        grid = np.linspace(lo, hi, 101)
        rr = np.array([residual(z) for z in grid])
        ids = np.where(rr[:-1] * rr[1:] <= 0)[0]
        if len(ids):
            lo, hi = grid[ids[0]], grid[ids[0] + 1]
        else:
            raise NumericalError("Could not bracket equal-voltage MP0D current split")
    I1 = float(brentq(residual, lo, hi, xtol=1e-8))
    I2 = Iapp - I1
    v1, d1, e1 = mp0d_voltage_and_rates(state1, p1, I1, Rextra1)
    v2, d2, e2 = mp0d_voltage_and_rates(state2, p2, I2, Rextra2)
    return 0.5 * (v1 + v2), np.array([I1, I2]), (d1, d2), (e1, e2)


def _state_on_lowrate_charge(p, Vtarget):
    ch = simulate_mp0d_halfcycle_fixed(p, 0.05, "charge", None, dq_frac=1e-3)
    k = int(np.argmin(np.abs(ch["V"] - Vtarget)))
    return ch["state"][k].copy()


def simulate_parallel_mp0d_pair(
    p1,
    p2,
    C_rate,
    mode,
    initial_voltage=3.30,
    Rextra1=0.0,
    Rextra2=0.0,
    dq_frac=1.5e-3,
    max_steps=4000,
    capacity_mode="surrogate",
):
    sgn = _mode_sign(mode)
    if not np.isfinite(C_rate) or C_rate <= 0:
        raise ValueError("C_rate must be positive and finite")
    if capacity_mode == "detailed":
        cap1 = apparent_capacity_cached(p1)
        cap2 = apparent_capacity_cached(p2)
    else:
        cap1 = lowrate_capacity_surrogate(p1)[0]
        cap2 = lowrate_capacity_surrogate(p2)[0]
    Iapp = sgn * C_rate * (cap1 + cap2)
    hint = 0.35 if sgn > 0 else 0.75
    x1 = init_mp0d_at_common_voltage(p1, initial_voltage, hint)
    x2 = init_mp0d_at_common_voltage(p2, initial_voltage, hint)
    Vcut = max(p1["Vmax"], p2["Vmax"]) if sgn > 0 else min(p1["Vmin"], p2["Vmin"])
    dQ = dq_frac * ((cap1 + cap2) / 2)
    dt = dQ * 3600 / max(abs(Iapp), 1e-12)
    ts = [0.0]
    X1 = [x1.copy()]
    X2 = [x2.copy()]
    vals = []
    vals.append(parallel_mp0d_split(x1, x2, p1, p2, Iapp, Rextra1, Rextra2))
    for _ in range(max_steps):
        V, I, _, _ = vals[-1]
        if (sgn > 0 and V >= Vcut) or (sgn < 0 and V <= Vcut):
            break
        _, _, (d1, d2), _ = parallel_mp0d_split(x1, x2, p1, p2, Iapp, Rextra1, Rextra2)
        a1 = np.clip(x1 + dt * d1, p1["num"]["x_min"], 1 - p1["num"]["x_min"])
        a2 = np.clip(x2 + dt * d2, p2["num"]["x_min"], 1 - p2["num"]["x_min"])
        _, _, (e1, e2), _ = parallel_mp0d_split(a1, a2, p1, p2, Iapp, Rextra1, Rextra2)
        x1 = np.clip(
            x1 + 0.5 * dt * (d1 + e1), p1["num"]["x_min"], 1 - p1["num"]["x_min"]
        )
        x2 = np.clip(
            x2 + 0.5 * dt * (d2 + e2), p2["num"]["x_min"], 1 - p2["num"]["x_min"]
        )
        ts.append(ts[-1] + dt)
        X1.append(x1.copy())
        X2.append(x2.copy())
        vals.append(parallel_mp0d_split(x1, x2, p1, p2, Iapp, Rextra1, Rextra2))
    V = np.array([z[0] for z in vals])
    I = np.array([z[1] for z in vals])
    Uzero = np.array([[z[3][0]["Uzero"], z[3][1]["Uzero"]] for z in vals])
    return dict(
        t=np.asarray(ts),
        V=V,
        I=I,
        Uzero=Uzero,
        state1=np.asarray(X1),
        state2=np.asarray(X2),
        Iapp=Iapp,
        mode=mode,
    )


def compute_current_metrics(t, I, Iapp=None):
    """Time-weighted two-branch metrics, including excess charge [Ah m^-2]."""
    t = np.asarray(t, dtype=float)
    I = np.asarray(I, dtype=float)
    if t.ndim != 1 or len(t) < 2 or I.shape != (len(t), 2):
        raise ValueError("Need at least two times and a (n_times, 2) current array")
    if (
        not np.all(np.isfinite(t))
        or not np.all(np.isfinite(I))
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError(
            "Times must increase strictly; times and currents must be finite"
        )
    difference = I[:, 0] - I[:, 1]
    rms = float(np.sqrt(trapezoid(difference**2, t) / (t[-1] - t[0])))
    peak = float(np.max(np.abs(difference)))
    excess = float(0.5 * trapezoid(np.abs(difference), t) / 3600)
    scale = abs(float(Iapp)) if Iapp is not None else 0.0
    if not np.isfinite(scale):
        raise ValueError("Iapp must be finite")
    return dict(
        I_rms_mismatch=rms,
        I_peak_mismatch=peak,
        Q_excess_Ahm2=excess,
        M_peak=peak / scale if scale > 1e-12 else np.nan,
        M_rms=rms / scale if scale > 1e-12 else np.nan,
    )


# ---------- Full SPMe-like multiparticle model ----------


def unpack_cell_state(y, p):
    ce = y[p["idx"]["ce"]]
    xp = y[p["idx"]["xp"]].reshape((p["disc"]["Npos"], p["disc"]["Npsd"]))
    cn = y[p["idx"]["cn"]]
    nn, ns = p["disc"]["Nneg"], p["disc"]["Nsep"]
    return dict(
        ce=ce,
        xp=xp,
        cn=cn,
        ce_neg=ce[:nn],
        ce_sep=ce[nn : nn + ns],
        ce_pos=ce[-p["disc"]["Npos"] :],
    )


def spherical_diffusion_rhs(c, D, Rp, J_out, D_nodes=None):
    c = np.asarray(c, dtype=float)
    if (
        len(c) < 1
        or Rp <= 0
        or D <= 0
        or (
            D_nodes is not None
            and (np.shape(D_nodes) != c.shape or np.any(np.asarray(D_nodes) <= 0))
        )
    ):
        raise ValueError(
            "Diffusivities and radius must be positive; D_nodes must match c"
        )
    N = len(c)
    rf = np.linspace(0, Rp, N + 1)
    rc = 0.5 * (rf[:-1] + rf[1:])
    Vsh = 4 * np.pi / 3 * (rf[1:] ** 3 - rf[:-1] ** 3)
    Af = 4 * np.pi * rf**2
    Nface = np.zeros(N + 1)
    if D_nodes is None:
        Df = np.full(N - 1, float(D))
        Dsurf = float(D)
    else:
        D_nodes = np.asarray(D_nodes, dtype=float)
        Df = 2 / (1 / D_nodes[:-1] + 1 / D_nodes[1:])
        Dsurf = float(D_nodes[-1])
    Nface[1:N] = -Df * np.diff(c) / np.diff(rc)
    Nface[-1] = J_out
    dc = -(Af[1:] * Nface[1:] - Af[:-1] * Nface[:-1]) / Vsh
    cs = c[-1] - J_out * (Rp - rc[-1]) / Dsurf
    return dc, cs


def electrolyte_rhs(ce, source, p):
    eps = p["eps_e_vec"]
    Dcell = p["elec"]["De"] * eps ** p["elec"]["brugg"]
    N = len(ce)
    J = np.zeros(N + 1)
    for k in range(N - 1):
        dist = 0.5 * p["dx_vec"][k] + 0.5 * p["dx_vec"][k + 1]
        Df = dist / (
            0.5 * p["dx_vec"][k] / Dcell[k] + 0.5 * p["dx_vec"][k + 1] / Dcell[k + 1]
        )
        J[k + 1] = -Df * (ce[k + 1] - ce[k]) / dist
    div = (J[1:] - J[:-1]) / p["dx_vec"]
    return (-div + source) / eps


def _graphite_diffusivity(c, p):
    """Node diffusivities [m^2/s], selected explicitly by parameter name."""
    x = np.clip(c / p["neg"]["cmax"], p["num"]["x_min"], 1 - p["num"]["x_min"])
    model = p["neg"].get("Ds_model", "constant")
    if model == "constant":
        return np.full_like(c, p["neg"]["Ds"])
    if model == "ecker2015":
        return graphite_diffusivity_ecker2015(x, p)
    if model == "baker_verbrugge":
        return graphite_diffusivity_baker_verbrugge(x, p)
    raise ValueError(f"Unknown graphite diffusivity: {model}")


def _bounded_butler_volmer(eta, exchange_current, x, thermal_voltage, x_min):
    """BV current [A/m^2 active area] and derivative, with one-sided blocking.

    Determine the bound mask from the unmasked trial current on every evaluation.
    Inward flux remains allowed; a continuous ramp in the last x_min of
    composition suppresses outward flux without an ODE discontinuity.
    The argument cap is a numerical overflow guard; its derivative is zero.
    """
    raw_argument = eta / (2 * thermal_voltage)
    argument = np.clip(raw_argument, -40, 40)
    trial = 2 * exchange_current * np.sinh(argument)
    # A narrow, continuous endpoint ramp avoids a discontinuous ODE when a
    # population reaches its composition bound. It is unity outside [xmin,2xmin]
    # and [1-2xmin,1-xmin], and zero for outward flux at the actual bound.
    lower_gate = np.clip((x - x_min) / x_min, 0.0, 1.0)
    upper_gate = np.clip((1 - x_min - x) / x_min, 0.0, 1.0)
    gate = np.where(trial > 0, lower_gate, upper_gate)
    current = gate * trial
    slope = gate * exchange_current / thermal_voltage * np.cosh(argument)
    slope = np.where(np.abs(raw_argument) >= 40, 0.0, slope)
    return current, slope


def cell_at_voltage(y, p, Vcell):
    """Evaluate one full MP-SPMe branch at a imposed terminal voltage [V].

    The scalar unknown is branch current density [A/m^2 geometric area].
    Graphite surface concentration, OCP and kinetics use the same reconstructed
    boundary value. The default graphite kinetics are linearized; set
    p['neg']['kinetics']='butler_volmer' for the potential-margin diagnostic.
    """
    state = unpack_cell_state(y, p)
    thermal_voltage = p["R"] * p["T"] / p["F"]
    x_min = p["num"]["x_min"]
    ce_n = max(float(np.mean(state["ce_neg"])), p["num"]["ce_min"])
    ce_p = np.maximum(state["ce_pos"], p["num"]["ce_min"])
    x_pos = np.clip(state["xp"], x_min, 1 - x_min)
    up = lfp_ocp_zk_high(x_pos, p)
    area_weights = p["dx_pos"] * p["pos"]["a_s_bins"][None, :]
    j0_pos = (
        p["pos"]["j0_ref"]
        * 2
        * np.sqrt(np.maximum(x_min, x_pos * (1 - x_pos)))
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
    concentration_drop = (
        2 * thermal_voltage * (1 - p["elec"]["t_plus"]) * np.log(ce_p / ce_n)
    )[:, None]
    diffusivities = _graphite_diffusivity(state["cn"], p)
    surface_distance = p["neg"]["R"] / (2 * len(state["cn"]))
    flux_per_current = -1 / (p["F"] * p["neg"]["a_s"] * p["geom"]["L_neg"])
    kinetics = p["neg"].get("kinetics", "linear")
    surface_method = p["neg"].get("surface_method", "extrapolated")
    if kinetics not in ("linear", "butler_volmer"):
        raise ValueError(f"Unknown graphite kinetics: {kinetics}")
    if surface_method not in ("extrapolated", "outer_shell"):
        raise ValueError(f"Unknown graphite surface method: {surface_method}")

    def graphite_terms(current):
        surface = state["cn"][-1]
        if surface_method == "extrapolated":
            surface -= current * flux_per_current * surface_distance / diffusivities[-1]
        x_surface = float(np.clip(surface / p["neg"]["cmax"], x_min, 1 - x_min))
        un = float(graphite_ocp_verbrugge(x_surface))
        exchange = (
            p["neg"]["j0_ref"]
            * 2
            * math.sqrt(max(x_min, x_surface * (1 - x_surface)))
            * math.sqrt(ce_n / p["elec"]["c_ref"])
        )
        exchange_area = exchange * p["neg"]["a_s"] * p["geom"]["L_neg"]
        rct = thermal_voltage / exchange_area
        eta_n = (
            -current * rct
            if kinetics == "linear"
            else -2 * thermal_voltage * np.arcsinh(current / (2 * exchange_area))
        )
        return un, float(eta_n), x_surface, float(surface / p["neg"]["cmax"]), rct

    def evaluate(current):
        un, eta_n, x_surface, x_raw, rct = graphite_terms(current)
        eta_p = Vcell + un + eta_n - up - concentration_drop - current * series
        reaction, slope = _bounded_butler_volmer(
            eta_p, j0_pos, x_pos, thermal_voltage, x_min
        )
        residual = current - float(np.sum(area_weights * reaction))
        return residual, reaction, slope, eta_p, un, eta_n, x_surface, x_raw, rct

    def residual_slope(current, values):
        # Includes the current dependence of graphite surface OCP and exchange
        # current; holding these fixed would give an inconsistent dI/dV.
        h = 1e-5 * max(1.0, abs(current))
        a = graphite_terms(current + h)
        b = graphite_terms(current - h)
        d_en_di = ((a[0] + a[1]) - (b[0] + b[1])) / (2 * h)
        return 1 - float(np.sum(area_weights * values[2] * (d_en_di - series)))

    tolerance = p["num"]["newton_tol"]
    current = 0.0
    for _ in range(p["num"]["newton_max"]):
        values = evaluate(current)
        if abs(values[0]) <= tolerance * max(1.0, abs(current)):
            break
        derivative = residual_slope(current, values)
        if not np.isfinite(derivative) or derivative <= 0:
            break
        update = current - values[0] / derivative
        if not np.isfinite(update):
            break
        # Backtracking prevents Newton overshoot across surface/kinetic limits.
        for _ in range(12):
            if abs(evaluate(update)[0]) < abs(values[0]):
                break
            update = 0.5 * (current + update)
        current = update
    values = evaluate(current)
    if abs(values[0]) > tolerance * max(1.0, abs(current)):
        scale = max(1.0, p["Q_nominal_ref"] / 3600, abs(current))
        for _ in range(24):
            lo, hi = -scale, scale
            if evaluate(lo)[0] * evaluate(hi)[0] <= 0:
                current = brentq(
                    lambda i: evaluate(i)[0], lo, hi, xtol=1e-12, rtol=1e-13
                )
                break
            scale *= 2
        else:
            raise NumericalError("Could not bracket branch-current root")
        values = evaluate(current)
    residual, reaction, slope, eta_p, un, eta_n, x_surface, x_raw, rct = values
    if not np.isfinite(residual) or abs(residual) > 10 * tolerance * max(
        1.0, abs(current)
    ):
        raise NumericalError(f"Branch current residual is {residual:g} A/m^2")
    derivative = residual_slope(current, values)
    if not np.isfinite(derivative) or derivative <= 0:
        raise NumericalError("Nonpositive differential branch resistance")
    d_id_v = float(np.sum(area_weights * slope)) / derivative
    molar_flux = reaction / p["F"]
    dx_pos = -3 * molar_flux / (p["lfp"]["psd"]["R"][None, :] * p["pos"]["cmax"])
    j_neg = current * flux_per_current
    dc_neg, _ = spherical_diffusion_rhs(
        state["cn"], p["neg"]["Ds"], p["neg"]["R"], j_neg, D_nodes=diffusivities
    )
    source_neg = np.full(
        p["disc"]["Nneg"], (1 - p["elec"]["t_plus"]) * p["neg"]["a_s"] * j_neg
    )
    source_pos = (1 - p["elec"]["t_plus"]) * np.sum(
        p["pos"]["a_s_bins"][None, :] * molar_flux, axis=1
    )
    dc_e = electrolyte_rhs(
        state["ce"], np.r_[source_neg, np.zeros(p["disc"]["Nsep"]), source_pos], p
    )
    dy = np.zeros(p["Nstate"])
    dy[p["idx"]["ce"]] = dc_e
    dy[p["idx"]["xp"]] = dx_pos.ravel()
    dy[p["idx"]["cn"]] = dc_neg
    return dict(
        dy=dy,
        I=current,
        dIdV=d_id_v,
        Up=up,
        Un=un,
        eta_p=eta_p,
        i_p=reaction,
        dir_p=np.sign(reaction),
        xn_surface=x_surface,
        xn_surface_raw=x_raw,
        Rct_n=rct,
        Rohm=rohms,
        eta_n=eta_n,
        Eneg=un + eta_n,
        current_residual=residual,
    )


def solve_parallel_voltage(t, y, cells, current_fun, Vguess=None):
    """Enforce sum(I_branch)=Iapp, with checked Newton/Brent solves."""
    ends = np.cumsum([0] + [p["Nstate"] for p in cells])
    applied = float(current_fun(t))
    if not np.isfinite(applied) or len(y) != ends[-1]:
        raise ValueError("Invalid applied current or parallel state size")

    def evaluate(voltage):
        values = [
            cell_at_voltage(y[a:b], p, voltage)
            for a, b, p in zip(ends[:-1], ends[1:], cells)
        ]
        currents = np.array([e["I"] for e in values])
        slope = sum(e["dIdV"] for e in values)
        return float(currents.sum() - applied), slope, currents, values

    lo = min(p["Vmin"] for p in cells) - 0.5
    hi = max(p["Vmax"] for p in cells) + 0.5
    voltage = float(np.clip(3.3 if Vguess is None else Vguess, lo, hi))
    tolerance = 1e-8 * max(1.0, abs(applied))
    for _ in range(12):
        residual, slope, currents, values = evaluate(voltage)
        if abs(residual) <= tolerance:
            return voltage, currents, values
        if not np.isfinite(slope) or slope <= 1e-12:
            break
        new_voltage = float(np.clip(voltage - residual / slope, lo, hi))
        if abs(new_voltage - voltage) < 1e-12:
            break
        voltage = new_voltage
    if evaluate(lo)[0] * evaluate(hi)[0] > 0:
        raise NumericalError(
            f"Applied current {applied:g} A/m^2 cannot be bracketed in voltage"
        )
    voltage = brentq(lambda v: evaluate(v)[0], lo, hi, xtol=1e-12)
    residual, _, currents, values = evaluate(voltage)
    if not np.isfinite(residual) or abs(residual) > tolerance:
        raise NumericalError(f"Parallel current residual is {residual:g} A/m^2")
    return voltage, currents, values


def state_from_xp(p, xp):
    xn = (p["QLi"] - p["Qp"] * xp) / p["Qn"]
    ce = np.full(p["Nelec"], p["elec"]["c_init"])
    xpstate = np.full((p["disc"]["Npos"], p["disc"]["Npsd"]), xp)
    cn = np.full(p["disc"]["Nr_neg"], xn * p["neg"]["cmax"])
    return np.r_[ce, xpstate.ravel(), cn]


def init_cell_at_ocv(p, Vtarget, soc_hint=0.4):
    """Initialize a homogeneous state at zero *net* current in the dynamic model.

    PSD populations can still exchange lithium internally: zero terminal current
    does not imply thermodynamic equilibrium of every population. soc_hint selects
    among multiple roots by normalized position in the admissible xp interval.
    """
    eps = p["num"]["x_min"]
    xmin = max(eps, (p["QLi"] - p["Qn"] * (1 - eps)) / p["Qp"])
    xmax = min(1 - eps, (p["QLi"] - p["Qn"] * eps) / p["Qp"])
    if xmin >= xmax:
        raise ValueError("No lithium-conserving initialization interval")

    # At I=0 there is no graphite surface correction or polarization.
    # Sum of positive BV currents must vanish at the requested terminal voltage.
    def residual(xp):
        xn = (p["QLi"] - p["Qp"] * xp) / p["Qn"]
        up = lfp_ocp_zk_high(np.full(p["disc"]["Npsd"], xp), p)
        eta = Vtarget + float(graphite_ocp_verbrugge(xn)) - up
        j0 = p["pos"]["j0_ref"] * 2 * np.sqrt(max(eps, xp * (1 - xp)))
        trial = 2 * j0 * np.sinh(np.clip(p["F"] * eta / (2 * p["R"] * p["T"]), -40, 40))
        trial = (
            np.where(
                trial > 0,
                np.clip((xp - eps) / eps, 0, 1),
                np.clip((1 - eps - xp) / eps, 0, 1),
            )
            * trial
        )
        return float(np.sum(p["pos"]["a_s_bins"] * p["geom"]["L_pos"] * trial))

    grid = np.linspace(xmin, xmax, 1000)
    values = np.array([residual(x) for x in grid])
    ids = np.where(values[:-1] * values[1:] <= 0)[0]
    if not len(ids):
        raise NumericalError(f"No dynamic zero-current state at {Vtarget} V")
    roots = [brentq(residual, grid[k], grid[k + 1], xtol=1e-14) for k in ids]
    xp = min(roots, key=lambda x: abs((xmax - x) / (xmax - xmin) - soc_hint))
    y = state_from_xp(p, xp)
    current = cell_at_voltage(y, p, Vtarget)["I"]
    if abs(current) > 1e-7:
        raise NumericalError("Initialization did not satisfy zero terminal current")
    return y, dict(xp=xp, V=float(Vtarget), error=0.0, current_residual=current)


def init_parallel_at_common_ocv(cells, Vtarget, soc_hint=0.4):
    ys = []
    infos = []
    for p in cells:
        y, i = init_cell_at_ocv(p, Vtarget, soc_hint)
        ys.append(y)
        infos.append(i)
    return np.concatenate(ys), infos


def simulate_cc_halfcycle(
    cells,
    y0,
    C_rate,
    mode,
    max_step=5.0,
    rtol=4e-5,
    atol=3e-7,
    max_time_factor=1.6,
    capacity_rate=0.05,
):
    """Integrate to the first shared voltage limit; raise on incomplete runs.

    C_rate uses the sum of branch C/20 usable capacities. BDF integrates the
    concentration/stoichiometry states; a converged algebraic solve supplies I,V.
    """
    sign = _mode_sign(mode)
    if not np.isfinite(C_rate) or C_rate <= 0 or max_time_factor <= 0:
        raise ValueError("C_rate and max_time_factor must be positive")
    caps = [lowrate_capacity(p, capacity_rate) for p in cells]
    applied = sign * C_rate * sum(caps)
    current = lambda t: applied
    cutoff = (
        min(p["Vmax"] for p in cells) if sign > 0 else max(p["Vmin"] for p in cells)
    )
    guess = [3.3]
    initial_v = solve_parallel_voltage(0, y0, cells, current)[0]
    if sign * (initial_v - cutoff) >= 0:
        raise NumericalError("Initial loaded voltage is already outside the cutoff")

    def rhs(t, y):
        voltage, _, evaluations = solve_parallel_voltage(t, y, cells, current, guess[0])
        guess[0] = voltage
        return np.concatenate([e["dy"] for e in evaluations])

    def event(t, y):
        return solve_parallel_voltage(t, y, cells, current, guess[0])[0] - cutoff

    event.terminal = True
    event.direction = sign
    sol = solve_ivp(
        rhs,
        (0, max_time_factor * 3600 / C_rate),
        y0,
        method="BDF",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=event,
    )
    if not sol.success or len(sol.t_events[0]) == 0:
        raise NumericalError(f"Half-cycle did not reach voltage cutoff: {sol.message}")
    voltages, currents = [], []
    for t, y in zip(sol.t, sol.y.T):
        v, i, _ = solve_parallel_voltage(t, y, cells, current, guess[0])
        guess[0] = v
        voltages.append(v)
        currents.append(i)
    return dict(
        t=sol.t,
        y=sol.y.T,
        V=np.asarray(voltages),
        I=np.asarray(currents),
        Iapp=applied,
        success=True,
        reached_cutoff=True,
        message=sol.message,
    )


def apply_parameter_variant(p, name, value):
    p = copy.deepcopy(p)
    name = name.lower()
    if name == "rmean_nm":
        p["lfp"]["R_mean"] = value * 1e-9
    elif name == "psd_cv":
        p["lfp"]["psd_cv"] = value
    elif name == "omega_rt":
        raise ValueError(
            "Omega is particle-size dependent in V6; vary particle size/PSD instead"
        )
    elif name == "ds_neg":
        p["neg"]["Ds"] = value
    elif name == "j0_pos_mult":
        p["pos"]["j0_ref"] *= value
    elif name == "j0_neg_mult":
        p["neg"]["j0_ref"] *= value
    elif name == "kappa":
        p["elec"]["kappa"] = value
    elif name == "de":
        p["elec"]["De"] = value
    elif name == "rcontact":
        p["R_contact"] = value
    else:
        raise ValueError(name)
    return update_derived_params(p)


# ---------- Low-rate characterization between terminal-voltage cutoffs ----------


def lowrate_ocvr_characterization(p, C_rate=0.05, n_curve=600):
    """Low-rate capacity/terminal-voltage characterization.

    Uses the equilibrium (Maxwell-constructed) LFP OCP and a state-dependent
    charge-transfer + electrolyte ASR. This diagnostic is used only for
    electrode balancing and capacity matching; the direction-dependent
    Zelic-Katrasnik branches are reserved for the multiparticle dynamics.
    """
    eps = p["num"]["balance_min"]
    xmin = max(eps, (p["QLi"] - p["Qn"] * (1 - eps)) / p["Qp"])
    xmax = min(1 - eps, (p["QLi"] - p["Qn"] * eps) / p["Qp"])
    if xmax <= xmin:
        raise RuntimeError("No lithium-conserving composition interval")
    Iabs = C_rate * p["Q_nominal_ref"] / 3600.0
    RT = p["R"] * p["T"]
    kn = p["elec"]["kappa"] * p["neg"]["eps_e"] ** p["elec"]["brugg"]
    ks = p["elec"]["kappa"] * p["sep"]["eps_e"] ** p["elec"]["brugg"]
    kp = p["elec"]["kappa"] * p["pos"]["eps_e"] ** p["elec"]["brugg"]
    Re = (
        p["geom"]["L_neg"] / (2 * kn)
        + p["geom"]["L_sep"] / ks
        + p["geom"]["L_pos"] / (2 * kp)
        + p["R_contact"]
    )

    def UR(x):
        xa = np.asarray(x, dtype=float)
        xac = np.clip(xa, eps, 1 - eps)
        xn = np.clip((p["QLi"] - p["Qp"] * xac) / p["Qn"], eps, 1 - eps)
        U = lfp_ocp_equilibrium(xac, p) - graphite_ocp_verbrugge(xn)
        j0n = p["neg"]["j0_ref"] * 2 * np.sqrt(np.maximum(eps, xn * (1 - xn)))
        j0p = p["pos"]["j0_ref"] * 2 * np.sqrt(np.maximum(eps, xac * (1 - xac)))
        Rn = RT / (p["F"] * j0n * p["neg"]["a_s"] * p["geom"]["L_neg"])
        Rp = RT / (p["F"] * j0p * p["pos"]["a_s_total"] * p["geom"]["L_pos"])
        return U, Rn + Rp + Re

    # Linear bulk grid plus logarithmic resolution near both composition bounds.
    d = xmax - xmin
    lin = np.linspace(xmin, xmax, 1800)
    edge = np.geomspace(1e-10, 1, 700)
    g = np.unique(np.r_[lin, xmin + d * edge, xmax - d * edge])
    g = g[(g >= xmin) & (g <= xmax)]
    g.sort()
    Ug, Rg = UR(g)
    Vcg = Ug + Iabs * Rg
    Vdg = Ug - Iabs * Rg

    def select_root(vals, target, which):
        z = vals - target
        ids = np.where(
            np.isfinite(z[:-1]) & np.isfinite(z[1:]) & (z[:-1] * z[1:] <= 0)
        )[0]
        if len(ids) == 0:
            raise RuntimeError("Low-rate terminal-voltage cutoff root not found")
        j = ids[0] if which == "first" else ids[-1]

        def f(x, sgn):
            u, r = UR(float(x))
            return float(u + sgn * Iabs * r - target)

        sgn = +1 if vals is Vcg else -1
        return float(brentq(lambda x: f(x, sgn), g[j], g[j + 1], xtol=1e-13))

    # xp decreases during charge. The high-SOC cutoff is the smallest charge
    # root; the low-SOC cutoff is the largest discharge root.
    xp_hi = select_root(Vcg, p["Vmax"], "first")
    xp_lo = select_root(Vdg, p["Vmin"], "last")
    if xp_lo <= xp_hi:
        raise RuntimeError("Invalid low-rate operating window")
    cap = float(p["Qp"] * (xp_lo - xp_hi) / 3600.0)
    xp_charge = np.linspace(xp_lo, xp_hi, n_curve)
    xp_discharge = np.linspace(xp_hi, xp_lo, n_curve)
    Uc, Rc = UR(xp_charge)
    Ud, Rd = UR(xp_discharge)
    V_charge = Uc + Iabs * Rc
    V_discharge = Ud - Iabs * Rd
    q_charge = p["Qp"] * (xp_lo - xp_charge) / 3600.0
    q_discharge = p["Qp"] * (xp_discharge - xp_hi) / 3600.0
    xn_hi = float((p["QLi"] - p["Qp"] * xp_hi) / p["Qn"])
    xn_lo = float((p["QLi"] - p["Qp"] * xp_lo) / p["Qn"])
    return dict(
        capacity_Ahm2=cap,
        xp_high=xp_hi,
        xp_low=xp_lo,
        xn_high=xn_hi,
        xn_low=xn_lo,
        Iabs=Iabs,
        q_charge=q_charge,
        V_charge=np.asarray(V_charge),
        q_discharge=q_discharge,
        V_discharge=np.asarray(V_discharge),
        xp_charge=xp_charge,
        xp_discharge=xp_discharge,
    )


def lowrate_capacity(p, C_rate=0.05):
    return lowrate_ocvr_characterization(p, C_rate=C_rate, n_curve=80)["capacity_Ahm2"]


def match_capacity_lowrate_ocvr(
    base, p_target, target_mode, bounds=(0, 0.5), C_rate=0.05
):
    target = lowrate_capacity(p_target, C_rate)
    f = (
        lambda s: lowrate_capacity(make_degraded_cell(base, target_mode, s), C_rate)
        - target
    )
    lo, hi = bounds
    flo, fhi = f(lo), f(hi)
    if flo * fhi <= 0:
        sev = float(brentq(f, lo, hi, xtol=1e-8))
    else:
        sev = float(
            minimize_scalar(
                lambda z: abs(f(z)),
                bounds=bounds,
                method="bounded",
                options={"xatol": 1e-7},
            ).x
        )
    pm = make_degraded_cell(base, target_mode, sev)
    cm = lowrate_capacity(pm, C_rate)
    if abs(cm - target) > 1e-4 * max(target, 1e-12):
        raise NumericalError(
            "Capacity target cannot be matched within the requested severity bounds"
        )
    return (
        pm,
        sev,
        dict(target_capacity=target, matched_capacity=cm, error_Ahm2=cm - target),
    )


# ---------- Fast low-rate capacity surrogate for dense degradation maps ----------
def lowrate_capacity_surrogate(p):
    """Fast deterministic low-rate capacity used in dense sweep maps.

    This now uses the same terminal-voltage-cutoff OCV-R characterization as
    the central capacity-matching analysis, avoiding fixed stoichiometric
    endpoints and the former composition-bound surrogate.
    """
    c = lowrate_ocvr_characterization(p, C_rate=0.05, n_curve=40)
    return c["capacity_Ahm2"], dict(
        xp_low=c["xp_low"],
        xp_high=c["xp_high"],
        xn_low=c["xn_low"],
        xn_high=c["xn_high"],
    )


def match_capacity_surrogate(base, p_target, target_mode, bounds=(0, 0.50)):
    target = lowrate_capacity_surrogate(p_target)[0]
    f = (
        lambda s: lowrate_capacity_surrogate(make_degraded_cell(base, target_mode, s))[
            0
        ]
        - target
    )
    lo, hi = bounds
    flo, fhi = f(lo), f(hi)
    if flo * fhi <= 0:
        sev = float(brentq(f, lo, hi, xtol=1e-6))
    else:
        sev = float(
            minimize_scalar(lambda z: abs(f(z)), bounds=bounds, method="bounded").x
        )
    pm = make_degraded_cell(base, target_mode, sev)
    cm = lowrate_capacity_surrogate(pm)[0]
    if abs(cm - target) > 1e-4 * max(target, 1e-12):
        raise NumericalError(
            "Capacity target cannot be matched within the requested severity bounds"
        )
    return (
        pm,
        sev,
        dict(target_capacity=target, matched_capacity=cm, error_Ahm2=cm - target),
    )


# ---------- Very fast OCV-R model for dense sweeps ----------
def scalar_cell_ocv(p, xp, direction):
    xn = (p["QLi"] - p["Qp"] * xp) / p["Qn"]
    if (
        xn <= p["num"]["balance_min"]
        or xn >= 1 - p["num"]["balance_min"]
        or xp <= p["num"]["balance_min"]
        or xp >= 1 - p["num"]["balance_min"]
    ):
        # still evaluate at clipped bounds to retain the steep thermodynamic asymptote
        xn = float(np.clip(xn, p["num"]["balance_min"], 1 - p["num"]["balance_min"]))
    return float(
        lfp_ocp_equilibrium(
            np.array(
                float(np.clip(xp, p["num"]["balance_min"], 1 - p["num"]["balance_min"]))
            ),
            p,
        )
        - graphite_ocp_verbrugge(xn)
    )


def scalar_xp_at_voltage(p, Vtarget, direction, soc_hint=0.4):
    eps = p["num"]["balance_min"]
    xmin = max(eps, (p["QLi"] - p["Qn"] * (1 - eps)) / p["Qp"])
    xmax = min(1 - eps, (p["QLi"] - p["Qn"] * eps) / p["Qp"])
    # combine linear grid with logarithmic resolution near both boundaries
    lin = np.linspace(xmin, xmax, 5000)
    lo = (
        np.geomspace(max(xmin, eps), min(xmax, 2e-2), 1200)
        if xmin < 2e-2
        else np.array([])
    )
    hi = (
        1 - np.geomspace(max(1 - xmax, eps), min(1 - xmin, 2e-2), 1200)
        if xmax > 0.98
        else np.array([])
    )
    g = np.unique(np.r_[lin, lo, hi])
    g = g[(g >= xmin) & (g <= xmax)]
    g.sort()
    xn = (p["QLi"] - p["Qp"] * g) / p["Qn"]
    xnc = np.clip(xn, p["num"]["balance_min"], 1 - p["num"]["balance_min"])
    v = lfp_ocp_equilibrium(g, p) - graphite_ocp_verbrugge(xnc) - Vtarget
    ids = np.where(v[:-1] * v[1:] <= 0)[0]
    if len(ids):
        # choose root closest to requested normalized position
        qs = (xmax - g[ids]) / (xmax - xmin)
        j = ids[np.argmin(abs(qs - soc_hint))]
        return float(
            brentq(lambda x: scalar_cell_ocv(p, x, direction) - Vtarget, g[j], g[j + 1])
        )
    raise NumericalError(f"No scalar OCV root at {Vtarget} V")


def simulate_ocvr_pair_fast(
    p1,
    p2,
    C_rate,
    mode,
    initial_voltage=3.30,
    Rextra1=0.0,
    Rextra2=0.0,
    dq_frac=5e-4,
    max_steps=6000,
):
    """Constant-ASR OCV-R pair, with adaptive integration and an exact cutoff."""
    return _simulate_ocvr_pair(
        p1,
        p2,
        C_rate,
        mode,
        initial_voltage,
        Rextra1,
        Rextra2,
        dq_frac,
        max_steps,
        state_resistance=False,
    )


def scalar_effective_asr(p, xp, R_extra=0.0):
    xn = float(
        np.clip(
            (p["QLi"] - p["Qp"] * xp) / p["Qn"],
            p["num"]["balance_min"],
            1 - p["num"]["balance_min"],
        )
    )
    xp = float(np.clip(xp, p["num"]["balance_min"], 1 - p["num"]["balance_min"]))
    RT = p["R"] * p["T"]
    j0n = (
        p["neg"]["j0_ref"] * 2 * math.sqrt(max(p["num"]["balance_min"], xn * (1 - xn)))
    )
    j0p = (
        p["pos"]["j0_ref"] * 2 * math.sqrt(max(p["num"]["balance_min"], xp * (1 - xp)))
    )
    Rn = RT / (p["F"] * max(j0n, 1e-15) * p["neg"]["a_s"] * p["geom"]["L_neg"])
    Rp = RT / (p["F"] * max(j0p, 1e-15) * p["pos"]["a_s_total"] * p["geom"]["L_pos"])
    kn = p["elec"]["kappa"] * p["neg"]["eps_e"] ** p["elec"]["brugg"]
    ks = p["elec"]["kappa"] * p["sep"]["eps_e"] ** p["elec"]["brugg"]
    kp = p["elec"]["kappa"] * p["pos"]["eps_e"] ** p["elec"]["brugg"]
    Re = (
        p["geom"]["L_neg"] / (2 * kn)
        + p["geom"]["L_sep"] / ks
        + p["geom"]["L_pos"] / (2 * kp)
    )
    return Rn + Rp + Re + p["R_contact"] + R_extra


def simulate_ocvr_pair_fast_stateR(
    p1,
    p2,
    C_rate,
    mode,
    initial_voltage=3.30,
    Rextra1=0.0,
    Rextra2=0.0,
    dq_frac=5e-4,
    max_steps=6000,
    cap1=None,
    cap2=None,
    x1_init=None,
    x2_init=None,
):
    """State-dependent-ASR OCV-R pair; dq_frac controls maximum time step.

    The former fixed-Euler implementation silently clipped electrode states.
    This implementation integrates lithium balance without state clipping and
    locates the shared terminal-voltage cutoff with an integration event.
    """
    return _simulate_ocvr_pair(
        p1,
        p2,
        C_rate,
        mode,
        initial_voltage,
        Rextra1,
        Rextra2,
        dq_frac,
        max_steps,
        cap1,
        cap2,
        x1_init,
        x2_init,
        state_resistance=True,
    )


def _simulate_ocvr_pair(
    p1,
    p2,
    C_rate,
    mode,
    initial_voltage,
    Rextra1,
    Rextra2,
    dq_frac,
    max_steps,
    cap1=None,
    cap2=None,
    x1_init=None,
    x2_init=None,
    state_resistance=True,
):
    sign = _mode_sign(mode)
    if not np.isfinite(C_rate) or C_rate <= 0 or dq_frac <= 0 or max_steps < 2:
        raise ValueError("Positive C_rate/dq_frac and max_steps >= 2 required")
    if Rextra1 < 0 or Rextra2 < 0:
        raise ValueError("Added resistances cannot be negative")
    cap1 = lowrate_capacity(p1) if cap1 is None else cap1
    cap2 = lowrate_capacity(p2) if cap2 is None else cap2
    if not np.isfinite(cap1 + cap2) or min(cap1, cap2) <= 0:
        raise ValueError("Branch capacities must be finite and positive")
    applied = sign * C_rate * (cap1 + cap2)
    hint = 0.35 if sign > 0 else 0.75
    x0 = [
        (
            scalar_xp_at_voltage(p1, initial_voltage, sign, hint)
            if x1_init is None
            else x1_init
        ),
        (
            scalar_xp_at_voltage(p2, initial_voltage, sign, hint)
            if x2_init is None
            else x2_init
        ),
    ]
    cutoff = min(p1["Vmax"], p2["Vmax"]) if sign > 0 else max(p1["Vmin"], p2["Vmin"])
    dt_max = dq_frac * 0.5 * (cap1 + cap2) * 3600 / abs(applied)
    constant_r = [
        estimate_effective_asr(p1) + Rextra1,
        estimate_effective_asr(p2) + Rextra2,
    ]

    def evaluate(x):
        u = np.array([scalar_cell_ocv(p1, x[0], sign), scalar_cell_ocv(p2, x[1], sign)])
        resistance = (
            np.array(
                [
                    scalar_effective_asr(p1, x[0], Rextra1),
                    scalar_effective_asr(p2, x[1], Rextra2),
                ]
            )
            if state_resistance
            else np.array(constant_r)
        )
        i1 = (u[1] - u[0] + resistance[1] * applied) / resistance.sum()
        currents = np.array([i1, applied - i1])
        return u[0] + resistance[0] * i1, currents, u, resistance

    def rhs(t, x):
        return -evaluate(x)[1] / np.array([p1["Qp"], p2["Qp"]])

    def cutoff_event(t, x):
        return evaluate(x)[0] - cutoff

    cutoff_event.terminal = True
    cutoff_event.direction = sign

    def bounds_event(t, x):
        xn = [(p["QLi"] - p["Qp"] * xx) / p["Qn"] for p, xx in zip((p1, p2), x)]
        return min(*x, *(1 - np.asarray(x)), *xn, *(1 - np.asarray(xn)))

    bounds_event.terminal = True
    bounds_event.direction = -1
    if bounds_event(0, x0) <= 0 or sign * cutoff_event(0, x0) >= 0:
        raise NumericalError("Invalid initial electrode state or loaded voltage")
    sol = solve_ivp(
        rhs,
        (0, dt_max * (max_steps - 1)),
        x0,
        max_step=dt_max,
        rtol=2e-7,
        atol=1e-10,
        events=(cutoff_event, bounds_event),
    )
    if not sol.success or len(sol.t_events[0]) == 0:
        raise NumericalError(
            f"OCV-R simulation did not reach voltage cutoff: {sol.message}"
        )
    evaluated = [evaluate(x) for x in sol.y.T]
    return dict(
        t=sol.t,
        xp=sol.y.T,
        V=np.array([e[0] for e in evaluated]),
        I=np.array([e[1] for e in evaluated]),
        U=np.array([e[2] for e in evaluated]),
        R=(
            np.array([e[3] for e in evaluated])
            if state_resistance
            else np.array(constant_r)
        ),
        Iapp=applied,
        mode=mode,
        success=True,
        reached_cutoff=True,
    )


# ---------- Mixed degradation helper (V8) ----------
def make_mixed_degraded_cell(
    base, lli=0.0, lamn=0.0, lamp=0.0, kinetic_coupling="decoupled"
):
    """Apply simultaneous LLI, LAMn and LAMp to a representative cell.

    Parameters are fractional severities (0.10 = 10%). LLI removes cyclable
    lithium in units of the fresh nominal usable charge (Q_nominal_ref). LAM reduces the active
    material volume fraction. In ``decoupled`` mode the reference exchange
    current density is rescaled so the leading-order product j0*a_s is retained,
    isolating thermodynamic/capacity effects. In ``coupled`` mode j0_ref is left
    unchanged, so LAM naturally reduces specific surface area and raises the
    charge-transfer contribution to branch resistance.
    """
    p = copy.deepcopy(base)
    lli = _validate_fraction(float(lli), "LLI")
    lamn = _validate_fraction(float(lamn), "LAMn", 0.95)
    lamp = _validate_fraction(float(lamp), "LAMp", 0.95)
    if any(v >= 0.95 for v in (lamn, lamp)):
        raise ValueError("LAM severity must remain below 95%")
    a0n = base["neg"]["a_s"]
    a0p = base["pos"]["a_s_total"]
    p["neg"]["eps_s"] = base["neg"]["eps_s"] * (1 - lamn)
    p["pos"]["eps_s"] = base["pos"]["eps_s"] * (1 - lamp)
    # Keep electrolyte porosity fixed: lost active solid is treated as inactive
    # solid so LAM does not silently introduce a pore-transport perturbation.
    p["neg"]["eps_e"] = base["neg"]["eps_e"]
    p["pos"]["eps_e"] = base["pos"]["eps_e"]
    p = update_derived_params(p)
    p["QLi_fresh"] = base["QLi_fresh"]
    p["Q_nominal_ref"] = base["Q_nominal_ref"]
    p["QLi"] = base["QLi_fresh"] - lli * base["Q_nominal_ref"]
    p["pos"]["j0_ref"] = base["pos"]["j0_ref"]
    p["neg"]["j0_ref"] = base["neg"]["j0_ref"]
    if kinetic_coupling == "decoupled":
        if lamp > 0:
            p["pos"]["j0_ref"] = (
                base["pos"]["j0_ref"] * a0p / max(p["pos"]["a_s_total"], 1e-30)
            )
        if lamn > 0:
            p["neg"]["j0_ref"] = (
                base["neg"]["j0_ref"] * a0n / max(p["neg"]["a_s"], 1e-30)
            )
    elif kinetic_coupling != "coupled":
        raise ValueError("kinetic_coupling must be 'decoupled' or 'coupled'")
    if not 0 < p["QLi"] < p["Qn"] + p["Qp"]:
        raise ValueError("Degradation leaves no admissible lithium inventory")
    p["deg"] = dict(
        mode="MIXED",
        severity=max(lli, lamn, lamp),
        LLI=lli,
        LAMn=lamn,
        LAMp=lamp,
        kinetic_coupling=kinetic_coupling,
    )
    return p


def capacity_difference_fraction(p1, p2, C_rate=0.05):
    q1 = lowrate_capacity(p1, C_rate)
    q2 = lowrate_capacity(p2, C_rate)
    return (q2 - q1) / (0.5 * (q1 + q2)), q1, q2
