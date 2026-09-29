import sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation"
sys.path.insert(0, str(ROOT))
from lfp_parallel import model as m


def main():
    base = m.get_reference_params()
    rows = []
    for label, a, b, kinetics in [
        ("common5_dLAMn5", (0.05, 0, 0), (0.05, 0.05, 0), "linear"),
        ("trajectory5_vs10", (0.05, 0.05, 0), (0.10, 0.10, 0), "linear"),
        ("common5_dLAMn5_bv", (0.05, 0, 0), (0.05, 0.05, 0), "butler_volmer"),
    ]:
        p1 = m.make_mixed_degraded_cell(base, *a)
        p2 = m.make_mixed_degraded_cell(base, *b)
        for p in [p1, p2]:
            p["neg"]["kinetics"] = kinetics
            if kinetics == "butler_volmer":
                p["neg"]["Ds_model"] = "ecker2015"
        y, info = m.init_parallel_at_common_ocv([p1, p2], 3.30, 0.4)
        start = time.time()
        s = m.simulate_cc_halfcycle(
            [p1, p2], y, 1.0, "charge", max_step=12, rtol=1.5e-4, atol=1.5e-6
        )
        z = m.compute_current_metrics(s["t"], s["I"], s["Iapp"])
        z.update(
            case=label,
            seconds=time.time() - start,
            steps=len(s["t"]),
            duration_min=s["t"][-1] / 60,
            V_final=s["V"][-1],
            success=s["success"],
            current_residual=float(np.max(np.abs(s["I"].sum(axis=1) - s["Iapp"]))),
        )
        ends = np.cumsum([0, p1["Nstate"], p2["Nstate"]])
        mn = []
        mx = []
        li = []
        salt = []
        for a1, b1, p in zip(ends[:-1], ends[1:], [p1, p2]):
            ev = [m.cell_at_voltage(yy[a1:b1], p, v) for yy, v in zip(s["y"], s["V"])]
            mn.append(min(e["Eneg"] for e in ev))
            mx.append(max(e["xn_surface_raw"] for e in ev))
            states = [m.unpack_cell_state(yy[a1:b1], p) for yy in s["y"]]
            rf = np.linspace(0, 1, p["disc"]["Nr_neg"] + 1)
            wv = np.diff(rf**3)
            inventories = np.array(
                [
                    p["Qn"] * np.sum(st["cn"] * wv) / p["neg"]["cmax"]
                    + p["Qp"] * np.mean(st["xp"] @ p["lfp"]["psd"]["w_volume"])
                    for st in states
                ]
            )
            li.append(
                float(np.max(np.abs(inventories - inventories[0])) / inventories[0])
            )
            salts = np.array(
                [np.sum(st["ce"] * p["eps_e_vec"] * p["dx_vec"]) for st in states]
            )
            salt.append(float(np.max(np.abs(salts - salts[0])) / salts[0]))
        z.update(
            min_Eneg=mn,
            max_xn_surface=mx,
            lithium_relative_drift=li,
            salt_relative_drift=salt,
        )
        rows.append(z)
        (OUT / "corrected_representative_runs.json").write_text(
            json.dumps(rows, indent=2)
        )
        np.savez_compressed(
            OUT / ("corrected_" + label + ".npz"),
            **{k: s[k] for k in ["t", "y", "V", "I", "Iapp"]},
        )
        print(z, flush=True)


if __name__ == "__main__":
    main()
