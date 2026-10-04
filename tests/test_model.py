"""Numerical regression tests; run python -m unittest discover -s tests -v."""

import copy
import importlib
import unittest
from unittest.mock import patch
import numpy as np
from lfp_parallel import model as m


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.p = m.get_reference_params()

    def test_psd_monodisperse_state_size(self):
        self.p["lfp"]["psd_cv"] = 0
        p = m.update_derived_params(self.p)
        self.assertEqual(p["disc"]["Npsd"], 1)
        state = m.state_from_xp(p, 0.3)
        self.assertEqual(len(state), p["Nstate"])
        self.assertEqual(m.lfp_ocp_zk0d(np.array([0.3]), np.array([1]), p).shape, (1,))

    def test_blocking_and_release(self):
        x = np.array([2e-6, 1 - 2e-6, 0.5])
        j, slope = m._bounded_butler_volmer(
            np.array([0.1, -0.1, 0.1]), np.ones(3), x, 0.026, 2e-6
        )
        np.testing.assert_array_equal(j[:2], 0)
        np.testing.assert_array_equal(slope[:2], 0)
        j, _ = m._bounded_butler_volmer(
            np.array([-0.1, 0.1, 0.1]), np.ones(3), x, 0.026, 2e-6
        )
        self.assertLess(j[0], 0)
        self.assertGreater(j[1], 0)

    def test_spherical_flux_conservation(self):
        for d in [1e-14, 1e-20]:
            c = np.linspace(1000, 1200, 13)
            dc, cs = m.spherical_diffusion_rhs(c, d, 5e-6, 1e-8, D_nodes=np.full(13, d))
            radii = np.linspace(0, 5e-6, 14)
            volumes = 4 * np.pi / 3 * np.diff(radii**3)
            np.testing.assert_allclose(
                dc @ volumes, -4 * np.pi * (5e-6) ** 2 * 1e-8, rtol=1e-12
            )
            self.assertAlmostEqual(cs, c[-1] - 1e-8 * (5e-6 / 26) / d, places=5)

    def test_electrolyte_weighted_interface(self):
        p = {
            "eps_e_vec": np.ones(2),
            "elec": {"De": 1.0, "brugg": 1.0},
            "dx_vec": np.array([1.0, 3.0]),
        }
        p["eps_e_vec"] = np.array([0.5, 1.0])
        c = np.array([2.0, 4.0])
        dc = m.electrolyte_rhs(c, np.zeros(2), p)
        flux = -(4 - 2) / (0.5 / 0.5 + 1.5 / 1.0)
        np.testing.assert_allclose(dc, np.array([-flux / 0.5, flux / 3]))
        self.assertAlmostEqual(float(dc @ (p["eps_e_vec"] * p["dx_vec"])), 0)

    def test_degradation_conservation_of_exchange_area(self):
        p = m.make_mixed_degraded_cell(self.p, 0.1, 0.2, 0.15)
        for side, area in [("neg", "a_s"), ("pos", "a_s_total")]:
            self.assertAlmostEqual(
                p[side]["j0_ref"] * p[side][area],
                self.p[side]["j0_ref"] * self.p[side][area],
            )
            self.assertEqual(p[side]["eps_e"], self.p[side]["eps_e"])
        self.assertAlmostEqual(
            p["QLi"], self.p["QLi_fresh"] - 0.1 * self.p["Q_nominal_ref"]
        )

    def test_reject_invalid_parameters(self):
        for value in [-0.1, 1.0, float("nan")]:
            with self.assertRaises(ValueError):
                m.make_degraded_cell(self.p, "LAMn", value)
        with self.assertRaises(ValueError):
            m.make_mixed_degraded_cell(self.p, lamn=-0.1)
        with self.assertRaises(ValueError):
            m.make_degraded_cell(self.p, "LLI", 0.1, "typo")
        with self.assertRaises(ValueError):
            m.simulate_ocvr_pair_fast(self.p, self.p, 1, "typo")

    def test_full_branch_balance_and_surface(self):
        for kinetics in ["linear", "butler_volmer"]:
            self.p["neg"]["kinetics"] = kinetics
            y, _ = m.init_cell_at_ocv(self.p, 3.3)
            e = m.cell_at_voltage(y, self.p, 3.38)
            self.assertLess(abs(e["current_residual"]), 1e-7)
            self.assertAlmostEqual(
                e["Un"], float(m.graphite_ocp_verbrugge(e["xn_surface"])), places=12
            )
            self.assertAlmostEqual(e["Eneg"], e["Un"] + e["eta_n"], places=12)
            st = m.unpack_cell_state(e["dy"], self.p)
            w = np.diff(np.linspace(0, 1, self.p["disc"]["Nr_neg"] + 1) ** 3)
            lithium = self.p["Qn"] * (st["cn"] @ w) / self.p["neg"]["cmax"] + self.p[
                "Qp"
            ] * np.mean(st["xp"] @ self.p["lfp"]["psd"]["w_volume"])
            self.assertLess(abs(lithium), 1e-7)
            salt = st["ce"] @ (self.p["eps_e_vec"] * self.p["dx_vec"])
            self.assertLess(abs(salt), 1e-10)

    def test_current_controlled_fullmodel_equivalence(self):
        from lfp_parallel.current_controlled import evaluate_at_current
        for kinetics in ('linear','butler_volmer'):
            p=m.get_reference_params()
            p['neg']['kinetics']=kinetics
            y,_=m.init_cell_at_ocv(p,3.3)
            y[p['idx']['ce']]*=np.linspace(.97,1.03,p['Nelec'])
            for voltage in (3.22,3.38):
                reference=m.cell_at_voltage(y,p,voltage)
                direct=evaluate_at_current(y,p,reference['I'])
                self.assertAlmostEqual(direct['V'],voltage,places=8)
                np.testing.assert_allclose(direct['dy'],reference['dy'],rtol=2e-7,atol=1e-8)
                self.assertLess(abs(direct['current_residual']),1e-7)
                derivative=m.unpack_cell_state(direct['dy'],p)
                weights=np.diff(np.linspace(0,1,p['disc']['Nr_neg']+1)**3)
                lithium=(p['Qn']*(derivative['cn']@weights)/p['neg']['cmax']
                         +p['Qp']*np.mean(derivative['xp']@p['lfp']['psd']['w_volume']))
                self.assertLess(abs(lithium),1e-7)
                self.assertLess(abs(derivative['ce']@(p['eps_e_vec']*p['dx_vec'])),1e-10)

    def test_differential_conductance(self):
        y, _ = m.init_cell_at_ocv(self.p, 3.3)
        for kinetics in ["linear", "butler_volmer"]:
            self.p["neg"]["kinetics"] = kinetics
            e = m.cell_at_voltage(y, self.p, 3.38)
            h = 1e-5
            finite = (
                m.cell_at_voltage(y, self.p, 3.38 + h)["I"]
                - m.cell_at_voltage(y, self.p, 3.38 - h)["I"]
            ) / (2 * h)
            np.testing.assert_allclose(e["dIdV"], finite, rtol=1e-5)

    def test_zero_current_initialization(self):
        for severity in [0, 0.1]:
            p = m.make_mixed_degraded_cell(self.p, severity, severity)
            y, info = m.init_cell_at_ocv(p, 3.3)
            self.assertLess(abs(m.cell_at_voltage(y, p, 3.3)["I"]), 1e-7)

    def test_brent_fallback(self):
        self.p["num"]["newton_max"] = 0
        y = m.state_from_xp(self.p, 0.3)
        e = m.cell_at_voltage(y, self.p, 3.4)
        self.assertLess(abs(e["current_residual"]), 1e-7)

    def test_identical_branch_symmetry_and_swap(self):
        p1 = self.p
        p2 = m.make_mixed_degraded_cell(p1, 0.05, 0.05)
        y1, _ = m.init_cell_at_ocv(p1, 3.3)
        y2, _ = m.init_cell_at_ocv(p2, 3.3)
        v, i, _ = m.solve_parallel_voltage(0, np.r_[y1, y1], [p1, p1], lambda t: 20)
        np.testing.assert_allclose(i, [10, 10], atol=1e-6)
        v, i, _ = m.solve_parallel_voltage(0, np.r_[y1, y2], [p1, p2], lambda t: 20)
        w, j, _ = m.solve_parallel_voltage(0, np.r_[y2, y1], [p2, p1], lambda t: 20)
        self.assertAlmostEqual(v, w, places=10)
        np.testing.assert_allclose(i, j[::-1], atol=1e-7)

    def test_metrics_time_weighting(self):
        z = m.compute_current_metrics(
            np.array([0, 1, 10]), np.array([[3, 1], [3, 1], [3, 1]]), 4
        )
        self.assertAlmostEqual(z["M_peak"], 0.5)
        self.assertAlmostEqual(z["M_rms"], 0.5)
        self.assertAlmostEqual(z["Q_excess_Ahm2"], 10 / 3600)
        with self.assertRaises(ValueError):
            m.compute_current_metrics([0, 0], [[1, 1], [1, 1]], 2)

    def test_reduced_identical_pair_cutoff_and_charge_balance(self):
        s = m.simulate_ocvr_pair_fast_stateR(self.p, self.p, 1, "charge", dq_frac=0.01)
        np.testing.assert_allclose(s["I"][:, 0], s["I"][:, 1], atol=1e-10)
        self.assertAlmostEqual(s["V"][-1], self.p["Vmax"], places=7)
        transferred = (s["xp"][0] - s["xp"][-1]) * self.p["Qp"]
        np.testing.assert_allclose(
            transferred, 0.5 * s["Iapp"] * s["t"][-1], rtol=1e-10
        )
        with self.assertRaises(m.NumericalError):
            m.simulate_ocvr_pair_fast_stateR(self.p, self.p, 1, "charge", max_steps=2)

    def test_full_incomplete_run_is_rejected(self):
        y, _ = m.init_cell_at_ocv(self.p, 3.3)
        with self.assertRaises(m.NumericalError):
            m.simulate_cc_halfcycle([self.p], y, 1, "charge", max_time_factor=1e-6)

    def test_threshold_checks_earlier_nonmonotone_crossing(self):
        from analysis import threshold_scan as r

        def point(base, family, bg, delta):
            return dict(
                family=family,
                background=bg,
                delta=delta,
                M_peak=0.9 if delta == 0.00025 else 0.0,
            )

        with patch.object(r, "run_point", side_effect=point):
            below, hit = r.find_threshold(
                None, {}, r.FAMILIES[0], 0, 0.8, step=0.00025, max_delta=0.001
            )
        self.assertEqual(hit["delta"], 0.00025)
        self.assertEqual(below["delta"], 0)

    def test_np_uses_independent_host_capacities(self):
        from analysis.np_sensitivity import design_np, base_at_np
        from analysis.np_sensitivity_anode_loading import base_at_np as negative_design
        base=m.get_reference_params()
        self.assertAlmostEqual(design_np(base),base['Qn']/base['Qp'])
        changed=m.get_reference_params()
        changed['balancing_reference']['xp_0']=0.5
        self.assertEqual(design_np(changed),design_np(base))
        for target in (.9,1.,1.2):
            for make_design in (base_at_np,negative_design):
                p=make_design(base,target)
                self.assertAlmostEqual(design_np(p),target,places=12)
                self.assertEqual(p['QLi'],base['QLi'])
            positive=base_at_np(base,target)
            self.assertEqual(positive['Qn'],base['Qn'])
            negative=negative_design(base,target)
            self.assertEqual(negative['Qp'],base['Qp'])

    def test_fullmodel_discharge_through_lfp_saturation(self):
        from analysis.np_sensitivity import base_at_np
        from analysis.bounded_fullmodel import simulate_discharge
        base=m.get_reference_params()
        base['disc'].update(Nneg=3,Nsep=2,Npos=3,Npsd=5,Nr_neg=7)
        base=m.update_derived_params(base)
        design=base_at_np(base,1.2)
        degraded=m.make_mixed_degraded_cell(design,lamp=.075)
        cells=[design,degraded]
        y,_=m.init_parallel_at_common_ocv(cells,3.35,.75)
        sim=simulate_discharge(cells,y,max_step=5.,rtol=1e-5,atol=1e-7)
        self.assertAlmostEqual(sim['V'][-1],2.5,places=6)
        self.assertTrue(np.allclose(sim['I'].sum(axis=1),sim['Iapp'],atol=1e-7))
        start=0
        for p in cells:
            states=sim['y'][:,start:start+p['Nstate']]
            xp=states[:,p['idx']['xp']].reshape(-1,p['disc']['Npos'],p['disc']['Npsd'])
            xn=states[:,p['idx']['cn']] @ np.diff(np.linspace(0,1,p['disc']['Nr_neg']+1)**3)/p['neg']['cmax']
            self.assertGreaterEqual(xp.min(),0.)
            self.assertLessEqual(xp.max(),1.)
            lithium=p['Qp']*np.mean(xp @ p['lfp']['psd']['w_volume'],axis=1)+p['Qn']*xn
            self.assertLess(np.max(abs(lithium-p['QLi']))/p['QLi'],1e-6)
            start+=p['Nstate']

    def test_analysis_imports_do_not_run(self):
        names = ["core_studies", "diffusivity_sensitivity",
                 "negative_electrode_potential_sensitivity", "np_sensitivity",
                 "robustness", "threshold_scan", "publication_figures",
                 "graphical_abstract", "manuscript_revision_checks",
                 "fullmodel_lamp_map", "bounded_fullmodel"]
        with patch.object(
            m,
            "get_reference_params",
            side_effect=AssertionError("Simulation on import"),
        ):
            for name in names:
                importlib.import_module("analysis." + name)


if __name__ == "__main__":
    unittest.main()
