"""A's 2026-11-07 simulator self-test case, as executable unit tests.

Source: A -> B handoff v2 (`handoff/A到B_v2_2026-11-07/03_仿真器自检用例.md`,
machine-readable values in `selftest_case.json`).

The lift direction in A's section 3 is written as ``w_hat - v_hat (w_hat . v_hat)``,
which is 90 deg away from the physical Magnus direction ``w_hat x v_hat`` that
produced the quoted spin axis under A's own inversion convention
``a_perp = k CL |v|^2 (w_hat x v_hat)``. A reviewed the evidence and confirmed
the correction (2026-11-07): the integration self-test uses
``unit(w_hat x v_hat)`` and handles the zero-cross-product case. With this
direction, constant (CD, CL) reproduces the posed case points to 1.4 mm /
2.6 mm, so the ``"constant coefficients cannot fit both points"`` conclusion in
A's section 3 is withdrawn. See `对A自检用例的核对.md`.
"""
import json
import unittest
from pathlib import Path

import numpy as np

from physics import Config, acceleration, simulate, quadratic_baseline

SELF = Path(__file__).with_name("selftest_case.json")
EPS = 1e-12


class SelfTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S = json.loads(SELF.read_text(encoding="utf-8"))
        inp = cls.S["input"]
        cls.r0 = np.asarray(inp["r0"], float)
        cls.r1 = np.asarray(inp["r_0.05"], float)
        cls.r2 = np.asarray(inp["r_0.10"], float)
        cls.dt = float(inp["dt_s"])
        cls.v = (cls.r2 - cls.r0) / (2 * cls.dt)
        cls.a = (cls.r2 - 2 * cls.r1 + cls.r0) / cls.dt**2
        cls.g = np.array([0.0, 0.0, -9.81])
        cls.aero = cls.a - cls.g
        cls.vhat = cls.v / np.linalg.norm(cls.v)
        cls.a_par = float(cls.aero @ cls.vhat)
        cls.a_perp = cls.aero - cls.a_par * cls.vhat
        cls.rho = 1.20
        cls.mass = 0.430
        cls.radius = 0.110
        cls.area = np.pi * cls.radius**2
        cls.k = cls.rho * cls.area / (2 * cls.mass)

    # -- 1. analytic kinematics (should hold to machine precision) ----------
    def test_kinematics(self):
        S = self.S["kinematics"]
        np.testing.assert_allclose(self.v, S["v_at_0.05_m_s"], atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(self.v)), S["speed_m_s"], places=3)
        np.testing.assert_allclose(self.v, [29.6, 8.6, 6.3], atol=1e-9)
        np.testing.assert_allclose(self.a, S["a_bar_m_s2"], atol=1e-9)
        np.testing.assert_allclose(self.aero, S["a_aero_m_s2"], atol=1e-2)
        self.assertAlmostEqual(self.a_par, S["a_parallel_m_s2"], places=3)
        np.testing.assert_allclose(self.a_perp, S["a_perp_m_s2"], atol=5e-4)
        self.assertAlmostEqual(float(np.linalg.norm(self.a_perp)),
                               S["a_perp_norm_m_s2"], places=3)
        np.testing.assert_allclose(self.v - self.a * self.dt,
                                   S["v0_m_s"], atol=1e-9)

    # -- 2. inverted apparent coefficients at rho = 1.20 --------------------
    def test_inverted_coefficients(self):
        ref = self.S["inverted_coefficients_by_density"]["rho_1.2"]
        self.assertAlmostEqual(self.k, ref["k_rhoA_2m"], places=5)
        cd_hat = abs(self.a_par) / (self.k * np.linalg.norm(self.v) ** 2)
        cl_hat = np.linalg.norm(self.a_perp) / (self.k * np.linalg.norm(self.v) ** 2)
        self.assertAlmostEqual(cd_hat, ref["Cd_hat"], places=3)
        self.assertAlmostEqual(cl_hat, ref["CL_hat"], places=3)

    # -- 3. constant-acceleration baseline (paper Baseline 1) ---------------
    def test_constant_accel_baseline(self):
        ref = self.S["constant_accel_baseline"]
        base = quadratic_baseline(np.stack([self.r1, self.r2]))
        self.assertAlmostEqual(base["time"], ref["T_s_full_precision"], places=5)
        self.assertAlmostEqual(base["endpoint"][0], ref["Y_m_full_precision"], places=4)
        self.assertAlmostEqual(base["endpoint"][1], ref["Z_m_full_precision"], places=4)
        # Team decision v2: report Z = 1.920 m, never 1.921.
        self.assertEqual(ref["Z_m"], 1.92)

    # -- 4. noise amplification: sqrt(5) for exact r(0), sqrt(6) upper bound -
    def test_noise_amplification(self):
        analytic = self.S["noise_propagation"]["analytic"]
        for sigma, key in ((.002, "2mm"), (.005, "5mm"), (.015, "15mm")):
            expected = np.sqrt(5) * sigma / self.dt**2
            self.assertAlmostEqual(analytic[key]["sd_a_r0exact"], expected, places=4)
        sd5 = analytic["5mm"]["sd_a_r0exact"]
        self.assertAlmostEqual(sd5, 4.4721, places=4)
        self.assertAlmostEqual(analytic["5mm"]["sd_a_r0noisy"], np.sqrt(6) * .005 / self.dt**2, places=4)

    # -- 5. end-to-end integration: physical Magnus direction ---------------
    def test_integration_reproduces_case_observations(self):
        """Constant (CD, CL) with the inferred spin axis must reproduce the
        posed observations (5 mm / 10 mm tolerances from A's document)."""
        S = self.S
        cd_hat = S["inverted_coefficients_by_density"]["rho_1.2"]["Cd_hat"]
        cl_hat = S["inverted_coefficients_by_density"]["rho_1.2"]["CL_hat"]
        axis = np.asarray(S["derived"]["spin_axis_unit"], float)
        axis /= np.linalg.norm(axis)
        # |omega| is irrelevant when fixed_cl is set: only the spin axis (lift
        # direction) matters. Pick a value that reproduces CL = 1.4 S at t=0.05.
        omega = axis * cl_hat * np.linalg.norm(self.v) / (1.4 * self.radius)
        theta = np.r_[S["kinematics"]["v0_m_s"], omega, cd_hat]
        sim = simulate(theta, Config(dt=1e-3, fixed_cd=cd_hat, fixed_cl=cl_hat))
        obs = sim["observations"][0]
        # Physical constant-coefficient reference (RK4, h = 1e-5): with
        # k*CL*|v|^2 * unit(omega x v) the case points are reproduced to ~mm.
        reference = np.array([[1.4993, 0.4489, 0.4397], [2.9587, 0.8578, 0.7394]])
        self.assertLess(np.linalg.norm(obs[0] - self.r1), .005)
        self.assertLess(np.linalg.norm(obs[1] - self.r2), .010)
        self.assertLess(np.linalg.norm(obs - reference, axis=1).max(), 1e-3)

    # -- 6. lift direction convention used by the simulator -----------------
    def test_lift_direction_is_transverse_cross_product(self):
        S = self.S
        axis = np.asarray(S["derived"]["spin_axis_unit"], float)
        expected = np.cross(axis, self.v)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(self.a_perp / np.linalg.norm(self.a_perp),
                                   expected, atol=1e-3)

    # -- 7. zero cross product must give zero lift (A's request) -----------
    def test_zero_cross_product_gives_no_lift(self):
        """omega parallel to v (or zero spin) -> no lift, no division by zero.

        A approved ``unit(w_hat x v_hat)`` with an explicit zero-cross case:
        when the cross product vanishes this test requires the lift to be 0.
        A near-parallel spin (numerically ~1e-12 relative) must behave the same.
        """
        k = self.rho * np.pi * self.radius**2 / (2 * self.mass)
        speed = np.linalg.norm(self.v)
        expected = np.array([0.0, 0.0, -9.81]) - k * 0.3 * speed * self.v
        cfg = Config(fixed_cl=0.2105, density=self.rho)
        for omega in (np.zeros(3),
                      self.vhat * 40.0,          # spin axis parallel to velocity
                      -self.vhat * 40.0,         # and antiparallel
                      self.vhat * 40.0 + np.array([1e-10, 0.0, 0.0])):  # near-parallel
            theta = np.r_[self.v, omega, .3][None, :]
            acc = acceleration(self.v[None, :], theta, 0.0, cfg)[0]
            self.assertTrue(np.all(np.isfinite(acc)))
            np.testing.assert_allclose(acc, expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
