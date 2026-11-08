import unittest
import numpy as np
from physics import Config, simulate, acceleration, quadratic_baseline, CASE


class PhysicsTests(unittest.TestCase):
    def test_ballistic_and_initial_ground(self):
        theta=np.array([[30.,3.,6.,0.,0.,0.,0.]])
        out=simulate(theta,Config(density=0,dt=.001))
        t=11/30
        self.assertEqual(out['status'][0],1)
        np.testing.assert_allclose(out['observations'][0,0], [1.5,.15,.11+.3-.5*9.81*.05**2],atol=1e-10)
        np.testing.assert_allclose(out['endpoint'][0],[3*t,.11+6*t-.5*9.81*t*t],atol=2e-6)

    def test_axial_and_zero_spin(self):
        v=np.array([[30.,0.,0.],[30.,0.,0.]])
        theta=np.array([[30,0,0,60,0,0,.2],[30,0,0,0,0,0,.2]],float)
        a=acceleration(v,theta,0,Config())
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_allclose(a[0],a[1])

    def test_drag_dissipation_and_lift_perpendicular(self):
        v=np.array([[25.,4.,6.]])
        th=np.array([[25.,4.,6.,20.,15.,30.,.2]])
        a=acceleration(v,th,0,Config(gravity=0))
        no=th.copy(); no[:,3:6]=0
        drag=acceleration(v,no,0,Config(gravity=0))
        self.assertLess(float(np.sum(drag*v)),0)
        self.assertAlmostEqual(float(np.sum((a-drag)*v)),0,places=10)

    def test_ground_before_goal(self):
        out=simulate([[20,0,.5,0,0,0,.2]])
        self.assertEqual(out['status'][0],2)
        self.assertTrue(np.isnan(out['endpoint']).all())

    def test_refinement(self):
        th=[[30,8,7,15,20,-30,.3]]
        a=simulate(th,Config(dt=.002)); b=simulate(th,Config(dt=.001))
        np.testing.assert_allclose(a['endpoint'],b['endpoint'],atol=1e-4)

    def test_case_baseline(self):
        b=quadratic_baseline(CASE)
        np.testing.assert_allclose(b['endpoint'],[2.495,1.921],atol=.001)


if __name__=='__main__': unittest.main()
