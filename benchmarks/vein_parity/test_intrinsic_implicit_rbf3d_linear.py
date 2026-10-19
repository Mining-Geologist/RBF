"""Synthetic tests for an independent 3D implicit linear+affine RBF.
No owner datasets, files or vendor classes required.
"""
import unittest
import numpy as np
from intrinsic_implicit_rbf3d_linear import (
    evaluate_biharmonic_3d_linear, orientation_constraints, implicit_surface_proxy)


class TestImplicitLinearRBF(unittest.TestCase):
    def test_saved_coefficient_formula(self):
        centres=np.array([[0.,0.,0.],[1.,0.,0.]])
        coeffs=np.array([.8,-.4,2.1,.3,-.2,.5])
        points=np.array([[0.,0.,0.],[1.,1.,1.]])
        ref=(np.linalg.norm(points[:,None]-centres[None],axis=2)@coeffs[:2]
             +2.1+points@coeffs[3:])
        np.testing.assert_allclose(
            evaluate_biharmonic_3d_linear(points,centres,coeffs,[0,0,0]),ref)

    def test_orientation_anchors(self):
        points,values=orientation_constraints([1,2,3],[2,0,0],4)
        np.testing.assert_allclose(points,[[5,2,3],[-3,2,3]])
        np.testing.assert_allclose(values,[1.75,-1.75])

    def test_observed_and_signed_controls(self):
        p=np.array([[-4,-3,0],[0,-3,1],[4,-3,0],[-4,3,0],[0,3,1],[4,3,0]],float)
        r,c=implicit_surface_proxy(p,[0,0,0],[1,-1,1],8)
        np.testing.assert_allclose(r(p),0,atol=1e-8)
        self.assertAlmostEqual(float(r(c[-2:-1])[0]),1.75,places=8)
        self.assertAlmostEqual(float(r(c[-1:])[0]),-1.75,places=8)

    def test_orientation_controls_change_field(self):
        p=np.array([[-4,-3,0],[0,-3,1],[4,-3,0],[-4,3,0],[0,3,1],[4,3,0]],float)
        a,_=implicit_surface_proxy(p,[0,0,0],[1,0,0],6)
        b,_=implicit_surface_proxy(p,[0,0,0],[0,1,0],6)
        self.assertGreater(float(abs(a([[2,1,3]])-b([[2,1,3]])).max()),1e-3)


if __name__=="__main__":
    unittest.main()
