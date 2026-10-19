"""Independent symmetric thickness correction regression tests."""
import unittest
import numpy as np
from native_wall_thickness_clamp import clamp_signed_wall_thickness


class TestWallThickness(unittest.TestCase):
    def test_maximum_reduces_both(self):
        h,f=clamp_signed_wall_thickness(106.,94.,maximum=10.)
        self.assertAlmostEqual(float(h),105.)
        self.assertAlmostEqual(float(f),95.)

    def test_minimum_expands_both(self):
        h,f=clamp_signed_wall_thickness(103.,97.,minimum=8.)
        self.assertAlmostEqual(float(h),104.)
        self.assertAlmostEqual(float(f),96.)

    def test_preserves_midpoint_and_clips_thickness(self):
        h=np.array([100.,104.,106.,115.])
        f=np.array([100.,100.,94.,96.])
        hh,ff=clamp_signed_wall_thickness(h,f,minimum=5.,maximum=10.)
        np.testing.assert_allclose(hh-ff,[5.,5.,10.,10.])
        np.testing.assert_allclose((hh+ff)/2,(h+f)/2)
        np.testing.assert_allclose(hh-h,-(ff-f))

    def test_no_constraints(self):
        h,f=clamp_signed_wall_thickness([104,100],[96,101])
        np.testing.assert_allclose(h,[104,100])
        np.testing.assert_allclose(f,[96,101])

    def test_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            clamp_signed_wall_thickness(1.,0.,minimum=10.,maximum=5.)


if __name__=="__main__":
    unittest.main()
