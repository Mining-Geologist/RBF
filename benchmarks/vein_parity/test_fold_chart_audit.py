"""Data-free tests for the oblique chart detector."""
import unittest
import numpy as np
import trimesh
from fold_chart_audit import audit, midpoint_mirror_invariance


class FoldChartTests(unittest.TestCase):
    def test_planar_square(self):
        mesh = trimesh.Trimesh(vertices=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],
                               faces=[[0,1,2],[0,2,3]], process=False)
        out = audit(mesh)
        self.assertTrue(out["oblique_chart_valid"])
        self.assertEqual(out["xy_projection"]["overlapping_triangle_pairs"], 0)

    def test_closed_sphere_fails(self):
        out = audit(trimesh.creation.icosphere(subdivisions=1))
        self.assertFalse(out["hemisphere_exists"])

    def test_midpoint_reflection(self):
        pts = np.array([[0,-40,100],[0,40,100],[0,0,75],[0,0,115]], float)
        self.assertTrue(midpoint_mirror_invariance(pts))
        pts[1,2] = 101
        self.assertFalse(midpoint_mirror_invariance(pts))


if __name__ == "__main__":
    unittest.main()
