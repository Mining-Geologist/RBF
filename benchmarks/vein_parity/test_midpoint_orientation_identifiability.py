"""Synthetic tests; no Leapfrog or user data needed."""
import unittest
import numpy as np
import trimesh
from midpoint_orientation_identifiability import symmetric_points, loocv, nearest_mesh_distances


class IdentifiabilityTest(unittest.TestCase):
    def test_symmetric_points(self):
        p = np.array([[0,-1,1],[0,1,1],[2,-1,3],[2,1,3]], float)
        self.assertTrue(symmetric_points(p))
        p[0,2] = 2
        self.assertFalse(symmetric_points(p))

    def test_mirror_equivalent_distance(self):
        v = np.array([[-2,-2,-.5],[2,-2,-.5],[-2,2,.5],[2,2,.5]],float)
        mesh = trimesh.Trimesh(vertices=v,faces=[[0,1,2],[1,3,2]],process=False)
        mirror = trimesh.Trimesh(vertices=v*[1,-1,1],faces=mesh.faces,process=False)
        p = np.array([[0,-1,1],[0,1,1],[1,-1,-.5],[1,1,-.5]],float)
        a = nearest_mesh_distances(mesh,p)
        b = nearest_mesh_distances(mirror,p)
        self.assertAlmostEqual(np.linalg.norm(a),np.linalg.norm(b),places=11)

    def test_symmetric_midpoint_cv_mirror(self):
        p = np.array([[x,y,3+(.5*x*x if x else 0)+z]
                      for y in [-2.,0.,2.] for x,z in [(-3.,0.),(0.,1.),(3.,0.)]],float)
        n = np.array([.35,.65,.7]);n /= np.linalg.norm(n)
        nf = n * np.array([1,-1,1])
        for smoothing in [.01,1,20]:
            self.assertAlmostEqual(loocv(p,n,smoothing),
                                   loocv(p,nf,smoothing),places=9)


if __name__ == '__main__':
    unittest.main()
