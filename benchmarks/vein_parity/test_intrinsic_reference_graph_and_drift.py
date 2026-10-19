"""Synthetic regression tests; no native vendor/project data required."""
import unittest
import numpy as np
from scipy.sparse.csgraph import dijkstra
from intrinsic_reference_graph_rbf import (
    mesh_edge_graph, anchored_source_distances,
    fit_linear_distance_rbf,evaluate_distance_rbf)
from fold_midpoint_linear_drift_audit import audit

class TestIndependentVeinResearch(unittest.TestCase):
    def setUp(self):
        xs=np.linspace(-1,1,7);xx,yy=np.meshgrid(xs,xs)
        self.v=np.column_stack((xx.ravel(),yy.ravel(),(.1*xx**2+.2*yy**2).ravel()))
        f=[]
        for row in range(6):
            for col in range(6):
                a=row*7+col;f.extend(((a,a+1,a+7),(a+1,a+8,a+7)))
        self.f=np.asarray(f,int)
    def test_graph_shortest_distance_is_symmetric(self):
        g=mesh_edge_graph(self.v,self.f)
        dd=dijkstra(g,indices=[0,48])
        self.assertTrue(np.isfinite(dd).all())
        self.assertAlmostEqual(dd[0,48],dd[1,0],places=8)
    def test_face_anchored_graph_distances(self):
        a=np.array([self.v[0]*.4+self.v[1]*.3+self.v[7]*.3,
                    self.v[-1]*.4+self.v[-2]*.3+self.v[-8]*.3])
        faceids=np.array([0,len(self.f)-1])
        dv,dc=anchored_source_distances(mesh_edge_graph(self.v,self.f),
                                          self.v,self.f,a,faceids)
        self.assertEqual(dv.shape,(2,49))
        np.testing.assert_allclose(dc,dc.T,atol=1e-12)
        np.testing.assert_allclose(np.diag(dc),0,atol=1e-12)
    def test_linear_distance_constant_drift(self):
        ids=np.array([0,6,24,42,48])
        d=dijkstra(mesh_edge_graph(self.v,self.f),indices=ids)
        weights,trend=fit_linear_distance_rbf(d[:,ids],np.ones(5)*4)
        np.testing.assert_allclose(evaluate_distance_rbf(d,weights,trend),4,atol=1e-5)
    def test_linear_distance_affine_drift(self):
        ids=np.array([0,6,24,42,48])
        d=dijkstra(mesh_edge_graph(self.v,self.f),indices=ids)
        xy=self.v[:,:2];target=1+.3*xy[ids,0]-.9*xy[ids,1]
        weights,trend=fit_linear_distance_rbf(d[:,ids],target,
                                      chart=xy[ids],drift="linear")
        np.testing.assert_allclose(evaluate_distance_rbf(d,weights,trend,xy),
                          1+.3*xy[:,0]-.9*xy[:,1],atol=1e-5)
    def test_symmetric_cloud_does_not_require_symmetric_ref(self):
        p=np.array([[-1,-1,0],[1,-1,0],[-1,1,0],[1,1,0],
                    [0,-1,1],[0,1,1],[0,-1,-1],[0,1,-1]],float)
        m=np.array([[-2,-2,0],[2,-2,0],[-2,2,0],[2,2,0],[0,0,0]],float)
        f=np.array([[0,1,4],[1,3,4],[3,2,4],[2,0,4]])
        result=audit(p,m,f,[.5,.5,.7])
        self.assertTrue(result["Y_reflection_symmetry"])
        self.assertGreater(result["midpoint_native_surface_rmse_m"],0)
        self.assertIn("linear_degree1_smoothing0",result["scores"])

if __name__=="__main__":unittest.main()
