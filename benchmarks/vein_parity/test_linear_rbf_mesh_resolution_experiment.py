"""Synthetic field discretization tests, no native geometry included."""
import unittest
import numpy as np
from scipy.spatial import Delaunay
from linear_rbf_mesh_resolution_experiment import (
    predict_piecewise_linear,experiment,EXTRA)

class TestResolutionExperiment(unittest.TestCase):
    def test_planar_field_exact_on_triangles(self):
        nodes=np.array([[0,0],[1,0],[0,1],[1,1]],float)
        dt=Delaunay(nodes)
        values=5+2*nodes[:,0]-3*nodes[:,1]
        pts=np.array([[.1,.1],[.45,.8],[.8,.15]],float)
        np.testing.assert_allclose(predict_piecewise_linear(dt,values,pts),
                                   5+2*pts[:,0]-3*pts[:,1],atol=1e-13)
    def test_finer_grid_reduces_linear_rbf_discretization_error(self):
        a=experiment(10.)
        b=experiment(5.)
        self.assertGreater(a['rmse_m'],b['rmse_m'])
        self.assertEqual(a["n_samples"],len(EXTRA))
        self.assertEqual(b["n_samples"],len(EXTRA))

if __name__=="__main__": unittest.main()
