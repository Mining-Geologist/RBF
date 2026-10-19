"""Synthetic tests for wall polarity against oriented reference triangles."""
import unittest
import numpy as np
import trimesh
from reference_contact_orientation_audit import compare


class TestReferenceNormalSign(unittest.TestCase):
    def test_horizontal_reference_polarity(self):
        v=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]],float)
        m=trimesh.Trimesh(vertices=v,faces=[[0,1,2],[0,2,3]],process=False)
        hw=np.array([[0,0,-1],[1,1,-1]],float)
        fw=np.array([[0,0,1],[1,1,1]],float)
        self.assertEqual(compare(m,hw,fw)["n_correct"],2)
        self.assertEqual(compare(m,fw,hw)["n_correct"],0)
        self.assertEqual(compare(m,fw,hw,normal_sign=1)["n_correct"],2)

    def test_reverse_triangle_winding_changes_polarity(self):
        v=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]],float)
        m=trimesh.Trimesh(vertices=v,faces=[[0,2,1],[0,3,2]],process=False)
        hw=np.array([[0,0,1]],float)
        fw=np.array([[0,0,-1]],float)
        self.assertEqual(compare(m,hw,fw)["n_correct"],1)


if __name__=="__main__":
    unittest.main()
