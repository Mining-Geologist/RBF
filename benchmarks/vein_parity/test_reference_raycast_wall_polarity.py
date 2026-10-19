"""Synthetic-only tests of ray intersection branch selection and fallback."""
import unittest
import numpy as np
import trimesh
from reference_raycast_wall_polarity import vertical_ray_nearest_triangles, compare_ray_first


class TestRaycastReferencePolarity(unittest.TestCase):
    def test_two_distinct_vertical_branch_crossings(self):
        v=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0],
                    [-2,-2,20],[2,-2,20],[2,2,20],[-2,2,20]],float)
        faces=np.array([[0,1,2],[0,2,3],
                        [4,6,5],[4,7,6]],int) # normals +Z / -Z
        mesh=trimesh.Trimesh(vertices=v,faces=faces,process=False)
        tri=vertical_ray_nearest_triangles(mesh,[[0,0,1],[0,0,19]])
        self.assertEqual(len(tri),2)
        self.assertGreater(mesh.face_normals[tri[0],2],0)
        self.assertLess(mesh.face_normals[tri[1],2],0)
        hw=np.array([[0,0,0],[0,0,20]],float)
        fw=np.array([[0,0,2],[0,0,18]],float)
        r=compare_ray_first(mesh,hw,fw)
        self.assertEqual(r["matched"],2)
        self.assertEqual(r["n_with_ray_intersection"],2)

    def test_no_vertical_hit_uses_euclidean_fallback(self):
        v=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]],float)
        mesh=trimesh.Trimesh(vertices=v,faces=[[0,1,2],[0,2,3]],process=False)
        self.assertIsNone(vertical_ray_nearest_triangles(mesh,[[9,9,5]])[0])
        hw=np.array([[9,9,-1]],float)
        fw=np.array([[9,9,1]],float)
        r=compare_ray_first(mesh,hw,fw)
        self.assertEqual(r["matched"],1)
        self.assertEqual(r["n_with_ray_intersection"],0)


if __name__=="__main__":unittest.main()
