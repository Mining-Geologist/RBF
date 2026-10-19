"""Standalone synthetic validation: python -m unittest -v test_reference_following_surface.py"""
import unittest
import numpy as np
from reference_following_surface import ReferenceFrame


def curved_fixture(nx=15):
    x=np.linspace(-20,20,nx)
    xx,yy=np.meshgrid(x,x)
    zz=3*np.sin(xx/15)+2*np.cos(yy/13)
    vertices=np.column_stack((xx.ravel(),yy.ravel(),zz.ravel()))
    triangles=[]
    for y in range(nx-1):
        for x in range(nx-1):
            a=y*nx+x
            b=a+1
            c=a+nx
            d=c+1
            triangles.extend(((a,b,d),(a,d,c)))
    return vertices,np.asarray(triangles,dtype=int)


class TestReferenceNormalSurface(unittest.TestCase):
    def test_nearly_constant_normal_offset_preserves_topology(self):
        v,f=curved_fixture()
        r=ReferenceFrame.from_mesh(v,f)
        inds=np.array([70,74,85,91,102,110,142,159,180])
        controls=v[inds]+r.smooth_normals[inds]*2.5
        predicted,faces=r.fit_wall(controls).predicted_mesh()
        np.testing.assert_array_equal(faces,f)
        np.testing.assert_allclose(predicted,v+2.5*r.smooth_normals,atol=.001,rtol=0)

    def test_interpolation_at_observed_normal_offset_points(self):
        v,f=curved_fixture()
        r=ReferenceFrame.from_mesh(v,f)
        inds=np.array([74,88,99,110,122,137,163,177,189])
        pts=v[inds]+r.smooth_normals[inds]*np.linspace(.2,.9,len(inds))[:,None]
        q,n,_=r.closest_normal_anchors(pts)
        predicted=r.fit_wall(pts).predict_at_reference_points(pts)
        expected=q+n*np.einsum("ij,ij->i",pts-q,n)[:,None]
        np.testing.assert_allclose(predicted,expected,atol=1e-7)

    def test_original_reference_unchanged_by_wall_fitting(self):
        v,f=curved_fixture()
        r=ReferenceFrame.from_mesh(v,f)
        ids=[70,84,91,105,115,130,143,170]
        r.fit_wall(v[ids]+2*r.smooth_normals[ids]).predicted_mesh()
        np.testing.assert_array_equal(r.vertices,v)
        np.testing.assert_array_equal(r.faces,f)


if __name__=="__main__":
    unittest.main()
