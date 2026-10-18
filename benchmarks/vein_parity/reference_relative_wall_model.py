"""Independent 2.5-D reconstruction of HW/FW as offsets from a reference mesh.

Research approximation only. A *saved* reference mesh is used as input; this
does NOT reproduce its construction or the separate native snapping field.
Do not use for overturned/folded multi-valued surfaces.

Requires numpy/scipy; no Leapfrog binary, license or native code.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial.distance import cdist
from scipy.linalg import solve


class ReferenceHeightField:
    def __init__(self, vertices, faces=None):
        xyz = np.asarray(vertices, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
            raise ValueError("reference mesh vertices must have shape (N,3)")
        if not np.isfinite(xyz).all():
            raise ValueError("reference vertices must be finite")
        # Fail clearly if a folded reference has two elevations at one XY.
        coord = np.round(xyz[:, :2], 7)
        _, index = np.unique(coord, axis=0, return_inverse=True)
        counts = np.bincount(index)
        zmax = np.full(len(counts), -np.inf)
        zmin = np.full(len(counts), np.inf)
        np.maximum.at(zmax, index, xyz[:, 2])
        np.minimum.at(zmin, index, xyz[:, 2])
        if np.any(zmax - zmin > 1e-4):
            raise ValueError("Reference is multivalued in XY; use a 3D manifold solver")
        if faces is not None:
            tri = np.asarray(faces, dtype=int)
            if tri.ndim != 2 or tri.shape[1] != 3:
                raise ValueError("Reference faces must be triangles (F,3)")
            a,b,c = xyz[tri[:,0],:2],xyz[tri[:,1],:2],xyz[tri[:,2],:2]
            orient = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0])
            if np.any(orient > 1e-7) and np.any(orient < -1e-7):
                raise ValueError("Reference has overturned or folded XY projection; use a true 3D mesh chart")
        self._interp = LinearNDInterpolator(xyz[:, :2], xyz[:, 2], fill_value=np.nan)
        self._nearest = NearestNDInterpolator(xyz[:, :2], xyz[:, 2])

    def height(self, xy):
        xy = np.asarray(xy, float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N,2)")
        z = np.asarray(self._interp(xy), float)
        miss = ~np.isfinite(z)
        if np.any(miss):
            # Explicit nearest-neighbour fallback OUTSIDE convex hull.
            z[miss] = self._nearest(xy[miss])
        return z


class AffineMultiquadric:
    """Exact MQ1 interpolant with affine polynomial drift (not a native claim)."""
    def __init__(self, xy, value, shape_m=1.):
        x = np.asarray(xy,float);y = np.asarray(value,float)
        if x.ndim != 2 or x.shape[1] != 2 or y.shape != (len(x),):
            raise ValueError("Expected xy (N,2), value (N,)")
        if len(x) < 3 or shape_m <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Insufficient/invalid controls")
        if np.linalg.matrix_rank(x-x.mean(axis=0)) != 2:
            raise ValueError("Controls are collinear")
        self.x = x;self.origin = x.mean(axis=0)
        self.scale = max(float(np.ptp(x, axis=0).max()),1.)
        self.shape_m = float(shape_m)
        p = self._poly(x)
        k = self._kernel(cdist(x,x))
        a = np.block([[k,p],[p.T,np.zeros((3,3))]])
        sol = solve(a,np.r_[y,np.zeros(3)],assume_a="sym")
        self.weights = sol[:len(x)]
        self.drift = sol[len(x):]

    def _poly(self, xy):
        q = (xy-self.origin)/self.scale
        return np.column_stack([np.ones(len(q)),q])

    def _kernel(self, distances):
        return np.sqrt(1.+(distances/self.shape_m)**2)

    def __call__(self, xy):
        q = np.asarray(xy,float)
        return self._kernel(cdist(q,self.x))@self.weights+self._poly(q)@self.drift


class ReferenceRelativeWalls:
    def __init__(self, reference_vertices, hw_contacts, fw_contacts, shape_m=1., reference_faces=None):
        self.reference = ReferenceHeightField(reference_vertices, reference_faces)
        self.raw = {}
        self.offsets = {}
        for wall, controls in (("HW",hw_contacts),("FW",fw_contacts)):
            xyz=np.asarray(controls,float)
            if xyz.ndim!=2 or xyz.shape[1]!=3:
                raise ValueError("HW/FW contact data must have shape (N,3)")
            self.raw[wall] = AffineMultiquadric(xyz[:,:2],xyz[:,2],shape_m)
            offset = xyz[:,2] - self.reference.height(xyz[:,:2])
            self.offsets[wall] = AffineMultiquadric(xyz[:,:2],offset,shape_m)

    def evaluate(self, xy, relative=True):
        xy=np.asarray(xy,float)
        return {
            wall: (self.reference.height(xy)+self.offsets[wall](xy)
                   if relative else self.raw[wall](xy))
            for wall in ("HW","FW")
        }

    def signed_thickness(self, xy):
        z=self.evaluate(xy)
        return z["HW"]-z["FW"]

    def is_inside(self, xyz):
        p=np.asarray(xyz,float)
        z=self.evaluate(p[:,:2])
        return (z["HW"]>z["FW"]) & (p[:,2]>=z["FW"]) & (p[:,2]<=z["HW"])
