"""Independent reference-normal offset RBF, not a recovered vendor algorithm.

Given any triangulated reference and observed 3D wall points, fit signed
normal offsets in a 2D reference chart and displace every reference vertex
along its own smoothed normal. No private reference meshes or datasets.
Dependencies: numpy, scipy, trimesh.
"""
from dataclasses import dataclass
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator


@dataclass
class ReferenceFrame:
    vertices: np.ndarray
    faces: np.ndarray
    origin: np.ndarray
    basis: np.ndarray
    smooth_normals: np.ndarray
    _mesh: trimesh.Trimesh
    _tree: cKDTree

    @classmethod
    def from_mesh(cls, vertices, faces):
        v=np.asarray(vertices, dtype=float)
        f=np.asarray(faces, dtype=int)
        if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3:
            raise ValueError("Expected (N,3) vertices and (M,3) faces")
        if len(v)<3 or len(f)<1:
            raise ValueError("Reference must contain vertices and triangles")
        origin=v.mean(axis=0)
        _,eig=np.linalg.eigh((v-origin).T@(v-origin))
        normal=eig[:,0]
        sub=f[:min(256,len(f))]
        orientation=np.cross(v[sub[:,1]]-v[sub[:,0]],v[sub[:,2]]-v[sub[:,0]])
        if np.mean(orientation@normal)<0:
            normal=-normal
        basis=np.column_stack((eig[:,2],eig[:,1],normal))
        mesh=trimesh.Trimesh(vertices=v,faces=f,process=False)
        smooth=mesh.vertex_normals.copy()
        smooth[smooth@normal<0]*=-1
        return cls(v,f,origin,basis,smooth,mesh,cKDTree(v))

    def closest_normal_anchors(self, positions, nearest_vertices=12):
        """Approximate nearest-triangle projection and interpolated local normals.

        For strongly elongated triangles, compare with nearest_vertices=32/64,
        or replace candidate search with a complete spatial triangle index.
        """
        points=np.asarray(positions, dtype=float).reshape(-1,3)
        _,ids=self._tree.query(points,k=min(len(self.vertices),nearest_vertices))
        if ids.ndim==1:ids=ids[:,None]
        anchors=np.empty_like(points)
        normals=np.empty_like(points)
        for i,(point,v_ids) in enumerate(zip(points,ids)):
            candidates=np.unique(self._mesh.vertex_faces[v_ids].ravel())
            candidates=candidates[candidates>=0]
            if len(candidates)==0:
                raise ValueError("Isolated reference vertex")
            tris=self.vertices[self.faces[candidates]]
            nearest=trimesh.triangles.closest_point(tris,np.broadcast_to(point,(len(tris),3)))
            idx=int(np.argmin(np.sum((nearest-point)**2,axis=1)))
            face=self.faces[candidates[idx]]
            q=nearest[idx]
            tri=self.vertices[face]
            a=tri[1]-tri[0]
            b=tri[2]-tri[0]
            c=q-tri[0]
            aa=a@a;ab=a@b;bb=b@b;ac=a@c;bc=b@c
            den=aa*bb-ab*ab
            if abs(den)<1e-14:raise ValueError("Degenerate reference triangle")
            u=(ac*bb-bc*ab)/den
            v=(bc*aa-ac*ab)/den
            weights=np.array([1-u-v,u,v])
            normal=weights@self.smooth_normals[face]
            normal/=np.linalg.norm(normal)
            if normal@self.basis[:,2]<0:normal=-normal
            anchors[i]=q
            normals[i]=normal
        xy=(anchors-self.origin)@self.basis[:,:2]
        return anchors,normals,xy

    def fit_wall(self, contacts, kernel="linear", degree=0, epsilon=1.0, v_scale=1.0):
        pts=np.asarray(contacts, dtype=float).reshape(-1,3)
        q,n,xy=self.closest_normal_anchors(pts)
        displacement=np.einsum("ij,ij->i",pts-q,n)
        xy[:,1]*=v_scale
        model=RBFInterpolator(xy,displacement,kernel=kernel,
                              degree=degree,epsilon=epsilon,smoothing=0)
        return ReferenceWall(self,model,v_scale)


@dataclass
class ReferenceWall:
    reference: ReferenceFrame
    rbf: RBFInterpolator
    v_scale: float

    def predicted_mesh(self):
        r=self.reference
        xy=(r.vertices-r.origin)@r.basis[:,:2]
        xy[:,1]*=self.v_scale
        displacement=self.rbf(xy)
        return r.vertices+r.smooth_normals*displacement[:,None],r.faces.copy()

    def predict_at_reference_points(self, positions):
        q,n,xy=self.reference.closest_normal_anchors(positions)
        xy[:,1]*=self.v_scale
        displacement=self.rbf(xy)
        return q+n*displacement[:,None]
