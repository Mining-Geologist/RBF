"""Independent triangular reference-mesh graph metric and linear-distance RBF.
No vendor implementations or privately supplied input geometry.
Mesh-graph distances approximate, but are NOT exact smooth-surface geodesics.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

def mesh_edge_graph(vertices,triangles):
    v=np.asarray(vertices,float);f=np.asarray(triangles,int)
    edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    d=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
    r=np.r_[edges[:,0],edges[:,1]]
    c=np.r_[edges[:,1],edges[:,0]]
    graph=coo_matrix((np.r_[d,d],(r,c)),shape=(len(v),len(v))).tocsr()
    # Shared edges may occur twice: restore physical edge length after summation.
    graph.sum_duplicates()
    ids=np.repeat(np.arange(len(v)),np.diff(graph.indptr))
    graph.data[:]=np.linalg.norm(v[ids]-v[graph.indices],axis=1)
    return graph

def anchored_source_distances(graph,vertices,triangles,anchors,face_ids):
    """Dijkstra sources at actual triangle-interior projected contact positions."""
    v=np.asarray(vertices,float);f=np.asarray(triangles,int)
    a=np.asarray(anchors,float);faces=np.asarray(face_ids,int)
    near=f[faces]
    used,reverse=np.unique(near,return_inverse=True)
    distance_from_mesh_vertices=dijkstra(graph,indices=used)
    offsets=np.linalg.norm(v[near]-a[:,None,:],axis=2)
    # Closest edge-network path entering at one of three source-face vertices.
    all_vertex_dist=np.min(
        distance_from_mesh_vertices[reverse.reshape(-1)].reshape(len(a),3,-1)
        +offsets[:,:,None],axis=1)
    cc=np.column_stack([np.min(all_vertex_dist[:,near[j]]+
                                offsets[j][None,:],axis=1)
                        for j in range(len(a))])
    same=faces[:,None]==faces[None,:]
    cc=np.where(same,np.linalg.norm(a[:,None,:]-a[None,:,:],axis=2),cc)
    cc=(cc+cc.T)/2
    np.fill_diagonal(cc,0)
    return all_vertex_dist,cc

def fit_linear_distance_rbf(pairwise_distances,values,chart=None,drift="constant",ridge=1e-8):
    """Fit k(d)=d; polynomial drift is constant or affine in *chart* coords."""
    d=np.asarray(pairwise_distances,float);y=np.asarray(values,float)
    n=len(y)
    if d.shape!=(n,n) or not np.isfinite(d).all():raise ValueError("invalid distances")
    if drift=="constant":p=np.ones((n,1))
    elif drift=="linear":
        if chart is None:raise ValueError("chart required for affine drift")
        p=np.column_stack([np.ones(n),np.asarray(chart,float)])
    else:raise ValueError("drift must be constant or linear")
    matrix=np.block([[d+ridge*np.eye(n),p],
                     [p.T,np.zeros((p.shape[1],p.shape[1]))]])
    answer=np.linalg.lstsq(matrix,np.r_[y,np.zeros(p.shape[1])],rcond=1e-11)[0]
    return answer[:n],answer[n:]

def evaluate_distance_rbf(source_to_vertices,weights,drift_coeff,chart_vertices=None):
    out=np.asarray(source_to_vertices).T@weights+drift_coeff[0]
    if len(drift_coeff)>1:
        if chart_vertices is None:raise ValueError("chart vertices required")
        out+=np.asarray(chart_vertices)@drift_coeff[1:]
    return out
