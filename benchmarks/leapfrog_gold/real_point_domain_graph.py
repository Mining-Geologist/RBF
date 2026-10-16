"""Experimental offline clustering from real point XYZ and structural SPD tensors.

No oracle/reference labels, Leapfrog runtime, or installed binaries are used.
"""
from dataclasses import dataclass
import heapq
import numpy as np
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class Configuration:
    neighbours: int = 12
    micro_size: int = 4
    max_fraction: float = .10
    structural_weight: float = 1.0
    distance_weight: float = .0
    affinity_cutoff: float = .60

def _consistency(a, sa, b, sb):
    m = (sa*a + sb*b) / (sa+sb)
    d = np.linalg.det(m)
    return float(1.0/d) if d > 0 and np.isfinite(d) else -1.

def _make_edges(x, n_neighbours):
    dist, ix = cKDTree(x).query(x, k=min(len(x), n_neighbours+1))
    a = np.repeat(np.arange(len(x)), ix.shape[1]-1)
    b = ix[:,1:].ravel().astype(int)
    valid = a != b
    a, b, d = a[valid], b[valid], dist[:,1:].ravel()[valid]
    u, v = np.minimum(a,b), np.maximum(a,b)
    order = np.lexsort((v,u))
    u, v, d = u[order], v[order], d[order]
    key = u*len(x)+v
    keep = np.r_[True, key[1:] != key[:-1]]
    return np.stack([u[keep],v[keep]],axis=1), d[keep]

def cluster_real_points(points, matrices, cfg: Configuration):
    points = np.asarray(points,float)
    matrices = np.asarray(matrices,float)
    n = len(points)
    if points.shape != (n,3) or matrices.shape != (n,3,3):
        raise ValueError("Expected points N x 3 and matrices N x 3 x 3")
    if n < 2 or cfg.neighbours < 1 or cfg.micro_size < 1 or not 0 < cfg.max_fraction <= 1:
        raise ValueError("Invalid configuration or too few points")
    edges, dist = _make_edges(points, cfg.neighbours)
    median_nearest = np.median(cKDTree(points).query(points,k=2)[0][:,1])
    scale = max(float(median_nearest),1e-12)
    det = np.linalg.det(matrices)
    if np.any(det <= 0): raise ValueError("Expected SPD anisotropy tensors")
    tensors = matrices / det[:,None,None]**(1/3)
    parent = np.arange(2*n)
    size = np.zeros(2*n,dtype=int); size[:n] = 1
    mean = np.zeros((2*n,3,3)); mean[:n] = tensors
    active = np.zeros(2*n,dtype=bool); active[:n] = True
    adjacency = [{} for _ in range(2*n)]
    for (a,b),d in zip(edges,dist):
        a,b = int(a),int(b)
        adjacency[a][b] = float(d)
        adjacency[b][a] = float(d)
    heap = []
    next_id = n
    cap = cfg.micro_size
    def push(a,b):
        if not active[a] or not active[b] or a==b: return
        a,b = sorted((int(a),int(b)))
        if size[a]+size[b] > cap or b not in adjacency[a]: return
        consistency = _consistency(mean[a],size[a],mean[b],size[b])
        priority = cfg.structural_weight*consistency - cfg.distance_weight*np.log1p(adjacency[a][b]/scale)
        heapq.heappush(heap,(-priority,a,b))
    def merge(a,b):
        nonlocal next_id
        c=next_id;next_id+=1
        sa,sb=size[a],size[b];size[c]=sa+sb
        mean[c]=(sa*mean[a]+sb*mean[b])/(sa+sb)
        active[a]=active[b]=False
        parent[a]=parent[b]=c
        near=dict(adjacency[a])
        for k,v in adjacency[b].items():
            near[k]=min(near.get(k,float('inf')),v)
        near.pop(a,None);near.pop(b,None)
        for k,d in near.items():
            if not active[k]:continue
            adjacency[k].pop(a,None)
            adjacency[k].pop(b,None)
            adjacency[k][c]=d; adjacency[c][k]=d
        active[c]=True
        for k in adjacency[c]:push(c,k)
    def stage(maxsize):
        nonlocal cap
        cap=maxsize;heap.clear()
        for a in range(next_id):
            if active[a]:
                for b in adjacency[a]:
                    if a<b:push(a,b)
        count=0
        while heap:
            neg,a,b=heapq.heappop(heap)
            if not active[a] or not active[b] or b not in adjacency[a]:
                continue
            if size[a]+size[b]>cap:continue
            consistency=_consistency(mean[a],size[a],mean[b],size[b])
            if consistency < cfg.affinity_cutoff:continue
            merge(a,b);count+=1
        return count
    micro=stage(cfg.micro_size)
    macro=stage(max(cfg.micro_size,int(np.floor(cfg.max_fraction*n))))
    roots=np.arange(n)
    while True:
        new=parent[roots]
        if np.array_equal(new,roots):break
        roots=new
    _,labels=np.unique(roots,return_inverse=True)
    return labels,dict(micro_merges=micro,macro_merges=macro,
        domain_count=int(len(np.unique(labels))),largest=int(np.bincount(labels).max()),
        edges=len(edges))
