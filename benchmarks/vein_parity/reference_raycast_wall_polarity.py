"""Reference-triangle contact-polarity prediction using drillhole-aware ray hits.

For a vertical borehole midpoint (x,y,z):
(1) intersect the vertical line (x,y,*) with all reference triangles;
(2) use the closest Z crossing and infer polarity from its wound normal;
(3) if no vertical crossing exists, use the nearest Euclidean triangle.

A behavioral surrogate for native saved reference contact assignments;
NOT a recovered Leapfrog proprietary projection implementation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh


def vertical_ray_nearest_triangles(mesh, points, eps=1e-9):
    xyz=np.asarray(points,float).reshape(-1,3)
    t=mesh.triangles
    a=t[:,0,:2]
    uvec=t[:,1,:2]-a
    vvec=t[:,2,:2]-a
    den=uvec[:,0]*vvec[:,1]-uvec[:,1]*vvec[:,0]
    valid=np.abs(den)>eps
    selected=[]
    for p in xyz:
        ap=p[:2]-a
        u=np.full(len(t),np.nan)
        v=np.full(len(t),np.nan)
        u[valid]=(ap[valid,0]*vvec[valid,1]-ap[valid,1]*vvec[valid,0])/den[valid]
        v[valid]=(uvec[valid,0]*ap[valid,1]-uvec[valid,1]*ap[valid,0])/den[valid]
        contains=valid&(u>=-eps)&(v>=-eps)&(u+v<=1+eps)
        ids=np.where(contains)[0]
        if not len(ids):
            selected.append(None)
            continue
        z=t[ids,0,2]+u[ids]*(t[ids,1,2]-t[ids,0,2])+v[ids]*(t[ids,2,2]-t[ids,0,2])
        selected.append(int(ids[np.argmin(np.abs(z-p[2]))]))
    return selected


def compare_ray_first(mesh, hw, fw, sign=-1):
    hw=np.asarray(hw,float).reshape(-1,3)
    fw=np.asarray(fw,float).reshape(-1,3)
    if hw.shape!=fw.shape: raise ValueError("HW and FW shape mismatch")
    mids=(hw+fw)/2
    _,euclidean_distance,nearest=trimesh.proximity.closest_point_naive(mesh,mids)
    hits=vertical_ray_nearest_triangles(mesh,mids)
    chosen=np.array([int(j) if j is not None else int(nearest[i]) for i,j in enumerate(hits)])
    normal_polarity=sign*np.sign(mesh.face_normals[chosen,2])
    observed=np.sign(hw[:,2]-fw[:,2])
    return {
        "n":len(hw),
        "matched":int(np.sum(normal_polarity==observed)),
        "n_with_ray_intersection":sum(j is not None for j in hits),
        "errors":list(np.where(normal_polarity!=observed)[0].astype(int)),
        "rows":[
            {"row":i,"ray_intersection":hits[i] is not None,
             "reference_distance_m":float(euclidean_distance[i]),
             "observed_polarity":int(observed[i]),
             "predicted_polarity":int(normal_polarity[i])}
            for i in range(len(mids))
        ]
    }


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",required=True)
    p.add_argument("--hangingwall",required=True)
    p.add_argument("--footwall",required=True)
    p.add_argument("--output")
    args=p.parse_args()
    model=trimesh.load_mesh(args.reference,process=False)
    hw=np.loadtxt(args.hangingwall,delimiter=",",skiprows=1)
    fw=np.loadtxt(args.footwall,delimiter=",",skiprows=1)
    result=json.dumps(compare_ray_first(model,hw,fw),indent=2)
    if args.output:Path(args.output).write_text(result)
    print(result)
