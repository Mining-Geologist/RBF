"""Independent 2D chart-reference kernel audit (NOT Leapfrog intrusion code).
Requires numpy, scipy, trimesh. Native oblique chart normal, when supplied
from the output, is POST-HOC information rather than an input-only prediction.
Example: python fold_midpoint_linear_drift_audit.py --midpoints data.csv
  --reference reference.obj --normal '-.35457,-.759834,-.544915'
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator
import trimesh

def audit(midpoints,vertices,faces,normal):
    p=np.asarray(midpoints,float);v=np.asarray(vertices,float)
    n=np.asarray(normal,float);n=n/np.linalg.norm(n)
    axis=np.array([0.,0.,1.])
    if abs(axis@n)>.99:axis=np.array([0.,1.,0.])
    u=np.cross(axis,n);u=u/np.linalg.norm(u);w=np.cross(n,u)
    frame=np.column_stack([u,w,n]);origin=p.mean(axis=0)
    c=(p-origin)@frame;t=(v-origin)@frame
    surface=trimesh.Trimesh(vertices=v,faces=np.asarray(faces,int),process=False)
    _,dist,_=trimesh.proximity.closest_point_naive(surface,p)
    def pca(x):return np.linalg.eigh(np.cov(x.T))[1][:,0]
    angle=float(np.degrees(np.arccos(np.clip(abs(pca(p)@pca(v)),0,1))))
    refl=p.copy();refl[:,1]*=-1
    sym=sorted(map(tuple,np.round(p,6)))==sorted(map(tuple,np.round(refl,6)))
    result={"n_controls":len(p),"n_reference_vertices":len(v),
      "midpoint_native_surface_rmse_m":float(np.sqrt(np.mean(dist**2))),
      "pca_normal_angle_deg":angle,"Y_reflection_symmetry":sym,
      "scores":{}}
    for kernel,degree in [("linear",0),("linear",1),("multiquadric",0),("thin_plate_spline",1)]:
        for smoothing in [0.,1.,20.]:
            key=f"{kernel}_degree{degree}_smoothing{smoothing:g}"
            try:
                f=RBFInterpolator(c[:,:2],c[:,2],kernel=kernel,
                    degree=degree,smoothing=smoothing,epsilon=1.)
                err=f(t[:,:2])-t[:,2]
                result["scores"][key]=float(np.sqrt(np.mean(err**2)))
            except Exception as exc:result["scores"][key]={"error":str(exc)}
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midpoints",required=True)
    parser.add_argument("--reference",required=True)
    parser.add_argument("--normal",required=True,help="x,y,z; chart normal, potentially output-derived")
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    cloud=np.loadtxt(args.midpoints,delimiter=",",skiprows=1)
    mesh=trimesh.load_mesh(args.reference,process=False)
    result=audit(cloud,mesh.vertices,mesh.faces,
                 [float(s) for s in args.normal.split(",")])
    output=json.dumps(result,indent=2)
    if args.output:Path(args.output).write_text(output)
    print(output)
