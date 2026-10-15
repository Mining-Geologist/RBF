"""Synthetic structural input regression checks independent of native Jizai build."""
import json
import numpy as np
from jizai_structural_field import nearest_vertex_trend_matrices, indicator_distance_values

def main():
    vertices = np.array([[0.,0.,0.],[10.,0.,0.],[0.,10.,0.],[10.,10.,0.]])
    faces = np.array([[0,1,2],[1,3,2]], dtype=int)
    R = 30.0
    s = 8.0
    query = np.array([[0,0,0],[0,0,R],[0,0,4*R],[0,0,4.5*R]],dtype=float)
    mats = nearest_vertex_trend_matrices(query,vertices,faces,strength=s,range_m=R)
    ratios = [s, 1+(s-1)*np.exp(-1), 1, 1]
    expect = np.stack([np.diag([r**(-1/3),r**(-1/3),r**(2/3)]) for r in ratios])
    max_abs = float(np.max(np.abs(mats-expect)))
    dets = np.linalg.det(mats)
    eigs = np.linalg.eigvalsh(mats)
    assert max_abs < 1e-12, max_abs
    assert np.allclose(dets, 1, rtol=0,atol=1e-12), dets
    assert np.all(eigs>0), eigs

    pts=np.array([[0.,0,0],[2.,0,0],[6.,0,0],[12.,0,0]])
    sign=np.array([1,-1,-1,1])
    vals,tol,clip,raw=indicator_distance_values(pts,sign)
    assert np.array_equal(raw,[-2.,2.,6.,-6.]),raw
    assert np.allclose(vals,[-.12,.12,.12,-.12]),vals
    assert abs(tol-0.00012)<1e-13
    assert abs(clip-0.12)<1e-13
    report={"source":"feature/automatic-subdomainer recovered math",
            "status":"standalone synthetic regression, not independent Leapfrog export parity",
            "tensor_max_abs_vs_analytic":max_abs,
            "tensor_determinants":dets.tolist(),
            "tensor_min_eigenvalue":float(np.min(eigs)),
            "tensor_cutoff_identity_exact":bool(np.array_equal(mats[2:],np.repeat(np.eye(3)[None,:,:],2,axis=0))),
            "indicator_raw":raw.tolist(),"indicator_clipped":vals.tolist(),
            "fit_accuracy":tol,"value_clip":clip}
    print(json.dumps(report,indent=2))
    with open("jizai_structural_input_test.json","w") as f: json.dump(report,f,indent=2)

if __name__=="__main__":
    main()
