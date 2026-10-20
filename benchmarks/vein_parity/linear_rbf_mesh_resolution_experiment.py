"""Resolution-only synthetic test for a continuous linear RBF field.

This is an independent discretization control, not a native Leapfrog mesher.
The exact same smooth/radial field is evaluated on two shifted triangular
grids; errors are measured against the unmeshed continuous field.
"""
import json
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay


SUPPORTS=np.array([[-40,-40],[40,-40],[0,0],[-40,40],[40,40]],float)
THICKNESS=np.array([8.,13.,18.,7.,10.])
EXTRA=np.array([[4,0],[12,0],[20,0],[0,4],[0,12],[-12,0],[8,8],[16,16]],float)
OUTER=np.array([(x,y) for x in (-80,-40,0,40,80) for y in (-80,-40,0,40,80)
                if (x,y) not in {(-40,-40),(40,-40),(0,0),(-40,40),(40,40)}],float)


def predict_piecewise_linear(triangulation, vertex_values, query_xy):
    sample=np.asarray(query_xy,float).reshape(-1,2)
    ids=triangulation.find_simplex(sample)
    if (ids.min()<0): raise ValueError("A query lies outside the synthetic mesh")
    local=sample-triangulation.transform[ids,2]
    uv=np.einsum('nij,nj->ni',triangulation.transform[ids,:2],local)
    bary=np.column_stack([uv,1-uv.sum(axis=1)])
    return np.einsum('ni,ni->n',vertex_values[triangulation.simplices[ids]],bary)


def experiment(spacing, query=EXTRA):
    xs=np.arange(-100+spacing/2,100+spacing/2+spacing/10,spacing)
    xx,yy=np.meshgrid(xs,xs,indexing="xy")
    nodes=np.column_stack([xx.ravel(),yy.ravel()])
    rbf=RBFInterpolator(SUPPORTS,THICKNESS,kernel="linear",degree=0)
    analytic=rbf(query).ravel()
    triangulation=Delaunay(nodes)
    approximate=predict_piecewise_linear(triangulation,rbf(nodes).ravel(),query)
    delta=approximate-analytic
    return {"spacing_m":float(spacing),"n_samples":len(query),
            "rmse_m":float(np.sqrt(np.mean(delta**2))),
            "max_abs_error_m":float(np.max(np.abs(delta))),
            "per_point_error_m":delta.tolist()}


if __name__=="__main__":
    a=experiment(10.)
    b=experiment(5.)
    print(json.dumps({"grid_10m":a,"grid_5m":b,
                      "rmse_improvement_factor":a["rmse_m"]/b["rmse_m"],
                      "native_g10_rmse_m_context_only":0.049899,
                      "native_g5_rmse_m_context_only":0.01175},indent=2))
