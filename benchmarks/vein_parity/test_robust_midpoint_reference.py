"""Purely synthetic falsification and mathematical controls for reference builder."""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from robust_midpoint_reference import fit_robust_reference


def _mq_wall_fit(x, z, q, ref):
    mu = np.mean(x, axis=0)
    scale = max(float(np.ptp(x,axis=0).max()),1.)
    def design(p):
        return np.column_stack((np.ones(len(p)),(p-mu)/scale))
    P,Q = design(x),design(q)
    K = np.sqrt(1.+cdist(x,x)**2)
    V = np.sqrt(1.+cdist(q,x)**2)
    fit = solve(np.block([[K,P],[P.T,np.zeros((3,3))]]),
                np.r_[z-ref(x),np.zeros(3)],assume_a="sym")
    return ref(q)+V@fit[:len(x)]+Q@fit[len(x):]


def main():
    grid=np.stack(np.meshgrid(np.arange(-80.,81.,40.),
                              np.arange(-80.,81.,40.),
                              indexing="ij"),axis=-1).reshape(-1,2)
    nominal=np.column_stack((grid,np.full(len(grid),100.)))
    flat=fit_robust_reference(nominal)
    assert np.max(np.abs(flat.height(grid)-100.))<1e-8
    # Four extreme end-of-hole points, including one dual-valued XY site.
    hole=np.delete(nominal,[7,12,17],axis=0)
    hole=np.vstack([hole, [[-40.,0.,196.],[40.,0.,4.],
                           [0.,0.,1.],[0.,0.,198.]]])
    robust=fit_robust_reference(hole)
    assert len(robust.rejected)==2,robust.rejected
    assert np.max(np.abs(robust.height(grid)-100.))<1.
    # Multiple vertically distinct XY sites must retain fold topology.
    fold=np.vstack([nominal, [[0.,0.,80.],[0.,0.,120.],
                              [40.,0.,80.],[40.,0.,120.]]])
    try:
        fit_robust_reference(fold)
    except ValueError as e:
        assert "Multisheet" in str(e)
    else:
        raise AssertionError("Cannot silently flatten folded input")
    # In a polynomially augmented MQ1, affine change of reference cancels.
    rng=np.random.default_rng(301)
    controls=rng.uniform(-60,60,(23,2))
    prediction_xyz=rng.uniform(-90,90,(31,2))
    heights=100+2*np.sin(controls[:,0]/30)-.15*controls[:,1]
    def r(q):
        return 90+q[:,0]**2*.004-q[:,1]*.05
    def r2(q):
        return r(q)+27.5+.25*q[:,0]-.34*q[:,1]
    first=_mq_wall_fit(controls,heights,prediction_xyz,r)
    second=_mq_wall_fit(controls,heights,prediction_xyz,r2)
    assert np.max(abs(first-second))<1e-8
    print("PASS: robust reference filters extremes; refuses folds; affine drift cancels reference datum and tilt")


if __name__=="__main__":
    main()
