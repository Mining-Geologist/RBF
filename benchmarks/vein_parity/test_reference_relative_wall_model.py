"""Synthetic reference-relative reconstruction checks; no Leapfrog binaries."""
import numpy as np
from reference_relative_wall_model import ReferenceHeightField, ReferenceRelativeWalls

def run():
    u=np.linspace(-10,10,11)
    xy=np.stack(np.meshgrid(u,u,indexing="ij"),axis=-1).reshape(-1,2)
    xyz=np.column_stack([xy, 0.02*xy[:,0]**2+0.01*xy[:,1]])
    ref=ReferenceHeightField(xyz)
    controlxy=xy[::6].copy()
    hw=np.column_stack([controlxy,ref.height(controlxy)+4+.25*controlxy[:,0]])
    fw=np.column_stack([controlxy,ref.height(controlxy)-4+.15*controlxy[:,1]])
    model=ReferenceRelativeWalls(xyz,hw,fw,1.)
    withheld=xy[1::4]
    predicted=model.evaluate(withheld)
    targethw=ref.height(withheld)+4+.25*withheld[:,0]
    targetfw=ref.height(withheld)-4+.15*withheld[:,1]
    assert np.max(np.abs(predicted["HW"]-targethw))<1e-6
    assert np.max(np.abs(predicted["FW"]-targetfw))<1e-6
    direct=model.evaluate(withheld,relative=False)
    assert np.sqrt(np.mean((direct["HW"]-targethw)**2))>0.01
    # Opposite wall polarity at an outside trace is a signed pinchout seed.
    hw[-1,2]=ref.height(controlxy[-1:])[0]-1
    fw[-1,2]=ref.height(controlxy[-1:])[0]+1
    pinch=ReferenceRelativeWalls(xyz,hw,fw,1.)
    assert pinch.signed_thickness(controlxy[-1:])[0]<0
    assert not pinch.is_inside(np.array([[*controlxy[-1],ref.height(controlxy[-1:])[0]]]))[0]
    # A true folded surface cannot be reduced to the same global XY twice.
    try:
        ReferenceHeightField(np.r_[xyz, [[0,0,100]]])
    except ValueError:pass
    else:raise AssertionError("must reject multi-valued folded geometry")
    square=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    try:
        ReferenceHeightField(square, np.array([[0,1,2],[0,3,2]]))
    except ValueError:pass
    else:raise AssertionError("must reject overlapping reversed reference face orientation")
    ReferenceHeightField(square,np.array([[0,1,2],[0,2,3]]))
    print("PASS: exact reference offset recovery, outside reversal, affine control, 3D fold guard")

if __name__=="__main__":
    run()
