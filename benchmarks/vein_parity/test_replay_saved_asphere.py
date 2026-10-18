"""Source-independent checks of a native saved 'asphere' kernel evaluator."""
import numpy as np
from replay_saved_asphere import (
    asphere_from_distances, evaluate_saved_solution, ASPHERE_TAIL
)

def test():
    x = np.array([0., 2., 5., 10., 12.])
    out = asphere_from_distances(x, 10.)
    assert out[0] == 1.0
    assert np.isclose(out[2], 0.625, atol=5e-9)
    assert np.all(np.diff(out) < 0.)
    assert np.isclose(out[3], ASPHERE_TAIL / np.sqrt(8.0))
    pts = np.array([[0., 0, 0], [20., 0, 0], [0., 20, 0]])
    coeff = np.array([.5, -.4, .25])
    test = evaluate_saved_solution(pts,coeff,pts,10.)
    assert np.allclose(test,asphere_from_distances(
        np.linalg.norm(pts[:,None,:]-pts[None,:,:],axis=2),10)@coeff)
    for bad_range in (0., -1.):
        try:asphere_from_distances(x,bad_range)
        except ValueError:pass
        else:raise AssertionError("Must reject invalid range")
    print("PASS: native saved asphere kernel, continuity, 3D centres and shape checks")

if __name__=="__main__":
    test()
