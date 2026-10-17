"""Synthetic controls for hybrid grid-to-real dynamic SubDomainer.

No native executable, data archive or reference memberships required.
"""
import numpy as np
from hybrid_real_grid_domainer import Settings, fit_hybrid, get_edges


def check_merge_and_determinism():
    rng = np.random.default_rng(7)
    xyz = np.r_[
        rng.normal((-1., 0, 0), (.2, .2, .2), (12, 3)),
        rng.normal((1., 0, 0), (.2, .2, .2), (12, 3)),
    ]
    tensors = np.repeat(np.eye(3)[None], 24, axis=0)
    labels = np.repeat([100, 200], 12)
    cfg = Settings(refine=False, knn=18, merge_small_fraction=.6,
                   merge_max_fraction=1., merge_min_similarity=.99,
                   merge_min_contacts=.005)
    y, info = fit_hybrid(xyz, tensors, labels, cfg)
    assert np.unique(y).size == 1 and info["merges"] == 1, info
    again, _ = fit_hybrid(xyz, tensors, labels, cfg)
    assert np.array_equal(y, again)


def check_domain_birth():
    rng = np.random.default_rng(66)
    xyz = np.r_[
        rng.normal((-10, 0, 0), (.1, .1, .1), (18, 3)),
        rng.normal((10, 0, 0), (.1, .1, .1), (18, 3)),
    ]
    tensors = np.zeros((36, 3, 3))
    tensors[:18] = np.diag([2., .5, 1.])
    tensors[18:] = np.diag([.5, 2., 1.])
    cfg = Settings(refine=False, knn=4, merge_small_fraction=0,
                   split=True, split_min_fraction=.1,
                   split_max_similarity=.9)
    y, info = fit_hybrid(xyz, tensors, np.full(36, 9), cfg)
    assert np.unique(y).size == 2 and info["births"] == 1, info
    assert y[0] != y[18]


def check_input_validation():
    xyz = np.array([[i, 0, 0] for i in range(8)], dtype=float)
    matrices = np.repeat(np.eye(3)[None], 8, axis=0)
    labels = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    cfg = Settings(refine=False, merge_small_fraction=0)
    y, info = fit_hybrid(xyz, matrices, labels, cfg)
    assert np.unique(y).size == 2 and info["merges"] == 0
    for bad in (matrices[:-1], np.zeros_like(matrices)):
        try:
            fit_hybrid(xyz, bad, labels, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected input validation failure")
    u, v = get_edges(xyz, 4)
    assert len(set(map(tuple, np.c_[u, v]))) == len(u)


if __name__ == "__main__":
    check_merge_and_determinism()
    check_domain_birth()
    check_input_validation()
    print("PASS: hybrid merge, optional birth, deterministic labels, input validation")
