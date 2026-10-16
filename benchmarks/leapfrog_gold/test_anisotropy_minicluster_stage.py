"""Synthetic tests for offline anisotropy-aware mini-cluster prototype."""
import numpy as np
from anisotropy_minicluster_stage import initial_miniclusters, two_stage_merge


def main():
    rng = np.random.default_rng(321)
    for shape in ((2, 2, 2), (4, 3, 2)):
        n = int(np.prod(shape))
        matrices = np.repeat(np.eye(3)[None], n, axis=0)
        original, nmerges = initial_miniclusters(
            matrices, shape, micro_size=1, micro_consistency=.995)
        assert nmerges == 0
        assert np.array_equal(original, np.arange(n))
        for maxsize in (2, 4):
            groups, merges = initial_miniclusters(
                matrices, shape, micro_size=maxsize, micro_consistency=.995)
            _, populations = np.unique(groups, return_counts=True)
            assert populations.max() <= maxsize
            assert np.unique(groups).size == n-merges

        points = rng.uniform(-1, 1, (80, 3))
        bounds_low = np.full(3, -1.)
        bounds_high = np.full(3, 1.)
        predictions, diag = two_stage_merge(
            points, matrices, shape, bounds_low, bounds_high,
            micro_size=4, maximum_fraction=1.)
        repeated, _ = two_stage_merge(
            points, matrices, shape, bounds_low, bounds_high,
            micro_size=4, maximum_fraction=1.)
        assert len(predictions) == len(points)
        assert np.array_equal(predictions, repeated)
        assert np.unique(predictions).size == 1
        assert diag["initial_merges"] + diag["stage2_merges"] == (
            n-diag["final_grid_domains"])
    print("PASS: bounded mini-cluster grouping, merge accounting, "
          "identity convergence and determinism")


if __name__ == "__main__":
    main()
