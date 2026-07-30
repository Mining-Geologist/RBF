from __future__ import annotations

import numpy as np

import polatory


def test_unaffected_heap_edges_survive_neighbour_merge() -> None:
    """Independent valid pairs must not disappear after a neighbouring merge.

    Four identical cells form a one-dimensional chain. With a hard maximum
    domain size of two, the correct result is two merged pairs. The old
    neighbour-version increment invalidated the untouched (2, 3) heap entry
    after merging (0, 1), leaving three domains instead.
    """

    points = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.5, 0.0, 0.0],
        ],
        dtype=float,
    )
    centroid_anisotropies = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
    builder = polatory.AutomaticStructuralDomainBuilder3(
        centroid_count=4,
        minimum_cluster_fraction=0.001,
        maximum_cluster_fraction=0.5,
        consistency_threshold=0.6,
    )

    labels, centroid_labels, minimum_points, maximum_points, merge_count = (
        builder._automatic_labels(
            points,
            centroid_anisotropies,
            np.array([0.0, 0.0, 0.0]),
            np.array([4.0, 0.0, 0.0]),
            (4, 1, 1),
        )
    )

    assert minimum_points == 1
    assert maximum_points == 2
    assert merge_count == 2
    assert np.unique(labels).size == 2
    assert np.unique(centroid_labels[centroid_labels >= 0]).size == 2
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
