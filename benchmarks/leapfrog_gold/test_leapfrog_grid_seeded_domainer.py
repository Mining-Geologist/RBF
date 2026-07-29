from __future__ import annotations

import numpy as np

from leapfrog_grid_seeded_domainer import (
    GridSeededDomainer,
    reciprocal_determinant_consistency,
    six_connected_neighbours,
    symmetric_determinant,
    weighted_mean_matrix,
)


def test_six_connected_neighbours_use_c_order_strides() -> None:
    assert six_connected_neighbours(13, (3, 3, 3)) == (4, 22, 10, 16, 12, 14)


def test_symmetric_determinant_matches_numpy() -> None:
    matrix = np.array(
        [
            [2.0, 0.25, -0.1],
            [0.25, 1.5, 0.3],
            [-0.1, 0.3, 0.8],
        ]
    )
    assert np.isclose(symmetric_determinant(matrix), np.linalg.det(matrix))


def test_weighted_matrix_mean_uses_domain_sizes() -> None:
    first = np.diag([4.0, 1.0, 1.0])
    second = np.diag([1.0, 4.0, 1.0])
    result = weighted_mean_matrix(first, 4, second, 1)
    assert np.allclose(result, 0.8 * first + 0.2 * second)


def test_identical_volume_normalised_matrices_have_unit_consistency() -> None:
    matrix = np.diag([0.5, 1.0, 2.0])
    assert np.isclose(np.linalg.det(matrix), 1.0)
    assert np.isclose(reciprocal_determinant_consistency(matrix), 1.0)


def test_identical_grid_merges_to_one_domain() -> None:
    matrices = np.repeat(np.eye(3)[None, :, :], 8, axis=0)
    result = GridSeededDomainer(consistency_thresh=0.6, max_points=8).fit(
        matrices,
        (2, 2, 2),
    )
    assert result.merge_count == 7
    assert len(result.domains) == 1
    assert np.unique(result.grid_domain_ids).size == 1


def test_max_points_is_a_hard_merged_domain_limit() -> None:
    matrices = np.repeat(np.eye(3)[None, :, :], 8, axis=0)
    result = GridSeededDomainer(consistency_thresh=0.6, max_points=3).fit(
        matrices,
        (2, 2, 2),
    )
    assert max(domain.size for domain in result.domains.values()) <= 3
