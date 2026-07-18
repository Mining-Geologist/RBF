import numpy as np

from polatory.labeled_domain_builder import (
    LabeledStructuralDomainBuilder3,
    SPHEROIDAL3_C,
    sample_single_input_anisotropies3,
)


def test_identity_anisotropy_uses_base_range_and_five_x_cap():
    points = np.column_stack(
        [np.arange(20, dtype=float), np.zeros(20), np.zeros(20)]
    )
    labels = np.zeros(20, dtype=np.int64)
    labels[2:] = 1
    anisotropies = np.repeat(np.eye(3)[None, :, :], len(points), axis=0)

    builder = LabeledStructuralDomainBuilder3(base_range=100.0)
    result = builder.compute(points, labels, anisotropies)

    first = result[0]
    assert len(first.core_indices) == 2
    assert len(first.support_indices) <= 10
    np.testing.assert_allclose(
        first.desired_internal_radius,
        100.0 / np.sqrt(SPHEROIDAL3_C),
    )
    np.testing.assert_allclose(
        first.local_kernel_range,
        first.internal_radius * np.sqrt(SPHEROIDAL3_C),
    )


def test_mean_matrix_and_bbox_expansion():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [10.0, 0.0, 0.0],
        ]
    )
    labels = np.array([4, 4, 9])
    matrices = np.array(
        [
            np.diag([2.0, 1.0, 0.5]),
            np.diag([4.0, 1.0, 0.25]),
            np.eye(3),
        ]
    )

    builder = LabeledStructuralDomainBuilder3(base_range=20.0)
    item = builder.compute(points, labels, matrices)[0]

    expected = np.diag([3.0, 1.0, 0.375])
    np.testing.assert_allclose(item.anisotropy, expected)
    expected_radius = 20.0 * 3.0 / np.sqrt(SPHEROIDAL3_C)
    np.testing.assert_allclose(item.internal_radius, expected_radius)

    expansion = expected_radius * np.linalg.norm(np.linalg.inv(expected), axis=1)
    np.testing.assert_allclose(
        item.bbox_min,
        np.array([0.0, 0.0, 0.0]) - expansion,
    )
    np.testing.assert_allclose(
        item.bbox_max,
        np.array([1.0, 2.0, 3.0]) + expansion,
    )


def test_single_input_decay_has_hard_four_range_cutoff():
    class Input:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        strength = 5.0
        range = 10.0

    matrices = sample_single_input_anisotropies3(
        np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]),
        Input(),
    )

    assert not np.allclose(matrices[0], np.eye(3))
    np.testing.assert_allclose(matrices[1], np.eye(3), atol=0.0, rtol=0.0)
