import numpy as np

from polatory.leapfrog_values import leapfrog_indicator_values3


def test_nearest_opposite_signed_distances_and_automatic_scales():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
        ]
    )
    indicators = np.array([1.0, -1.0, 1.0, -1.0])

    result = leapfrog_indicator_values3(points, indicators)

    np.testing.assert_allclose(
        result.raw_signed_distances,
        np.array([-3.0, 3.0, -4.0, 4.0]),
    )
    np.testing.assert_allclose(result.data_diagonal, 14.0)
    np.testing.assert_allclose(result.fit_accuracy, 14.0e-5)
    np.testing.assert_allclose(result.clipping_distance, 0.14)
    np.testing.assert_allclose(
        result.values,
        np.array([-0.14, 0.14, -0.14, 0.14]),
    )


def test_explicit_clip_preserves_near_contact_values():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    indicators = np.array([1.0, -1.0, 1.0])

    result = leapfrog_indicator_values3(
        points,
        indicators,
        fit_accuracy=0.01,
        clipping_distance=2.0,
    )

    np.testing.assert_allclose(
        result.raw_signed_distances,
        np.array([-0.5, 0.5, -4.5]),
    )
    np.testing.assert_allclose(result.values, np.array([-0.5, 0.5, -2.0]))
    np.testing.assert_allclose(result.fit_accuracy, 0.01)
    np.testing.assert_allclose(result.clipping_distance, 2.0)
