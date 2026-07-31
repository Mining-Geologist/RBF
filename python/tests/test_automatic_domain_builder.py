import numpy as np

from polatory.automatic_domain_builder import AutomaticStructuralDomainBuilder3
from polatory.labeled_domain_builder import LabeledStructuralDomainBuilder3


class _TrendInput:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [20.0, 0.0, 5.0],
            [20.0, 20.0, 5.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    strength = 5.0
    range = 10.0


def _points() -> np.ndarray:
    x = np.linspace(-5.0, 35.0, 8)
    y = np.linspace(-2.0, 22.0, 5)
    z = np.linspace(-8.0, 18.0, 4)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _builder() -> AutomaticStructuralDomainBuilder3:
    return AutomaticStructuralDomainBuilder3(
        centroid_count=120,
        minimum_cluster_fraction=0.001,
        maximum_cluster_fraction=0.20,
        consistency_threshold=0.60,
        base_range=30.0,
        minimum_support_points=2,
    )


def test_automatic_builder_is_deterministic_and_assigns_every_point():
    points = _points()
    first = _builder()
    second = _builder()

    first_domains = first.build_from_inputs(
        points,
        [_TrendInput()],
        model_parameters=[0.0, 10.0, 30.0],
    )
    second_domains = second.build_from_inputs(
        points,
        [_TrendInput()],
        model_parameters=[0.0, 10.0, 30.0],
    )

    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_array_equal(first.centroid_labels_, second.centroid_labels_)
    assert len(first.labels_) == len(points)
    assert np.all(first.labels_ >= 0)
    assert len(np.unique(first.labels_)) == len(first_domains)
    assert len(first_domains) == len(second_domains)
    assert np.prod(first.centroid_grid_shape_) == first.centroid_count


def test_maximum_population_and_arbitrary_strength_range():
    points = _points()

    class OtherInput(_TrendInput):
        strength = 2.75
        range = 17.5

    builder = AutomaticStructuralDomainBuilder3(
        centroid_count=150,
        minimum_cluster_fraction=0.001,
        maximum_cluster_fraction=0.25,
        consistency_threshold=0.55,
        base_range=42.0,
    )
    domains = builder.build_from_inputs(
        points,
        [OtherInput()],
        model_parameters=[0.0, 12.0, 42.0],
    )

    counts = np.bincount(builder.labels_)
    assert counts.max() <= int(np.floor(0.25 * len(points)))
    assert len(domains) == len(counts)
    assert builder.diagnostics_.maximum_points == int(np.floor(0.25 * len(points)))
    assert builder.diagnostics_.final_domain_count == len(domains)


def test_generated_labels_use_exact_postcluster_builder():
    points = _points()
    builder = _builder()
    builder.build_from_inputs(
        points,
        [_TrendInput()],
        model_parameters=[0.0, 10.0, 30.0],
    )

    from polatory.labeled_domain_builder import sample_single_input_anisotropies3

    matrices = sample_single_input_anisotropies3(points, _TrendInput())
    exact = LabeledStructuralDomainBuilder3(
        base_range=30.0,
        minimum_support_points=2,
    )
    expected = exact.compute(points, builder.labels_, matrices)
    actual = builder.diagnostics_.postcluster

    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected):
        assert observed.label == reference.label
        np.testing.assert_array_equal(observed.core_indices, reference.core_indices)
        np.testing.assert_array_equal(observed.support_indices, reference.support_indices)
        np.testing.assert_allclose(observed.anisotropy, reference.anisotropy)
        np.testing.assert_allclose(observed.bbox_min, reference.bbox_min)
        np.testing.assert_allclose(observed.bbox_max, reference.bbox_max)
        np.testing.assert_allclose(
            observed.local_kernel_range,
            reference.local_kernel_range,
        )


def test_point_order_does_not_change_spatial_partition():
    points = _points()
    permutation = np.random.default_rng(1234).permutation(len(points))

    original = _builder()
    shuffled = _builder()
    original.build_from_inputs(
        points,
        [_TrendInput()],
        model_parameters=[0.0, 10.0, 30.0],
    )
    shuffled.build_from_inputs(
        points[permutation],
        [_TrendInput()],
        model_parameters=[0.0, 10.0, 30.0],
    )

    restored = np.empty_like(shuffled.labels_)
    restored[permutation] = shuffled.labels_
    np.testing.assert_array_equal(original.labels_, restored)
