"""No Leapfrog installation or proprietary input is used by these smoke tests."""
import numpy as np
from dynamic_domain_refinement import refine_domains


def main():
    rng = np.random.default_rng(42)
    points = rng.normal(size=(120, 3))
    matrices = np.repeat(np.eye(3)[None], len(points), axis=0)
    first = np.repeat([42, 73, 151], 40)

    reverted, diagnostics = refine_domains(
        points, matrices, first, max_changed_fraction=0
    )
    assert np.array_equal(reverted, first)
    assert not diagnostics["accepted"]

    moved, _ = refine_domains(
        points, matrices, first, max_changed_fraction=1
    )
    repeated, _ = refine_domains(
        points, matrices, first, max_changed_fraction=1
    )
    assert moved.shape == first.shape
    assert np.array_equal(moved, repeated)
    assert set(np.unique(moved)) == {42, 73, 151}

    unchanged, metadata = refine_domains(
        points, matrices, np.zeros(len(points), dtype=int)
    )
    assert np.all(unchanged == 0) and metadata["accepted"]

    try:
        refine_domains(points[:10], matrices, first)
    except ValueError:
        pass
    else:
        raise AssertionError("Shape validation did not reject mismatched arrays")
    print("PASS: offline reassignment gate, ID preservation, determinism and shape validation")


if __name__ == "__main__":
    main()
