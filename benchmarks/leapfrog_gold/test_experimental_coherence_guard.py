"""Synthetic, source-independent tests for exploratory hybrid graph-coherence gate."""
import numpy as np
from experimental_coherence_guard import (
    local_graph_coherence, fit_hybrid_with_exploratory_gate,
)
from hybrid_real_grid_domainer import Settings


def main():
    rng = np.random.default_rng(140)
    points = np.r_[
        rng.normal(loc=(-20, 0, 0), scale=.3, size=(24, 3)),
        rng.normal(loc=(20, 0, 0), scale=.3, size=(24, 3)),
    ]
    contiguous = np.repeat([9, 11], 24)
    mixed = np.tile([9, 11], 24)
    assert local_graph_coherence(points, contiguous) > local_graph_coherence(points, mixed)
    assert np.isclose(
        local_graph_coherence(points, contiguous),
        local_graph_coherence(points, contiguous + 1000),
    )
    tensors = np.repeat(np.eye(3)[None], len(points), axis=0)
    cfg = Settings(refine=False, merge_small_fraction=0., split=False)
    result, diagnostics = fit_hybrid_with_exploratory_gate(points, tensors, contiguous, cfg)
    assert len(result) == len(points)
    assert diagnostics["accepted_large_change"] is False
    repeated, _ = fit_hybrid_with_exploratory_gate(points, tensors, contiguous, cfg)
    assert np.array_equal(result, repeated)
    try:
        fit_hybrid_with_exploratory_gate(points, tensors, contiguous, cfg, min_gain=-1.)
    except ValueError:
        pass
    else:
        raise AssertionError("negative threshold must raise")
    print("PASS: reference-free coherence responds to geometry; deterministic hybrid gate")


if __name__ == "__main__":
    main()
