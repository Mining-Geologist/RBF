"""EXPLORATORY, reference-free gate for large hybrid domain reassignments.

Created after inspecting two independent folded holdouts: do not describe their
reported scores as out-of-sample evidence for this rule. A genuinely new
geometry is needed to validate generalization. No native Leapfrog runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from hybrid_real_grid_domainer import Settings, fit_hybrid, get_edges


@dataclass(frozen=True)
class CoherenceDecision:
    accepted: bool
    criterion: str
    original_coherence: float
    proposed_coherence: float
    difference: float
    baseline_guard_accepted: bool


def local_graph_coherence(points, labels, neighbours=8):
    """Unit-interval fraction of same-domain nearest-neighbour graph links.

    Edge weights decline as exp(-separation/median graph distance).
    No reference labels or known target number of clusters are consulted.
    """
    x = np.asarray(points, dtype=float)
    memberships = np.asarray(labels)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2 or memberships.shape != (len(x),):
        raise ValueError("Expected points (N,3) and labels (N,) with N>=2")
    u, v = get_edges(x, neighbours)
    if len(u) == 0:
        return 1.0
    distances = np.linalg.norm(x[u] - x[v], axis=1)
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-distances / max(scale, 1e-12))
    return float(np.average(memberships[u] == memberships[v], weights=weights))


def fit_hybrid_with_exploratory_gate(points, tensors, grid_labels, settings=Settings(),
                                     min_gain=0.0):
    """Use small-change safety gate except when big changes raise spatial coherence.

    This is a research hypothesis selected AFTER viewing independent-fold
    outcomes, not prospectively held-out validated. Keep outside main.
    """
    if not np.isfinite(min_gain) or min_gain < 0:
        raise ValueError("min_gain must be a finite nonnegative number")
    guarded, guarded_stats = fit_hybrid(points, tensors, grid_labels, settings)
    if guarded_stats["stage2"]["accepted"]:
        score = local_graph_coherence(points, guarded, settings.knn)
        return guarded, {
            "criterion": "original bounded-movement safeguard",
            "accepted_large_change": False,
            "original_coherence": local_graph_coherence(points, grid_labels, settings.knn),
            "proposed_coherence": score,
            "difference": score - local_graph_coherence(points, grid_labels, settings.knn),
            "guarded": guarded_stats,
        }
    unrestricted, unrestricted_stats = fit_hybrid(
        points, tensors, grid_labels, replace(settings, changed_gate=1.0))
    original_score = local_graph_coherence(points, grid_labels, settings.knn)
    proposed_score = local_graph_coherence(points, unrestricted, settings.knn)
    accept = (proposed_score - original_score) > min_gain
    return (unrestricted if accept else guarded), {
        "criterion": "large-change graph-coherence increase (exploratory)",
        "accepted_large_change": bool(accept),
        "original_coherence": original_score,
        "proposed_coherence": proposed_score,
        "difference": proposed_score - original_score,
        "guarded": guarded_stats,
        "unrestricted": unrestricted_stats,
    }
