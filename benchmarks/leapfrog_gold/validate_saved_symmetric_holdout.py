"""Independent third-geometry LVA benchmark, using only saved user-owned projects.

Offline-only. Requires the original authored helper modules adjacent to this
script (included in companion research kit), NumPy, SciPy, sklearn and thrift.
No executable, license service, Windows runner, or proprietary binaries run.
Never pass reference labels into either clusterer.

python validate_saved_symmetric_holdout.py \
  --project "Same project after adding LVA.zip" \
  --endpoint-corrections first_fold_boundary_corrections_inferred.csv \
  --output symmetric_fold_validation.json
"""
from __future__ import annotations
import argparse
import json
import sys
import types
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _initialize_standalone_imports(helpers: Path):
    """Load authored standalone builder, substituting unavailable native types."""
    sys.path.insert(0, str(helpers))
    pkg = types.ModuleType("polatory")
    pkg.__path__ = [str(helpers)]
    sys.modules["polatory"] = pkg
    structural = types.ModuleType("polatory._structural")
    structural.StructuralDomain3 = object
    structural.StructuralDomainBuilder3 = type("StructuralDomainBuilder3", (), {})
    structural.StructuralTrendType = type(
        "StructuralTrendType", (), {"STRONGEST_ALONG_INPUTS": 0, "NON_DECAYING": 1}
    )
    sys.modules["polatory._structural"] = structural
    aux = types.ModuleType("polatory.automatic_domain_builder")
    aux.AutomaticStructuralDomainDiagnostics3 = type("AutomaticStructuralDomainDiagnostics3", (), {})
    sys.modules["polatory.automatic_domain_builder"] = aux
    labeled = types.ModuleType("polatory.labeled_domain_builder")
    labeled.LabeledStructuralDomainBuilder3 = type("LabeledStructuralDomainBuilder3", (), {})
    labeled.sample_single_input_anisotropies3 = lambda *args, **kwargs: None
    sys.modules["polatory.labeled_domain_builder"] = labeled
    from polatory.leapfrog_automatic_domain_builder_fixed import AutomaticStructuralDomainBuilder3
    from validate_orientation_saved_project import (
        apply_original_points, load_saved_project, tensor_field
    )
    from hybrid_real_grid_domainer import Settings, fit_hybrid
    return AutomaticStructuralDomainBuilder3, apply_original_points, load_saved_project, tensor_field, Settings, fit_hybrid


def sample_tensors(points, vertices, gradients, strength, trend_range, rule, tensor_field):
    """Evaluate the predeclared tie rule without consulting target memberships."""
    tree = cKDTree(vertices)
    dist, nearest = tree.query(points, k=8)
    is_tied = np.abs(dist - dist[:, :1]) <= 1.e-8
    tied_count = int((is_tied.sum(axis=1) > 1).sum())
    if rule == "kdtree-default":
        first_dist, chosen = tree.query(points, k=1)
    elif rule == "min-index":
        chosen = np.min(np.where(is_tied, nearest, np.iinfo(np.int64).max), axis=1)
        first_dist = dist[:, 0]
    elif rule == "max-index":
        chosen = np.max(np.where(is_tied, nearest, -1), axis=1)
        first_dist = dist[:, 0]
    else:
        raise ValueError(f"Unknown predeclared tie rule: {rule}")
    tensors = tensor_field(strength, trend_range, first_dist, gradients[chosen])
    return tensors, tied_count


def run(args):
    (Builder, apply_points, load_project, tensor_field,
     Settings, fit_hybrid) = _initialize_standalone_imports(args.helpers)
    record = load_project(
        args.project, "p/uaaaaaaaaaaa.lfdata", "p/4aaaaaaaaaaa.lfdata", None
    )
    canonical, corrections = apply_points(record["points"], args.endpoint_corrections)
    reference = np.full(len(canonical), -1, np.int64)
    for domain_id, point_indices in zip(record["domain_ids"], record["domain_indices"]):
        if np.any(reference[point_indices] != -1):
            raise ValueError("Overlapping saved domain indices")
        reference[point_indices] = domain_id
    if np.any(reference == -1):
        raise ValueError("Incomplete domain membership in saved project")

    fixed = Builder(
        centroid_count=6000, minimum_cluster_fraction=.001,
        maximum_cluster_fraction=.10, consistency_threshold=.6
    )
    rows = []
    for source, xyz in (("inferred-endpoints", canonical), ("raw-rbf-seeds", record["points"])):
        minimum, maximum, axes, grid_shape, centroids = fixed._prepare_grid(xyz)
        for rule in ("kdtree-default", "min-index", "max-index"):
            grid_tensors, grid_ties = sample_tensors(
                centroids, record["vertices"], record["gradients"],
                record["strength"], record["trend_range"], rule, tensor_field
            )
            point_tensors, point_ties = sample_tensors(
                xyz, record["vertices"], record["gradients"],
                record["strength"], record["trend_range"], rule, tensor_field
            )
            # Predictions completed without accessing reference membership.
            initial, _, _, _, _ = fixed._automatic_labels(
                xyz, grid_tensors, minimum, maximum, grid_shape
            )
            predicted, diagnostics = fit_hybrid(
                xyz, point_tensors, initial, Settings()
            )
            rows.append({
                "coordinates": source,
                "tie_rule": rule,
                "grid_shape": list(grid_shape),
                "point_count": len(xyz),
                "grid_ties": grid_ties,
                "point_ties": point_ties,
                "reference_domains": len(record["domain_ids"]),
                "initial_domains": int(np.unique(initial).size),
                "hybrid_domains": int(np.unique(predicted).size),
                "initial_ari": float(adjusted_rand_score(reference, initial)),
                "hybrid_ari": float(adjusted_rand_score(reference, predicted)),
                "initial_nmi": float(normalized_mutual_info_score(reference, initial)),
                "hybrid_nmi": float(normalized_mutual_info_score(reference, predicted)),
                "guard_accepted": bool(diagnostics["stage2"]["accepted"]),
                "proposed_reassignment_fraction": diagnostics["stage2"]["changed_fraction"],
                "accepted_merges": diagnostics["merges"],
                "born_domains": diagnostics["births"],
            })
            print(f"{source:18s} {rule:15s}: "
                  f"grid={rows[-1]['initial_ari']:.6f} "
                  f"hybrid={rows[-1]['hybrid_ari']:.6f} "
                  f"merges={rows[-1]['accepted_merges']}", flush=True)
    result = {
        "project": args.project.name,
        "geometry": "independent symmetric fold",
        "reference_type": "saved local RBF core memberships, not verified Extract Clusters category",
        "known_limitations": (
            "147 point ties; two terminal real-input XYZ corrections are inferred "
            "rather than independently verified. Saved core labels need not equal "
            "Leapfrog's extracted clustering labels."
        ),
        "inferred_changed_endpoints": len(corrections),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--endpoint-corrections", type=Path, required=True)
    p.add_argument("--helpers", type=Path, default=Path(__file__).resolve().parent,
                   help="Path to authored offline helper scripts and two clusterers")
    p.add_argument("--output", type=Path, default=Path("symmetric_fold_holdout.json"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
