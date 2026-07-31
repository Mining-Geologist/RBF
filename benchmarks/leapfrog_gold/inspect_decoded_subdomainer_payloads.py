"""Inventory decoded Leapfrog SubDomainer payloads and candidate label vectors.

Recent real-stage diagnostics grouped the real locations using the reconstructed
coarse labels.  Before changing the clustering algorithm again, this script checks
whether the decoded benchmark already contains Leapfrog's original coarse, parent,
group, centroid, or intermediate assignments.

The script is read-only.  It inventories the selected case plus shared root-level
files, inspects CSV/JSON/NPY/NPZ payloads, and reports one-dimensional integer-like
vectors that could represent point or centroid assignments.  Point-length vectors
are compared with the decoded final ``point_clusters.csv`` labels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LABEL_TERMS = (
    "cluster",
    "domain",
    "label",
    "group",
    "partition",
    "component",
    "parent",
    "coarse",
    "assignment",
    "membership",
)


def _looks_labelled(name: str) -> bool:
    lowered = name.lower()
    return any(term in lowered for term in LABEL_TERMS)


def _integer_like(values: np.ndarray) -> bool:
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        return False
    if values.dtype.kind in "biu":
        return True
    if values.dtype.kind != "f":
        return False
    finite = values[np.isfinite(values)]
    if len(finite) != len(values):
        return False
    return bool(np.all(np.abs(finite - np.round(finite)) <= 1e-9))


def _normalise_labels(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if values.dtype.kind == "f":
        values = np.round(values)
    _, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.int64)


def _weighted_purity(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = _normalise_labels(reference)
    candidate = _normalise_labels(candidate)
    correct = 0
    for value in np.unique(candidate):
        indices = np.flatnonzero(candidate == value)
        if len(indices):
            counts = np.bincount(reference[indices])
            correct += int(counts.max())
    return correct / max(len(reference), 1)


def _comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    reference_i = _normalise_labels(reference)
    candidate_i = _normalise_labels(candidate)
    intersections = len(np.unique(np.column_stack((reference_i, candidate_i)), axis=0))
    return {
        "ari": float(adjusted_rand_score(reference_i, candidate_i)),
        "ami": float(adjusted_mutual_info_score(reference_i, candidate_i)),
        "candidate_to_final_purity": _weighted_purity(reference_i, candidate_i),
        "final_to_candidate_purity": _weighted_purity(candidate_i, reference_i),
        "intersections": float(intersections),
    }


def _top_counts(values: np.ndarray, limit: int = 8) -> str:
    counts = Counter(_normalise_labels(values).tolist())
    return ",".join(f"{label}:{count}" for label, count in counts.most_common(limit))


def _summarise_vector(
    *,
    source: str,
    key: str,
    values: np.ndarray,
    point_count: int,
    centroid_count: int | None,
    final_labels: np.ndarray,
) -> dict[str, Any] | None:
    values = np.asarray(values)
    if values.ndim != 1 or not _integer_like(values):
        return None
    count = len(values)
    if count == 0:
        return None
    labels = _normalise_labels(values)
    row: dict[str, Any] = {
        "source": source,
        "key": key,
        "length": count,
        "unique": int(len(np.unique(labels))),
        "role": (
            "point"
            if count == point_count
            else "centroid"
            if centroid_count is not None and count == centroid_count
            else "other"
        ),
        "top_counts": _top_counts(labels),
    }
    if count == point_count:
        row.update(_comparison(final_labels, labels))
        row["exact_final"] = bool(np.array_equal(labels, _normalise_labels(final_labels)))
    return row


def _walk_json(value: Any, prefix: str = "$") -> Iterable[tuple[str, np.ndarray]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        array = np.asarray(value)
        if array.ndim == 1:
            yield prefix, array
        else:
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    yield from _walk_json(child, f"{prefix}[{index}]")


def _candidate_files(root: Path, case_dir: Path) -> list[Path]:
    files: set[Path] = set()
    for path in root.iterdir():
        if path.is_file():
            files.add(path)
    if case_dir.is_dir():
        files.update(path for path in case_dir.rglob("*") if path.is_file())
    return sorted(files, key=lambda path: str(path).lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="S3_R100")
    parser.add_argument(
        "--decoded-root", type=Path, default=Path("Leapfrog_LVA_decoded_benchmark")
    )
    parser.add_argument(
        "--centroid-count",
        type=int,
        default=6000,
        help="Expected first-stage centroid-vector length; use 0 to disable.",
    )
    args = parser.parse_args()

    root = args.decoded_root
    case_name = args.case.strip().upper()
    case_dir = root / case_name
    points_path = root / "Used Data.csv"
    labels_path = case_dir / "point_clusters.csv"
    if not points_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Expected {points_path} and {labels_path}; decoded benchmark is incomplete."
        )

    points_frame = pd.read_csv(points_path)
    final_frame = pd.read_csv(labels_path)
    if "cluster" not in final_frame:
        raise ValueError(f"{labels_path} has no 'cluster' column")
    point_count = len(points_frame)
    final_labels = final_frame["cluster"].to_numpy()
    centroid_count = args.centroid_count if args.centroid_count > 0 else None

    files = _candidate_files(root, case_dir)
    print(
        f"decoded_root={root} case={case_name} files={len(files)} "
        f"points={point_count} final_domains={len(np.unique(final_labels))} "
        f"expected_centroids={centroid_count if centroid_count is not None else 'disabled'}"
    )
    print("\nDecoded file inventory:")
    for path in files:
        relative = path.relative_to(root)
        print(f" {str(relative):<72s} {path.stat().st_size:>12d} bytes")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in files:
        relative = str(path.relative_to(root))
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                frame = pd.read_csv(path)
                print(
                    f"\nCSV {relative}: rows={len(frame)} columns={list(frame.columns)}"
                )
                for column in frame.columns:
                    values = frame[column].to_numpy()
                    row = _summarise_vector(
                        source=relative,
                        key=str(column),
                        values=values,
                        point_count=point_count,
                        centroid_count=centroid_count,
                        final_labels=final_labels,
                    )
                    if row is not None and (
                        _looks_labelled(str(column))
                        or row["role"] in {"point", "centroid"}
                        or int(row["unique"]) <= 128
                    ):
                        rows.append(row)
            elif suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                for key, values in _walk_json(payload):
                    row = _summarise_vector(
                        source=relative,
                        key=key,
                        values=values,
                        point_count=point_count,
                        centroid_count=centroid_count,
                        final_labels=final_labels,
                    )
                    if row is not None and (
                        _looks_labelled(key)
                        or row["role"] in {"point", "centroid"}
                        or int(row["unique"]) <= 128
                    ):
                        rows.append(row)
            elif suffix == ".npy":
                values = np.load(path, allow_pickle=False)
                row = _summarise_vector(
                    source=relative,
                    key="array",
                    values=values,
                    point_count=point_count,
                    centroid_count=centroid_count,
                    final_labels=final_labels,
                )
                if row is not None:
                    rows.append(row)
            elif suffix == ".npz":
                with np.load(path, allow_pickle=False) as payload:
                    for key in payload.files:
                        row = _summarise_vector(
                            source=relative,
                            key=key,
                            values=payload[key],
                            point_count=point_count,
                            centroid_count=centroid_count,
                            final_labels=final_labels,
                        )
                        if row is not None:
                            rows.append(row)
        except Exception as exc:  # diagnostic should continue through malformed files
            errors.append(f"{relative}: {type(exc).__name__}: {exc}")

    print("\nCandidate assignment vectors:")
    if not rows:
        print(" none")
    else:
        rows.sort(
            key=lambda row: (
                {"point": 0, "centroid": 1, "other": 2}[str(row["role"])],
                -float(row.get("ari", -1.0)),
                str(row["source"]),
                str(row["key"]),
            )
        )
        print(
            " role     length unique source::key                                      "
            "ARI    AMI  candPur finalPur intersections exact top_counts"
        )
        for row in rows:
            identifier = f"{row['source']}::{row['key']}"
            if row["role"] == "point":
                print(
                    f" {str(row['role']):<8s} {int(row['length']):>6d} {int(row['unique']):>6d} "
                    f"{identifier:<48.48s} {float(row['ari']):>6.3f} "
                    f"{float(row['ami']):>6.3f} "
                    f"{float(row['candidate_to_final_purity']):>7.3f} "
                    f"{float(row['final_to_candidate_purity']):>8.3f} "
                    f"{int(row['intersections']):>13d} "
                    f"{str(bool(row['exact_final'])):<5s} {row['top_counts']}"
                )
            else:
                print(
                    f" {str(row['role']):<8s} {int(row['length']):>6d} {int(row['unique']):>6d} "
                    f"{identifier:<48.48s} {'-':>6s} {'-':>6s} {'-':>7s} "
                    f"{'-':>8s} {'-':>13s} {'-':<5s} {row['top_counts']}"
                )

    point_candidates = [
        row
        for row in rows
        if row["role"] == "point" and not bool(row.get("exact_final", False))
    ]
    centroid_candidates = [row for row in rows if row["role"] == "centroid"]
    print("\nInterpretation gate:")
    if point_candidates:
        best = max(point_candidates, key=lambda row: float(row.get("ari", -1.0)))
        print(
            " found non-final point-length assignments; strongest candidate is "
            f"{best['source']}::{best['key']} with {best['unique']} groups, "
            f"ARI={float(best['ari']):.4f}, intersections={int(best['intersections'])}."
        )
        print(
            " Use this vector as the real-stage parent/coarse grouping before testing "
            "any further partition algorithm."
        )
    elif centroid_candidates:
        print(
            " found centroid-length assignments but no alternative point-length vector. "
            "The next step is to reproduce Leapfrog's centroid-to-point parent mapping "
            "from these decoded centroid labels."
        )
    else:
        print(
            " no alternative point- or centroid-length assignment vector was found. "
            "The decoded payload does not expose the preceding grouping directly; next "
            "test nearest-source vertex/face identity, distance and mesh topology because "
            "the current SPD matrix discards those fields."
        )

    if errors:
        print("\nInspection errors:")
        for error in errors:
            print(f" {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
