"""Run the automatic-domain oracle comparison with robust CSV schema detection.

This wrapper fixes decoded benchmark CSVs whose coordinate headers differ in case,
spacing, punctuation, BOM encoding, or naming convention.  It patches only the input
loader and delegates all clustering and metric calculations to
``compare_automatic_domains_to_oracle.py``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import compare_automatic_domains_to_oracle as comparison


def _normalise_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lstrip("\ufeff").strip().lower())


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if len(frame.columns) > 1:
        return frame
    # A single parsed column often means the file uses semicolons or tabs.
    sniffed = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    return sniffed if len(sniffed.columns) > len(frame.columns) else frame


def _resolve_column(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    description: str,
    path: Path,
) -> str:
    normalised: dict[str, list[str]] = {}
    for column in frame.columns:
        normalised.setdefault(_normalise_header(column), []).append(str(column))

    for alias in aliases:
        matches = normalised.get(_normalise_header(alias), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"{path} has ambiguous {description} columns for alias {alias!r}: {matches}."
            )

    raise ValueError(
        f"Could not identify the {description} column in {path}. "
        f"Available columns: {list(frame.columns)!r}."
    )


def _load_oracle_inputs(
    decoded_root: Path,
    case_name: str,
) -> tuple[np.ndarray, np.ndarray, Path]:
    points_path = decoded_root / "Used Data.csv"
    mesh_path = decoded_root / "Reference Mesh.obj"
    labels_path = decoded_root / case_name / "point_clusters.csv"
    missing = [
        path
        for path in (points_path, mesh_path, labels_path)
        if not path.is_file()
    ]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required decoded benchmark files are missing:\n{details}")

    points_frame = _read_csv_flexible(points_path)
    x_column = _resolve_column(
        points_frame,
        ("xe", "x", "easting", "east", "xcoord", "coordx", "xcoordinate", "pointx"),
        description="X/easting coordinate",
        path=points_path,
    )
    y_column = _resolve_column(
        points_frame,
        ("ye", "y", "northing", "north", "ycoord", "coordy", "ycoordinate", "pointy"),
        description="Y/northing coordinate",
        path=points_path,
    )
    z_column = _resolve_column(
        points_frame,
        ("ze", "z", "elevation", "elev", "rl", "zcoord", "coordz", "zcoordinate", "pointz"),
        description="Z/elevation coordinate",
        path=points_path,
    )
    coordinate_columns = [x_column, y_column, z_column]
    points = points_frame[coordinate_columns].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{points_path} contains non-finite coordinates.")

    labels_frame = _read_csv_flexible(labels_path)
    label_column = _resolve_column(
        labels_frame,
        ("cluster", "clusterid", "clusternumber", "domain", "domainid", "rawdomainid"),
        description="Leapfrog cluster label",
        path=labels_path,
    )
    oracle_labels = pd.to_numeric(labels_frame[label_column], errors="raise").to_numpy(
        dtype=np.int64
    )
    if len(oracle_labels) != len(points):
        raise ValueError(
            f"Point/label length mismatch: {len(points)} points versus "
            f"{len(oracle_labels)} labels. Coordinate columns were {coordinate_columns!r}; "
            f"label column was {label_column!r}."
        )

    print(
        f"Decoded CSV schema: coordinates={coordinate_columns}; label={label_column!r}",
        flush=True,
    )
    return points, oracle_labels, mesh_path


comparison._load_oracle_inputs = _load_oracle_inputs


if __name__ == "__main__":
    raise SystemExit(comparison.main())
