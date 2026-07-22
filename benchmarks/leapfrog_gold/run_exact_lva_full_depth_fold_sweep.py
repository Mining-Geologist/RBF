"""Run the full-depth LVA parameter sweep on a categorical fold dataset.

Required environment variables:
- POLATORY_SWEEP_DATASET: CSV containing labeled sample coordinates.
- POLATORY_SWEEP_TREND_OBJ: structural trend mesh in OBJ format.

The default CSV mapping matches Fold_dataset.csv:
- coordinates: xm,ym,zm
- category column: Geology
- inside -> +1, outside -> -1

All strength/range cases are generated even when no Leapfrog oracle mesh exists.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Keep the normal 4 x 10 sweep unless the command overrides it.
os.environ.setdefault("POLATORY_SWEEP_STRENGTHS", "2,3,4,5")
os.environ.setdefault(
    "POLATORY_SWEEP_RANGES",
    "50,100,150,200,250,300,350,400,450,500",
)

import run_exact_lva_full_depth_parameter_sweep as sweep  # noqa: E402

suite = sweep.suite
diagnostic = sweep.diagnostic
full_depth = sweep.full_depth

DATASET_PATH = Path(os.environ.get("POLATORY_SWEEP_DATASET", "")).expanduser()
TREND_PATH = Path(os.environ.get("POLATORY_SWEEP_TREND_OBJ", "")).expanduser()

LABEL_COLUMN = os.environ.get("POLATORY_FOLD_LABEL_COLUMN", "Geology").strip()
XYZ_COLUMNS = tuple(
    item.strip()
    for item in os.environ.get("POLATORY_FOLD_XYZ_COLUMNS", "xm,ym,zm").split(",")
    if item.strip()
)
INSIDE_LABEL = os.environ.get("POLATORY_FOLD_INSIDE_LABEL", "inside").strip().casefold()
OUTSIDE_LABEL = os.environ.get("POLATORY_FOLD_OUTSIDE_LABEL", "outside").strip().casefold()

if len(XYZ_COLUMNS) != 3:
    raise ValueError(
        "POLATORY_FOLD_XYZ_COLUMNS must contain exactly three comma-separated columns"
    )


def _required_path(path: Path, variable: str) -> Path:
    if not str(path):
        raise ValueError(f"{variable} must point to an existing file")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{variable} does not exist: {path}")
    return path


def _parse_padding(spans: np.ndarray) -> np.ndarray:
    text = os.environ.get("POLATORY_FOLD_MODEL_PADDING", "").strip()
    if not text:
        return np.maximum(
            np.full(3, 3.0 * float(suite.SURFACE_RESOLUTION)),
            0.05 * np.asarray(spans, dtype=np.float64),
        )

    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) == 1:
        values *= 3
    if len(values) != 3 or any(value < 0.0 for value in values):
        raise ValueError(
            "POLATORY_FOLD_MODEL_PADDING must be one non-negative number or X,Y,Z"
        )
    return np.asarray(values, dtype=np.float64)


def _read_fold_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(path)
    required = set(XYZ_COLUMNS) | {LABEL_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    points = frame[list(XYZ_COLUMNS)].to_numpy(dtype=np.float64)
    if len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError("Fold dataset coordinates must be non-empty and finite")

    labels = frame[LABEL_COLUMN].astype(str).str.strip().str.casefold()
    known = labels.isin([INSIDE_LABEL, OUTSIDE_LABEL])
    if not bool(known.all()):
        unknown = sorted(labels.loc[~known].unique().tolist())
        raise ValueError(
            f"Unexpected {LABEL_COLUMN} values {unknown}; expected "
            f"{INSIDE_LABEL!r} and {OUTSIDE_LABEL!r}"
        )

    # The structural interpolant uses -1 as its outside/background value.
    indicators = np.where(
        labels.to_numpy() == INSIDE_LABEL,
        1.0,
        -1.0,
    ).astype(np.float64)
    return points, indicators, frame


def _extent_points(frame: pd.DataFrame, samples: np.ndarray) -> np.ndarray:
    groups: list[np.ndarray] = [np.asarray(samples, dtype=np.float64)]
    for columns in (("xb", "yb", "zb"), ("xm", "ym", "zm"), ("xe", "ye", "ze")):
        if set(columns).issubset(frame.columns):
            values = frame[list(columns)].to_numpy(dtype=np.float64)
            if np.all(np.isfinite(values)):
                groups.append(values)
    return np.vstack(groups)


def _configure_paths_and_bounds(
    points: np.ndarray,
    frame: pd.DataFrame,
    trend_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    extent = np.vstack([_extent_points(frame, points), trend_vertices])
    minimum = np.min(extent, axis=0)
    maximum = np.max(extent, axis=0)
    spans = maximum - minimum
    if np.any(spans <= 0.0):
        raise ValueError(f"Degenerate model extent: minimum={minimum}, maximum={maximum}")

    padding = _parse_padding(spans)
    model_min = minimum - padding
    model_max = maximum + padding

    suite.MODEL_MIN = model_min
    suite.MODEL_MAX = model_max
    # The confirmed fix reads this module global while each domain is constructed.
    full_depth._MODEL_MIN_Z = float(model_min[2])

    output_name = os.environ.get(
        "POLATORY_FOLD_OUTPUT_NAME",
        "fold-exact-lva-full-depth-sweep",
    ).strip()
    if not output_name:
        raise ValueError("POLATORY_FOLD_OUTPUT_NAME cannot be empty")
    output_dir = suite.ROOT / "benchmark-results" / output_name
    suite.OUTPUT_DIR = output_dir
    suite.MESH_DIR = output_dir / "meshes"
    suite.PLOT_DIR = output_dir / "overlays"
    diagnostic.DIAGNOSTIC_DIR = output_dir / "basal-diagnostics"
    full_depth.baseline.CSV_DIR = output_dir / "inspection-csv"

    # Synthetic sweep case references are resolved beneath DATA_DIR. This folder
    # contains no Sx_Ry oracle meshes unless the user deliberately adds them.
    suite.DATA_DIR = DATASET_PATH.parent
    return model_min, model_max, padding


def main() -> int:
    dataset_path = _required_path(DATASET_PATH, "POLATORY_SWEEP_DATASET")
    trend_path = _required_path(TREND_PATH, "POLATORY_SWEEP_TREND_OBJ")

    points, indicators, frame = _read_fold_dataset(dataset_path)
    trend_vertices, trend_faces = suite.read_obj(trend_path)
    model_min, model_max, padding = _configure_paths_and_bounds(
        points,
        frame,
        trend_vertices,
    )

    cases = sweep._sweep_cases()
    suite.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "dataset": str(dataset_path),
        "trend_mesh": str(trend_path),
        "xyz_columns": list(XYZ_COLUMNS),
        "label_column": LABEL_COLUMN,
        "inside_label": INSIDE_LABEL,
        "outside_label": OUTSIDE_LABEL,
        "input_points": int(len(points)),
        "inside_points": int(np.count_nonzero(indicators > 0.0)),
        "outside_points": int(np.count_nonzero(indicators < 0.0)),
        "model_bounds_min": model_min.tolist(),
        "model_bounds_max": model_max.tolist(),
        "model_padding": padding.tolist(),
        "surface_resolution": float(suite.SURFACE_RESOLUTION),
        "cases": {},
    }

    print(
        "PROGRESS\tFold dataset sweep: "
        f"{len(points)} midpoint samples "
        f"({report['inside_points']} inside, {report['outside_points']} outside).",
        flush=True,
    )
    print(
        f"PROGRESS\tTrend mesh: {len(trend_vertices)} vertices, "
        f"{len(trend_faces)} triangles.",
        flush=True,
    )
    print(
        f"PROGRESS\tModel bounds: min={model_min.tolist()}, "
        f"max={model_max.tolist()}, padding={padding.tolist()}.",
        flush=True,
    )
    print(
        f"PROGRESS\tRunning {len(cases)} full-depth LVA cases; "
        f"results directory: {suite.OUTPUT_DIR}",
        flush=True,
    )

    for index, case in enumerate(cases, start=1):
        print(
            f"PROGRESS\tCase {index}/{len(cases)}: {case.name}",
            flush=True,
        )
        report["cases"][case.name] = suite.build_case(
            case,
            points,
            indicators,
            trend_vertices,
            trend_faces,
        )
        (suite.OUTPUT_DIR / "metrics.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    report["parameter_response"] = suite.response_metrics(report["cases"])
    (suite.OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    suite.write_summary(report)
    diagnostic._write_cross_case_comparison()
    print(
        f"PROGRESS\tCompleted fold sweep. Summary: "
        f"{suite.OUTPUT_DIR / 'summary.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
