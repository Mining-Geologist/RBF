"""Run the WolfPass oracle-label Leapfrog parity benchmark locally.

This script requires the private benchmark files but contains no benchmark data.
It verifies exact post-cluster construction, fits Polatory, generates an OBJ, and
prints symmetric surface-distance and volume metrics against Leapfrog.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import polatory
from polatory import three as p3


def read_obj_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append(
                    [int(value.split("/")[0]) - 1 for value in line.split()[1:4]]
                )
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def surface_metrics(candidate_path: Path, reference_path: Path) -> dict[str, float]:
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError("surface comparison requires pyvista") from error

    candidate = pv.read(candidate_path).extract_surface().triangulate().clean()
    reference = pv.read(reference_path).extract_surface().triangulate().clean()
    c_to_r = np.abs(candidate.compute_implicit_distance(reference)["implicit_distance"])
    r_to_c = np.abs(reference.compute_implicit_distance(candidate)["implicit_distance"])
    distances = np.concatenate([c_to_r, r_to_c])
    return {
        "symmetric_mean_distance": float(np.mean(distances)),
        "symmetric_rmse_distance": float(np.sqrt(np.mean(distances**2))),
        "symmetric_p95_distance": float(np.quantile(distances, 0.95)),
        "symmetric_max_distance": float(np.max(distances)),
        "candidate_volume": float(candidate.volume),
        "reference_volume": float(reference.volume),
        "relative_volume_error": float(
            abs(candidate.volume - reference.volume) / max(reference.volume, 1e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="S5_R100")
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--trend-mesh", type=Path, required=True)
    parser.add_argument("--decoded", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--output", type=Path, default=Path("polatory_oracle.obj"))
    parser.add_argument("--blend-power", type=float, default=1.0)
    parser.add_argument("--fit-tolerance", type=float, default=0.028722813232690145)
    parser.add_argument("--resolution", type=float, default=25.0)
    args = parser.parse_args()

    strength = float(args.case[1])
    trend_range = float(args.case.split("R")[1])
    base_range = 400.0
    sill = 100.0

    frame = pd.read_csv(args.points)
    points = frame[["xe", "ye", "ze"]].to_numpy(float)
    values = frame["SDF"].to_numpy(float)
    labels = pd.read_csv(
        args.decoded / args.case / "point_clusters.csv"
    )["cluster"].to_numpy(np.int64)
    vertices, faces = read_obj_triangles(args.trend_mesh)

    trend_input = polatory.StructuralTrendInput3(
        vertices, faces, strength, trend_range
    )
    model = p3.Model(p3.CovSpheroidal3([sill, base_range]), 0)
    builder = polatory.LabeledStructuralDomainBuilder3(base_range=base_range)
    domains = builder.build_from_inputs(
        points,
        labels,
        [trend_input],
        model_parameters=[sill, base_range],
    )

    expected = pd.read_csv(
        args.decoded / args.case / "recovered_local_domains.csv"
    ).set_index("cluster")
    support_frame = pd.read_csv(
        args.decoded / args.case / "support_membership.csv"
    )
    for domain in builder.diagnostics_ or []:
        row = expected.loc[domain.label]
        expected_support = set(
            support_frame.query("cluster == @domain.label")["point_index"].astype(int)
        )
        if set(domain.support_indices) != expected_support:
            raise RuntimeError(f"support mismatch in cluster {domain.label}")
        if abs(domain.internal_radius - row.internal_radius) > 1e-7:
            raise RuntimeError(f"radius mismatch in cluster {domain.label}")

    interpolant = polatory.StructuralInterpolant3(
        model,
        outside_value=-1.0,
        blend_power=args.blend_power,
    )
    interpolant.fit(
        points,
        values,
        domains,
        tolerance=args.fit_tolerance,
        max_iter=100,
    )
    predictions = interpolant.evaluate(points)
    errors = predictions - values
    print("training_rmse", float(np.sqrt(np.mean(errors**2))))
    print("training_max_abs", float(np.max(np.abs(errors))))

    bbox = p3.Bbox(
        np.array([[444600.0, 492600.0, 2000.0]]),
        np.array([[445900.0, 494600.0, 3600.0]]),
    )
    field = polatory.StructuralRbfFieldFunction(interpolant)
    mesh = polatory.Isosurface(bbox, args.resolution, np.eye(3)).generate(
        field, isovalue=0.0, refine=1
    )
    mesh.export_obj(str(args.output))
    print("output", args.output)

    if args.reference_result is not None:
        for name, value in surface_metrics(args.output, args.reference_result).items():
            print(name, value)


if __name__ == "__main__":
    main()
