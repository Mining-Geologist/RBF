"""Compare saved Leapfrog SubDomainer tensors with candidate pointwise LVA rules.

Requires helper safe_lf_pickle.py and parse_parquet.py from the accompanying
orientation research kit; no pickle reducers are executed. Original projects
are read-only and must be supplied by their owner.

Example:
 python benchmarks/leapfrog_gold/validate_orientation_saved_domains.py \
  --project LVA_Test2.aproj_data.zip \
  --sub n/maaaaaaaaaaa.lfdata --seeds n/yaaaaaaaaaaa.lfdata \
  --field l/uaaaaaaaaaaa.lfdata --canonical original_input.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_lf(archive, suffix):
    from safe_lf_pickle import decode
    files = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(files) != 1:
        raise ValueError(f"Expected one path with suffix {suffix!r}; found {len(files)}")
    data = archive.read(files[0])
    if b"lfdata-1.0" not in data[:64]:
        raise ValueError("Unexpected lfdata header")
    return decode(data[64:]).state


def load_project(path, subname, seedname, fieldname):
    with zipfile.ZipFile(path) as z:
        sub = read_lf(z, subname)
        seeds = read_lf(z, seedname)["seeds"]
        field = read_lf(z, fieldname) if fieldname else sub["anisotropy_field"]
    trend = field.state if hasattr(field, "state") else field
    mesh, strength, trend_range = trend[0]
    domains = sub["domains"]
    return dict(
        points=np.array(seeds, dtype=float),
        vertices=np.array(mesh.state["location"], dtype=float),
        triangles=np.array(mesh.state["tri"], dtype=int),
        gradients=np.array(mesh.state["gradient"], dtype=float),
        strength=float(strength), trend_range=float(trend_range),
        indices=[np.array(d.state["indices"], dtype=int) for d in domains.values()],
        ids=[int(k) for k in domains],
        reference=np.array([d.state["average_mtx"] for d in domains.values()], dtype=float),
    )


def restore_points(points, csv_path):
    result = points.copy()
    changed = []
    if csv_path is None:
        return result, changed
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty canonical point CSV")
    indexed = "point_index_0based" in rows[0]
    if not indexed and len(rows) != len(result):
        raise ValueError("Full point CSV must contain every input row")
    for rowno, row in enumerate(rows):
        index = int(row["point_index_0based"]) if indexed else rowno
        xyz = np.array([float(row[name]) for name in ("X", "Y", "Z")])
        if not 0 <= index < len(result) or not np.all(np.isfinite(xyz)):
            raise ValueError("Invalid point index or XYZ")
        if np.linalg.norm(xyz - result[index]) > 1.e-8:
            changed.append(index)
        result[index] = xyz
    return result, changed


def tensor_field(dist, normal, strength, trend_range):
    if strength <= 0 or trend_range <= 0:
        raise ValueError("Expected positive strength and range")
    n = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    q = np.exp(-dist / trend_range)
    q[dist >= 4 * trend_range] = 0
    r = 1 + (strength - 1) * q
    tangent = r ** (-1 / 3)
    return (tangent[:, None, None] * np.eye(3)[None]
            + (r ** (2 / 3) - tangent)[:, None, None]
            * n[:, :, None] * n[:, None, :])


def evaluate_saved_domains(data, points, output):
    from scipy.spatial import cKDTree
    tree = cKDTree(data["vertices"])
    dist, indices = tree.query(points)
    second = tree.query(points, k=2)[0]
    tied = np.abs(second[:, 0] - second[:, 1]) <= 1e-8
    field = tensor_field(dist, data["gradients"][indices],
                         data["strength"], data["trend_range"])
    recovered = np.array([field[index].mean(axis=0) for index in data["indices"]])
    err = recovered - data["reference"]
    rows = []
    for d, domain_id in enumerate(data["ids"]):
        ix = data["indices"][d]
        rows.append(dict(
            domain_id=domain_id, point_count=len(ix), ties=int(np.sum(tied[ix])),
            matrix_rmse=float(np.sqrt(np.mean(err[d] ** 2))),
            y_error=float(err[d, 1, 1]), xz_error=float(err[d, 0, 2])))
    output.mkdir(exist_ok=True, parents=True)
    with (output / "per_domain.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = dict(
        input_count=len(points), domain_count=len(rows),
        tie_count=int(tied.sum()),
        domains_without_ties=int(sum(row["ties"] == 0 for row in rows)),
        full_matrix_rmse=float(np.sqrt(np.mean(err**2))),
        yy_rmse=float(np.sqrt(np.mean(err[:, 1, 1]**2))),
        max_abs_error=float(np.max(np.abs(err))),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--sub", required=True, help="Suffix of saved SubDomainer lfdata")
    p.add_argument("--seeds", required=True, help="Suffix of saved StructuralInterpolant lfdata")
    p.add_argument("--field", help="Suffix of separate saved AnisotropyStandard lfdata")
    p.add_argument("--canonical", type=Path, help="Original input XYZ CSV, or sparse row corrections")
    p.add_argument("--helpers-dir", type=Path, help="Directory with safe_lf_pickle.py and parse_parquet.py")
    p.add_argument("--output", type=Path, default=Path("orientation-validation"))
    opts = p.parse_args()
    if opts.helpers_dir:
        sys.path.insert(0, str(opts.helpers_dir.resolve()))
    data = load_project(opts.project, opts.sub, opts.seeds, opts.field)
    points, changed = restore_points(data["points"], opts.canonical)
    result = evaluate_saved_domains(data, points, opts.output)
    result["canonical_coordinate_corrections"] = changed
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
