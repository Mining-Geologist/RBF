"""Find an overlap-free oblique 2D chart of a triangulated reference surface.
Independent geometry utility; not native Leapfrog code.
Dependencies: numpy, scipy, shapely>=2, trimesh.
"""
import argparse
import json
import numpy as np
import trimesh
from scipy.optimize import linprog
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def axes_from_direction(direction):
    n = np.asarray(direction, float)
    n /= np.linalg.norm(n)
    helper = np.eye(3)[np.argmin(abs(n))]
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(n, u), n])


def feasible_chart_normal(face_normals):
    """Find any n having positive dot product with every face normal."""
    faces = np.asarray(face_normals, float)
    fit = linprog(np.zeros(3), A_ub=-faces, b_ub=-np.ones(len(faces)),
                  bounds=[(None, None)]*3, method="highs")
    if not fit.success:
        return None
    n = fit.x/np.linalg.norm(fit.x)
    return n if np.min(faces @ n) > 1e-9 else None


def projected_overlap(mesh, xy_basis, eps=1e-7):
    uv = mesh.vertices @ xy_basis
    polygons = [Polygon(uv[f]) for f in mesh.faces]
    tree = STRtree(polygons)
    pairs = 0
    area = 0.0
    for i, pi in enumerate(polygons):
        if pi.area <= eps:
            continue
        for j in tree.query(pi):
            j = int(j)
            if j <= i or np.intersect1d(mesh.faces[i], mesh.faces[j]).size:
                continue
            ai = pi.intersection(polygons[j]).area
            if ai > eps:
                pairs += 1
                area += ai
    return {"overlapping_triangle_pairs": pairs, "overlap_area": float(area)}


def audit(mesh):
    if not mesh.is_winding_consistent:
        raise ValueError("Inconsistent face winding")
    horizontal = projected_overlap(mesh, np.eye(3)[:, :2])
    result = {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
              "euler_number": int(mesh.euler_number),
              "xy_projection": horizontal}
    n = feasible_chart_normal(mesh.face_normals)
    if n is None:
        result.update({"hemisphere_exists": False, "oblique_chart_valid": False})
        return result
    oblique = projected_overlap(mesh, axes_from_direction(n)[:, :2])
    result.update({"hemisphere_exists": True, "oblique_normal": n.tolist(),
                   "minimum_face_normal_dot": float(np.min(mesh.face_normals @ n)),
                   "oblique_projection": oblique,
                   "oblique_chart_valid": oblique["overlapping_triangle_pairs"] == 0})
    return result


def mirror_distance(mesh, *, seed=42, samples=900):
    """True point-to-triangle reflection distance (not nearest mesh vertex)."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        pts = trimesh.sample.sample_surface(mesh, samples)[0]
    finally:
        np.random.set_state(state)
    pts[:, 1] *= -1
    _, distance, _ = trimesh.proximity.closest_point_naive(mesh, pts)
    interior = ((abs(pts[:, 0]) < 80) & (abs(pts[:, 1]) < 60) &
                (pts[:, 2] > 20) & (pts[:, 2] < 180))
    return {"mean_m": float(distance.mean()),
            "median_m": float(np.median(distance)),
            "p95_m": float(np.percentile(distance, 95)),
            "interior_n": int(interior.sum()),
            "interior_mean_m": float(distance[interior].mean()) if interior.any() else None}


def midpoint_mirror_invariance(xyz):
    p = np.asarray(xyz, float)
    reflected = p.copy()
    reflected[:, 1] *= -1
    rows = lambda a: sorted(map(tuple, np.round(a, 8)))
    return rows(p) == rows(reflected)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference_obj")
    ap.add_argument("--midpoints", help="Optional CSV columns x,y,z")
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()
    mesh = trimesh.load(args.reference_obj, process=False, force="mesh")
    out = audit(mesh)
    if args.midpoints:
        pts = np.loadtxt(args.midpoints, delimiter=",", skiprows=1)
        out["midpoint_y_mirror_invariant"] = midpoint_mirror_invariance(pts)
    if args.mirror:
        out["mirror_distance"] = mirror_distance(mesh)
    print(json.dumps(out, indent=2))
