"""Independent data-only ambiguity probe for curved/folded vein references.

Usage:
    python midpoint_orientation_identifiability.py --midpoints MIDPOINTS.csv \
        --reference Reference_Surface.obj [--hangingwall HW.csv --footwall FW.csv]

All geometry comes from user-supplied local files. No vendor code is loaded.
The post-hoc native chart is deliberately used ONLY for an ambiguity witness,
NOT as a legitimate input-only reference-surface prediction.
Dependencies: numpy, scipy, trimesh.
"""
import argparse
import json
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import linprog
import trimesh


def symmetric_points(points, axis=1, ndigits=7):
    a = np.asarray(points, float)
    b = a.copy()
    b[:, axis] *= -1
    return sorted(map(tuple, np.round(a, ndigits))) == sorted(map(tuple, np.round(b, ndigits)))


def chart_axes(normal):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    helper = np.eye(3)[np.argmin(abs(n))]
    u = np.cross(n, helper)
    u = u / np.linalg.norm(u)
    return np.column_stack((u, np.cross(n, u), n))


def any_native_chart_normal(mesh):
    nn = mesh.face_normals
    fit = linprog(np.zeros(3), A_ub=-nn, b_ub=-np.ones(len(nn)),
                  bounds=[(None, None)] * 3, method='highs')
    return fit.x / np.linalg.norm(fit.x) if fit.success else None


def loocv(pts, n, smoothing=1, scale=40):
    a = (pts - pts.mean(axis=0)) @ chart_axes(n)
    uv = a[:, :2] / scale
    h = a[:, 2]
    errors = []
    for i in range(len(a)):
        take = np.arange(len(a)) != i
        model = RBFInterpolator(uv[take], h[take], kernel='thin_plate_spline',
                                degree=1, smoothing=smoothing)
        errors.append(float(model(uv[i:i+1])[0] - h[i]))
    return float(np.linalg.norm(errors) / np.sqrt(len(a)))


def mesh_fit_rmse(pts, mesh, n, smoothing=1, scale=40):
    axes = chart_axes(n)
    origin = pts.mean(axis=0)
    p = (pts - origin) @ axes
    v = (mesh.vertices - origin) @ axes
    model = RBFInterpolator(p[:, :2] / scale, p[:, 2],
                            kernel='thin_plate_spline', degree=1,
                            smoothing=smoothing)
    return float(np.sqrt(np.mean((model(v[:, :2] / scale) - v[:, 2])**2)))


def nearest_mesh_distances(mesh, points):
    _, distances, _ = trimesh.proximity.closest_point_naive(mesh, points)
    return np.asarray(distances)


def compare(midpoints, reference, hangingwall=None, footwall=None, smoothing=1):
    pts = np.asarray(midpoints, float)
    n = any_native_chart_normal(reference)
    if n is None:
        raise ValueError('Reference has no common open hemisphere; no global chart witness')
    mirror_n = n.copy()
    mirror_n[1] *= -1
    mirrored_vertices = reference.vertices.copy()
    mirrored_vertices[:, 1] *= -1
    mirrored = trimesh.Trimesh(vertices=mirrored_vertices, faces=reference.faces,
                               process=False)
    d = nearest_mesh_distances(reference, pts)
    dm = nearest_mesh_distances(mirrored, pts)
    du = nearest_mesh_distances(mirrored, reference.vertices)
    _, _, pca = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    _, _, pca_ref = np.linalg.svd(reference.vertices - reference.vertices.mean(axis=0),
                                 full_matrices=False)
    result = {
        'points': len(pts), 'faces': len(reference.faces),
        'midpoints_y_mirror_symmetric': symmetric_points(pts),
        'posthoc_native_chart_normal': n.tolist(),
        'posthoc_mirrored_chart_normal': mirror_n.tolist(),
        'midpoint_pca_vs_reference_pca_angle_deg': float(
            np.degrees(np.arccos(np.clip(abs(pca[-1] @ pca_ref[-1]), -1, 1)))),
        'chart_orientation_acute_separation_deg': float(
            np.degrees(np.arccos(np.clip(abs(n @ mirror_n), -1, 1)))),
        'native_face_alignment_min': float(np.min(reference.face_normals @ n)),
        'mirrored_orientation_native_face_alignment_min': float(
            np.min(reference.face_normals @ mirror_n)),
        'midpoint_closest_to_native_rmse_m': float(np.linalg.norm(d) / np.sqrt(len(d))),
        'midpoint_closest_to_mirror_rmse_m': float(np.linalg.norm(dm) / np.sqrt(len(dm))),
        'midpoint_rmse_difference_m': float(
            np.linalg.norm(d)/np.sqrt(len(d)) - np.linalg.norm(dm)/np.sqrt(len(dm))),
        'native_vertices_closest_to_mirror_mean_m': float(np.mean(du)),
        'native_vertices_closest_to_mirror_rmse_m': float(
            np.linalg.norm(du)/np.sqrt(len(du))),
        'native_chart_midpoint_loocv_rmse_m': loocv(pts, n, smoothing),
        'mirrored_chart_midpoint_loocv_rmse_m': loocv(pts, mirror_n, smoothing),
        'native_chart_mesh_normal_rmse_m': mesh_fit_rmse(pts, reference, n, smoothing),
        'mirrored_chart_mesh_normal_rmse_m': mesh_fit_rmse(
            pts, reference, mirror_n, smoothing)
    }
    if hangingwall is not None and footwall is not None:
        all_contacts = np.vstack((hangingwall, footwall))
        result.update({
            'unlabelled_contacts_y_mirror_symmetric': symmetric_points(all_contacts),
            'hangingwall_labels_y_mirror_symmetric': symmetric_points(hangingwall),
            'footwall_labels_y_mirror_symmetric': symmetric_points(footwall)
        })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--midpoints', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--hangingwall')
    p.add_argument('--footwall')
    p.add_argument('--smoothing', type=float, default=1.0)
    args = p.parse_args()
    pts = np.loadtxt(args.midpoints, delimiter=',', skiprows=1)
    mesh = trimesh.load(args.reference, process=False, force='mesh')
    hw = np.loadtxt(args.hangingwall, delimiter=',', skiprows=1) if args.hangingwall else None
    fw = np.loadtxt(args.footwall, delimiter=',', skiprows=1) if args.footwall else None
    print(json.dumps(compare(pts, mesh, hw, fw, args.smoothing), indent=2))


if __name__ == '__main__':
    main()
