"""Independent vein segment selection and reference-normal wall orientation.

The behavior of multiple intersections OFF is unknown: 'outermost' and
'largest' are competing explicit *hypotheses*, not vendor claims.
No proprietary library or runtime is imported.
"""
from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass(frozen=True)
class Segment:
    start_md: float
    end_md: float

    def __post_init__(self):
        if not np.isfinite([self.start_md, self.end_md]).all() or self.end_md <= self.start_md:
            raise ValueError("Depths must be finite and end > start")


def select_segments(segments, mode="all"):
    """Return contacts for enabled mode or candidate disabled-mode policies."""
    segments = sorted(segments, key=lambda s: s.start_md)
    for a, b in zip(segments, segments[1:]):
        if a.end_md > b.start_md + 1e-9:
            raise ValueError("Overlapping vein segments")
    if mode == "all":
        return segments
    if not segments:
        return []
    if mode == "outermost":
        return [Segment(segments[0].start_md, max(s.end_md for s in segments))]
    if mode == "largest":
        return [max(segments, key=lambda s: s.end_md-s.start_md)]
    raise ValueError("mode must be all, outermost or largest")


def sample_trace(path, measured_depth):
    """Linear interpolation along already-desurveyed (MD,X,Y,Z) trace."""
    q = np.asarray(path, float)
    if q.ndim != 2 or q.shape[1] != 4 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("path must be finite (N,4)")
    if np.any(np.diff(q[:, 0]) <= 0) or measured_depth < q[0,0] or measured_depth > q[-1,0]:
        raise ValueError("Invalid measured depth or path")
    return np.array([np.interp(measured_depth, q[:,0], q[:,i]) for i in (1,2,3)])


def predict_orientations(reference_mesh, path, segments, *,
                         mode="all", invert_reference=False, sample_at="start"):
    """Classify contacts using dot(reference normal, downhole tangent).

    If dot is positive, first contact is HW; otherwise FW.
    Selection of start contact was inspired by empirical owner-exported data
    and is not evidence of source-code identity. Mesh face winding matters.
    """
    chosen = select_segments(segments, mode)
    if not chosen:
        return []
    if sample_at not in ("start", "midpoint", "end"):
        raise ValueError("sample_at must be start, midpoint or end")
    depths = [
        s.start_md if sample_at=="start" else
        s.end_md if sample_at=="end" else
        0.5*(s.start_md+s.end_md)
        for s in chosen
    ]
    probes = np.array([sample_trace(path, md) for md in depths])
    endpoints = [(sample_trace(path, s.start_md), sample_trace(path, s.end_md)) for s in chosen]
    tangent = np.array([b-a for a,b in endpoints], float)
    tangent /= np.linalg.norm(tangent, axis=1)[:,None]
    _, distances, faces = trimesh.proximity.closest_point_naive(reference_mesh, probes)
    normals = np.asarray(reference_mesh.face_normals)[faces]
    if invert_reference:
        normals = -normals
    alignment = np.einsum("ij,ij->i", normals, tangent)
    results = []
    for i, s in enumerate(chosen):
        start_is_hw = bool(alignment[i] > 0)
        results.append({
            "start_md": s.start_md, "end_md": s.end_md,
            "start_wall": "HW" if start_is_hw else "FW",
            "end_wall": "FW" if start_is_hw else "HW",
            "reference_distance_m": float(distances[i]),
            "normal_dot_downhole": float(alignment[i]),
            "sample_xyz": probes[i].tolist()
        })
    return results
