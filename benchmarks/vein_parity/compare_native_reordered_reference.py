"""Compare reference-surface OBJ coordinates and identical indexed-triangle topology.

No proprietary sample OBJ files are distributed. Usage:
    python compare_native_reordered_reference.py original.obj reordered.obj
"""
import argparse
import json
import numpy as np


def read_obj(path):
    v, f = [], []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                v.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                f.append([int(x.split("/")[0]) for x in parts[1:]])
    return np.asarray(v, float), np.asarray(f, dtype=np.int64)


def compare(original, reordered):
    va, fa = read_obj(original)
    vb, fb = read_obj(reordered)
    same_shapes = va.shape == vb.shape and fa.shape == fb.shape
    same_faces = bool(same_shapes and np.array_equal(fa, fb))
    out = {
        "original_vertices": len(va), "reordered_vertices": len(vb),
        "original_triangles": len(fa), "reordered_triangles": len(fb),
        "same_triangle_indices_in_same_order": same_faces
    }
    if va.shape == vb.shape:
        delta = va - vb
        norm = np.linalg.norm(delta, axis=1)
        out.update({
            "max_axis_coordinate_difference_m": float(np.max(np.abs(delta))),
            "max_corresponding_vertex_distance_m": float(norm.max()),
            "mean_corresponding_vertex_distance_m": float(norm.mean()),
            "rms_corresponding_vertex_distance_m": float(np.sqrt(np.mean(norm**2))),
            "all_vertices_equal_at_8_decimal_places": bool(
                np.array_equal(np.round(va,8), np.round(vb,8))),
        })
        if same_faces:
            out["upper_bound_on_symmetric_hausdorff_distance_m"] = float(norm.max())
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original")
    p.add_argument("reordered")
    a = p.parse_args()
    print(json.dumps(compare(a.original,a.reordered),indent=2))
