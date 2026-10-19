"""Orientation parity audit from a known triangulated reference and paired HW/FW contacts.

No vendor code or bundled owner data. Mesh winding and proximity must be
checked before transferring the result to an unrelated geological model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def compare(reference_mesh, hw_xyz, fw_xyz, normal_sign=-1):
    hw = np.asarray(hw_xyz, dtype=float)
    fw = np.asarray(fw_xyz, dtype=float)
    if hw.ndim != 2 or hw.shape[1] != 3 or fw.shape != hw.shape:
        raise ValueError("HW and FW must be paired XYZ arrays, equal shape (N,3)")
    midpoint = (hw+fw)/2
    _, distance, tri = trimesh.proximity.closest_point_naive(reference_mesh, midpoint)
    observed = np.sign(hw[:,2]-fw[:,2])
    predicted = normal_sign * np.sign(reference_mesh.face_normals[tri,2])
    valid = (observed!=0)&(predicted!=0)
    return {
        "n_valid": int(np.sum(valid)),
        "n_correct": int(np.sum(valid & (observed==predicted))),
        "rows": [
            {"row":i,"x":float(p[0]),"y":float(p[1]),"z":float(p[2]),
             "reference_distance_m":float(distance[i]),
             "observed_contact_polarity":int(observed[i]),
             "predicted_contact_polarity":int(predicted[i]),
             "match":bool(valid[i] and observed[i]==predicted[i])}
            for i,p in enumerate(midpoint)
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",required=True)
    parser.add_argument("--hangingwall",required=True)
    parser.add_argument("--footwall",required=True)
    parser.add_argument("--sign",type=int,choices=[-1,1],default=-1)
    parser.add_argument("--output")
    args=parser.parse_args()
    ref=trimesh.load_mesh(args.reference,process=False)
    hw=np.loadtxt(args.hangingwall,delimiter=",",skiprows=1).reshape(-1,3)
    fw=np.loadtxt(args.footwall,delimiter=",",skiprows=1).reshape(-1,3)
    out=json.dumps(compare(ref,hw,fw,args.sign),indent=2)
    if args.output:Path(args.output).write_text(out)
    print(out)
