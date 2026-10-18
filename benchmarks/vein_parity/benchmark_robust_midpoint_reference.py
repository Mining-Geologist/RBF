"""Owner-local offline benchmark: compare midpoint-built references against saved OBJ.

The training input is ONLY midpoint_xyz.csv, never the reference OBJ.
Meshes are opened ONLY after the model has been fitted for scoring.

Usage:
 python benchmark_robust_midpoint_reference.py --directory LOCAL_CASES --output results.csv
where each subfolder contains midpoint_xyz.csv (columns x,y,z)
and Reference_Surface.obj. No Leapfrog executable is required.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import trimesh
from robust_midpoint_reference import fit_robust_reference


def measure(directory):
    results=[]
    for folder in sorted(Path(directory).iterdir()):
        if not folder.is_dir():
            continue
        input_file=folder/"midpoint_xyz.csv"
        target_file=folder/"Reference_Surface.obj"
        if not input_file.is_file() or not target_file.is_file():
            continue
        midpoints=np.loadtxt(input_file,delimiter=",",skiprows=1,ndmin=2)
        try:
            model=fit_robust_reference(midpoints)
        except ValueError as e:
            results.append({"case":folder.name,"status":"not_heightfield",
                "n_inputs":len(midpoints),"n_inliers":0,"outliers":0,
                "rmse_interior_m":"","rmse_full_m":"","reason":str(e)})
            continue
        # Native reference is touched ONLY after the pointcloud fit.
        mesh=trimesh.load(target_file,process=False)
        vertices=np.asarray(mesh.vertices)
        errors=model.height(vertices[:,:2])-vertices[:,2]
        interior=np.all(np.abs(vertices[:,:2])<=98.,axis=1)
        results.append({"case":folder.name,"status":"measured",
            "n_inputs":len(midpoints),"n_inliers":len(model.inlier_sites),
            "outliers":len(model.rejected),
            "rmse_interior_m":float(np.sqrt(np.mean(errors[interior]**2))),
            "rmse_full_m":float(np.sqrt(np.mean(errors**2))),"reason":""})
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--directory",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=Path("reference_scores.csv"))
    a=ap.parse_args()
    results=measure(a.directory)
    if not results:
        ap.error("No matching case subdirectories")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="") as fd:
        writer=csv.DictWriter(fd,fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for row in results:
        print(row)


if __name__=="__main__":
    main()
