"""Score a frozen MQ1 reference-relative wall surrogate against local saved OBJ files.

Inputs must come from owner-exported Vein_Benchmark_Extracted directory.
The reference OBJ is a GIVEN conditioning geometry, NOT constructed by code.
Geological contact/pinchout XYZ CSV control points are the only fitted data.
Saved HW/FW vertices are accessed only after fitting, for RMSE scoring.

Usage: python benchmark_reference_relative_walls.py --data /path/to/extracted
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
from reference_relative_wall_model import ReferenceRelativeWalls

CASES=("01_FLAT_BASELINE","02_PINCHOUT_A_empty_OFF",
       "02_PINCHOUT_B_empty_ON","03_CURVED_VARIABLE_THICKNESS",
       "04_MULTIPLE_INTERSECTIONS","05_END_OF_HOLE")


def benchmark(root:Path):
    rows=[]
    for case in CASES:
        folder=root/case
        reference=trimesh.load(folder/"Reference_Surface.obj",process=False)
        controls={}
        for side in ("Hangingwall","Footwall"):
            chunks=[]
            for source in ("Observed","Pinchout"):
                path=folder/(f"{source}_{side.lower()}_points.csv")
                if path.exists():
                    chunks.append(pd.read_csv(path).iloc[:,:3].to_numpy(float))
            controls[side]=np.vstack(chunks)
        try:
            model=ReferenceRelativeWalls(
                reference.vertices,controls["Hangingwall"],controls["Footwall"],
                shape_m=1.,reference_faces=reference.faces
            )
        except ValueError as exc:
            rows.append(dict(case=case,side="N/A",scope="not-heightfield",
                             method="skipped-fold",n=0,rmse_m="",reason=str(exc)))
            continue
        for side in ("Hangingwall","Footwall"):
            # Native mesh touched only after independent prediction parameters fit.
            native=trimesh.load(folder/(side+"_Interpolated.obj"),process=False)
            p=native.vertices
            for label,good in (
                ("all",np.ones(len(p),bool)),
                ("inside98",np.all(np.abs(p[:,:2])<=98.,axis=1))
            ):
                for method,relative in (("absolute",False),("ref_offset",True)):
                    predicted=model.evaluate(p[good,:2],relative=relative)
                    err=predicted["HW" if side=="Hangingwall" else "FW"]-p[good,2]
                    rows.append(dict(case=case,side=side,scope=label,
                         method=method,n=int(good.sum()),
                         rmse_m=float(np.sqrt(np.mean(err**2))),reason=""))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",required=True,type=Path)
    ap.add_argument("--output",type=Path,default=Path("reference_relative_scores.csv"))
    args=ap.parse_args()
    results=benchmark(args.data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["case","side","scope","method","n","rmse_m","reason"])
        writer.writeheader()
        writer.writerows(results)
    for r in results:
        if r["case"]=="03_CURVED_VARIABLE_THICKNESS" and r["scope"]=="inside98":
            print(r["side"],r["method"],f"{r['rmse_m']:.6f} m")


if __name__=="__main__":main()
