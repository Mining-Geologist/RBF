"""Reproducible synthetic vein option experiments: no Leapfrog runtime or user data."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from vein_parity_lab import (
    Frame, Hole, Interval, Options, build_vein,
)

FRAME = Frame.from_normal((0, 0, 0), (0, 0, 1))


def make_hole(name, x, y, target=True, *, empty=False, logged_far=False,
              multiple=False, end_vein=False, zshift=0.0):
    trace = np.array([[0, x, y, 10+zshift], [20, x, y, -10+zshift]])
    if empty:
        intervals = []
    elif logged_far:
        intervals = [Interval(0, 4, "WASTE")]
    elif end_vein:
        intervals = [Interval(0, 2, "VEIN"), Interval(2, 20, "WASTE")]
    elif multiple:
        intervals = [Interval(0, 5, "WASTE"), Interval(5, 7, "VEIN"),
                     Interval(7, 13, "WASTE"), Interval(13, 15, "VEIN"),
                     Interval(15, 20, "WASTE")]
    elif target:
        intervals = [Interval(0, 9+zshift, "WASTE"),
                     Interval(9+zshift, 11+zshift, "VEIN"),
                     Interval(11+zshift, 20, "WASTE")]
    else:
        intervals = [Interval(0, 20, "WASTE")]
    return Hole(name, trace, intervals)


def foundation():
    return [make_hole(f"DH_{i}_{j}", i, j)
            for i in (-10, 0, 10) for j in (-10, 0, 10)]


def test_pinchout_flags():
    holes = foundation()
    empty = make_hole("EMPTY", 20, 0, empty=True)
    holes.append(empty)
    off = build_vein(holes, FRAME, Options())
    assert off.contains(np.array([[20., 0., 0.]])).item()
    ignored = build_vein(holes, FRAME, Options(pinch_out=True,
                                                pinch_empty_traces=False))
    assert ignored.pinch_candidates[-1].decision == "completely-empty-disabled"
    active = build_vein(holes, FRAME, Options(pinch_out=True,
                                               pinch_empty_traces=True))
    assert active.pinch_candidates[-1].decision == "eligible"
    uv = FRAME.to_local([[20, 0, 0]])[:, :2]
    hw, fw, delta = active.surfaces(uv)
    assert delta[0] < 0, (hw, fw, delta)
    assert not active.contains(np.array([[20., 0., 0.]])).item()
    assert active.contains(np.array([[0., 0., 0.]])).item()
    excluded = build_vein(holes, FRAME, Options(
        pinch_out=True, pinch_empty_traces=True,
        excluded_pinch_holes=frozenset(["EMPTY"])))
    assert excluded.pinch_candidates[-1].decision == "manual-exclusion"
    assert excluded.contains(np.array([[20., 0., 0.]])).item()


def test_logged_near_and_far():
    holes = foundation()
    far = make_hole("FAR", 20, 0, logged_far=True)
    near = make_hole("NEAR", -20, 0, target=False)
    no_distance = build_vein(holes+[far,near], FRAME,
                             Options(pinch_out=True, dont_pinch_far_logged=False))
    decisions = {d.hole:d.decision for d in no_distance.pinch_candidates}
    assert decisions["FAR"] == "eligible" and decisions["NEAR"] == "eligible"
    guarded = build_vein(holes+[far,near], FRAME,
                         Options(pinch_out=True, dont_pinch_far_logged=True,
                                 logged_distance=3))
    decisions = {d.hole:d.decision for d in guarded.pinch_candidates}
    assert decisions["FAR"] == "logged-interval-too-far", decisions
    assert decisions["NEAR"] == "eligible", decisions


def test_ends_and_multiple():
    holes = foundation()
    holes.append(make_hole("OPEN", 15, 10, end_vein=True))
    m = build_vein(holes, FRAME, Options(
        include_end_wall_points=False, include_end_midpoints=False))
    ends = [c for c in m.contacts if c.hole=="OPEN"]
    assert len(ends)==1, ends
    omitted_mid = [p for p in m.midpoints if p.hole=="OPEN"]
    assert len(omitted_mid)==1 and omitted_mid[0].included is False
    enabled = build_vein(holes, FRAME, Options(
        include_end_wall_points=True, include_end_midpoints=True))
    assert len([c for c in enabled.contacts if c.hole=="OPEN"])==2
    assert [p.included for p in enabled.midpoints if p.hole=="OPEN"]==[True]
    folded = foundation()+[make_hole("MULTI", 18, 12, multiple=True)]
    multiple = build_vein(folded, FRAME, Options(multiple_intersections=True))
    single = build_vein(folded, FRAME, Options(multiple_intersections=False))
    assert len([p for p in multiple.midpoints if p.hole=="MULTI"]) == 2
    assert len([p for p in single.midpoints if p.hole=="MULTI"]) == 1


def test_thickness_and_polarity():
    holes = foundation()
    base = build_vein(holes, FRAME, Options())
    uv = FRAME.to_local([[0,0,0]])[:,:2]
    hw, fw, signed = base.surfaces(uv)
    assert 1.95 < signed[0] < 2.05, signed
    assert hw[0]>0 and fw[0]<0
    maximum = build_vein(holes, FRAME, Options(max_thickness=1.))
    top, bottom, signed = maximum.surfaces(uv)
    assert np.isclose(top[0]-bottom[0], 1., atol=1e-5)
    minimum = build_vein(holes, FRAME, Options(min_thickness=4.))
    top, bottom, _ = minimum.surfaces(uv)
    assert np.isclose(top[0]-bottom[0], 4., atol=1e-5)
    try:
        Options(pinch_out=True,min_thickness=2.)
    except ValueError:
        pass
    else:
        raise AssertionError("Minimum thickness with pinchout must be rejected")


def test_curved_reference():
    holes = [make_hole(f"B{i}C{j}",i,j,zshift=.02*i*j)
             for i in (-10,0,10) for j in (-10,0,10)]
    model = build_vein(holes, FRAME, Options(reference="curved"))
    values = [model.ref(FRAME.to_local([[i,j,0]])[:,:2])[0]
              for i,j in [(-10,-10),(10,10),(-10,10)]]
    assert max(values)-min(values) > 1.0, values


def test_closed_volume(out):
    # A no-pinch, box-capped planar volume should be nonempty and watertight.
    flat = build_vein(foundation(), FRAME, Options())
    mesh = flat.closed_volume_mesh(-14,14,-14,14,nu=38,nv=38,nw=22)
    assert mesh.is_watertight, (mesh.euler_number, mesh.is_winding_consistent)
    assert mesh.volume > 0
    mesh.export(out/"simple_vein.obj")
    return dict(vertices=len(mesh.vertices), triangles=len(mesh.faces),
                watertight=bool(mesh.is_watertight),
                volume=float(mesh.volume))


def make_example(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    holes = foundation()+[make_hole("EMPTY",20,0,empty=True)]
    options = Options(pinch_out=True, pinch_empty_traces=True)
    model = build_vein(holes, FRAME, options)
    top, bottom, signed = model.surface_meshes(-20,24,-15,15,nu=55,nv=39)
    np.savez_compressed(out/"synthetic_pinchout_surfaces.npz",
                        hangingwall_xyz=top, footwall_xyz=bottom,
                        signed_thickness=signed)
    with (out/"contacts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hole","md","x","y","z","wall","source"])
        for c in model.contacts:
            writer.writerow([c.hole,c.md,*c.xyz,c.wall,c.source])
    with (out/"pinchout_decisions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hole","decision","intersection_md","logged_gap_md"])
        for c in model.pinch_candidates:
            writer.writerow([c.hole,c.decision,c.md,c.distance_to_logged_interval])
    mesh = model.closed_volume_mesh(-19,24,-14,14,nu=58,nv=42,nw=25)
    assert mesh.is_watertight, mesh.euler_number
    mesh.export(out/"pinchout_vein.obj")
    return {"pinchout_mesh_watertight":bool(mesh.is_watertight),
            "pinchout_mesh_volume":float(mesh.volume),
            "pinchout_mesh_triangles":int(len(mesh.faces))}


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    test_pinchout_flags()
    test_logged_near_and_far()
    test_ends_and_multiple()
    test_thickness_and_polarity()
    test_curved_reference()
    first = test_closed_volume(out)
    extra = make_example(out)
    summary = {"status":"PASS", "test_groups":6,
               "reconstruction_not_native_parity":True,
               "simple_volume":first,"synthetic_pinchout":extra}
    (out/"test_results.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path, default=Path("vein-lab-output"))
    args=p.parse_args()
    run(args.out)
