# Vein reverse-engineering lab — Leapfrog Geo 2026.1 behaviours and limits

Date: 2026-10-17. Branch: research/vein-pinchout-parity-20261017.

**Scope:** independent modelling behaviour reconstruction for interoperability/research, not a recovered native algorithm. Absolutely no copied vendor code, binaries, licensed data or decompiled source has been added to this public branch. All tests use synthetic hole paths and intervals; no Leapfrog executable or licence service is involved.

## What is directly observable in the public manual

Official primary reference:
https://help.seequent.com/Geo/2026.1/en-GB/Content/geo-models/veins.htm

1. Veins are made of separate hangingwall, footwall and reference surfaces, segments, midpoints and a boundary.
2. Contact-derived veins use the midpoint cloud and wall information to initialize a **curved best-fit reference**; veins created from non-drilling objects initialize with a plane. A custom mesh can also be a reference.
3. Drillhole segment types include auto/user HW, FW and excluded. A segment entirely enclosed within the vein lithology is excluded from wall contact generation. Users may manually change segment orientation. Midpoints also have auto/user included/excluded statuses.
4. Hole-end wall contacts can be included/excluded. Hole-end midpoint points can remain in the data but be excluded from reference interpolation.
5. Multiple intersections of one vein in a drillhole can be enabled, important for folded deposits. The exact order and heuristic of automatic segment categorization are not documented.
6. Pinchout adds **synthetic outside intervals**, reversing hangingwall/footwall orientation relative to interior veins, resulting in two wall surfaces **crossing** and a zero-thickness boundary. This is not merely deleting faces or setting thickness to zero after interpolation.
7. Options: global enable; allow completely unlogged traces; don't pinchout in logged holes if logged lithology is too far from the modeled wall intersection; exclude individual pinchout segments manually.
8. With pinchout enabled, minimum-thickness is disabled. Pinchout constraints apply before maximum thickness. Snapping to contact data happens **after** thickness computation.
9. Boundary filtering and snapping separately support Off/All/Drilling-only/Custom/Inherit, with spatial filters. These have not yet been reproduced.
10. Vein systems require priority/chronology and per-vein interactions, including termination against HW/FW of higher-priority veins. Single system output volume versus multiple individual output volumes is configurable.
System reference: https://help.seequent.com/Geo/2026.1/en-GB/Content/geo-models/vein-system.htm

## Private GitHub file inspection, not source disassembly

User-owned private repository: Mining-Geologist/LF_SC, main:
- Leapfrog_2026.1/bin/client/VeinSurfacingWidgets.ui confirms checkboxes/fields: enable_maximum, maximum_thickness, minimum_thickness, pinch_out, pinch_out_on_empty_traces, dont_pinch_out_on_empty_intersections, distance_to_vein.
- VeinSegmentsDlg.ui confirms: include_points_at_ends, include_midpoints_at_ends, multiple_intersections.
- VeinReferenceSurfaceDlg.ui confirms: curved/planar/custom reference types and planar dip/azimuth controls.
- VeinBoundaryPlaneDlg.ui: reference plane location, dip and dip azimuth.
- VeinInteractionsWidgets.ui: ranked priority and interaction relationships.
- VeinContactSourceWidget.ui: distinct HW/FW input source slots.

The primary Leapfrog.zpa in that repository is an **LFS pointer, not the 51.9MB binary payload**. We cannot honestly claim to have inspected its internal vein geometry functions from the GitHub text pointer. Actual exact interpolation equations, auto-orientation choices, regularization, snapping, and meshing remain unverified.

## Independent executable approximation

Code: benches/vein_parity/vein_parity_lab.py (actual location: benchmarks/vein_parity/vein_parity_lab.py)

Input: measured-depth drillhole trajectories and intervals. Convert segments to reference u/v/w coordinates, extract paired contacts and midpoints, estimate planar or thin-plate-spline curved reference, fit separate scalar 2D offsets for HW and FW, then use the preliminary surfaces to find intersections of non-vein drillholes. On eligible holes, add reversible synthetic outside HW/FW constraints to **invert signed wall separation**. Refit wall fields. Form solid only where signed gap is positive, apply min/max thickness rules, and extract an OBJ volume with marching cubes and a negative ghost-cell halo for reliable closure.

**Important numerical hypotheses (not confirmed native behavior):** thin-plate spline in local 2D reference coordinates; projection along one fixed normal for curved reference; selection of largest segment when multiple intersections disabled; synthetic outside interval width = 0.15*median observed segment projected thickness; contact distance measured along hole MD; thresholding and symmetric thickness offset; marching-cubes implementation. Leapfrog's actual engine may differ.

This implementation **cannot** yet represent a multiply-valued overturned/folded reference in global u/v, nor simulate custom reference mesh normals, native snapping, data boundary filtering, manual vein-segment orientations, mesh resolution/adaptation, or vein-system boolean chronology. It is a **functioning limited research model**, not "all options matched".

## Validation status

CI https://github.com/Mining-Geologist/RBF/actions/runs/39012370339 passed six synthetic test groups:
- eligible/disabled/excluded completely empty trace pinchouts;
- near/far logged-hole distance and eligibility;
- start/end contact and midpoint inclusion; multiple target intersections;
- wall polarity, min/max thickness compatibility;
- curved midpoint reference behaviour;
- watertight positive-volume clipped simple and pinchout 3D meshes.

Results:
- no-pinch clipped example: 4,618 vertices, 9,232 triangles, watertight, positive volume 1,540.8845.
- pinchout example: 11,116 triangles, watertight, positive volume 1,935.1752.
Both are measured against **synthetic mathematical constraints**, not against Leapfrog-exported outputs.

## Next parity benchmark

The next highest-information evidence is a *matched* dataset of (drillhole paths+intervals; exact option state; the separate exported HW mesh, FW mesh, reference mesh, pinchout segments and final volume). Without it, we can verify geological consistency and rules but not quantify geometric equivalence. For testing, systematically toggle **one control per identical data input** and compute wall-to-wall distances, equal-location signed-thickness differences, contact-fit error, pinchout location error, mesh Hausdorff statistics, Dice/IoU volume overlap, watertightness and topology. Any use of an installed Leapfrog instance would be a separate user-authorized step; it is not required to use this independent prototype.
