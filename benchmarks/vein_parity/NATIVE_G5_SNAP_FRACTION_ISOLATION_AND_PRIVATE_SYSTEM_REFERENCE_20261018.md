# G5 snap-fraction native controlled comparison — 2026-10-18

**Scope:** owner-ran 5m-resolution variable-thickness vein model, with two otherwise matched runs of 50% and 100% contact snapping. The owner supplied both saved models; we inspected SQLite object graphs and decoded LFData numerical mesh payloads read-only, without executing vendor binaries or loading serialized proprietary classes. No owner native meshes, project content, hole coordinates or other private geological files were committed to this public branch.

## Native settings

- Both cases: vein resolution **5m**, reference resolution **10m**, same five vein observations, same collars/surveys and bounding geometry.
- Controlled setting: `snap_fraction` 1.0 vs 0.5.
- Reference mesh **266 vertices / 463 triangles**, exactly identical indexed geometry across cases.
- Raw/pre-pinchout volume **5,038 vertices / 10,072 triangles**, exactly identical indexed geometry across cases, despite distinct serialized LFData checksums.
- Synthetic pinchout intervals **28/28 identical** in measured-depth from/to values and selected drillhole positions.
- Final HW **3,031 vertices / 5,702 triangles**, **identical indexed geometry and saved result bytes**.
- Final FW **3,011/5,662 at 100%**, **3,010/5,660 at 50%**; significantly changed mesh geometry.
- Final refined solid **719/1,418 at 100%**, **717/1,414 at 50%**; changed mesh geometry.

## Measured real differences

Compare vertical intersections against each final FW on 1,225 regular XY points plus 625 finer central XY positions. On 1,849 positions with equal crossing counts, **FW difference RMSE 0.236272m, max 5.451890m, p95 absolute 0.10051m**. All 1,850 native raw and reference checks match exactly; final HW is exactly identical. Note multiple HW vertical intersections around the center of this model (locally multivalued).

At five real contact positions, 100% snapping intersects the recorded FW elevations within floating tolerance. The 50% result agrees with four of five FW contacts but misses the central FW contact by **+5.451889574486m**, whereas all five HW contacts agree even at 50%. Hence a universal simple '50% of every wall's contact correction' interpretation is invalid for this case. Comparing with a prior uniform-thickness pinchout control where both HW/FW changed further suggests highly geometry-dependent branch/constraint treatment.

Both final refined meshes are watertight; their consistently **negative signed orientation** gives absolute enclosed volumes **10,284.260023m³** (100%) versus **10,115.468019m³** (50%), change **168.792004m³, 1.641% of the 100% volume**. Treat as oriented mesh volumes; no independent voxelized overlap was computed.

**Inference:** Snap fraction affects *downstream final wall/solid construction* in this benchmark, not the reference, raw vein or synthetic pinchout interval locations. Only FW changes here, not HW. This is specific to the model's saved options and potentially locally overlapping surface geometry; it is not a general exact snap algorithm.

## Separate private multi-vein model audit

An additional owner-supplied real multi-vein project was inspected at the **saved object-dependency level only**, without publishing owner geology or mesh data. It contains five component veins; four reference blocks use **the first component vein's hanging-wall mesh object as their custom reference**. This explicitly demonstrates a **wall-to-reference dependency** in a multi-vein system and offers a path to testing reference-relative interpolation on real heterogeneous geometry. This is *not* evidence that the automatic first-vein reference-generation problem, chronology or termination/boolean engine is solved. Proprietary project contents remain private.

## Next analyses

1. Numerical reference-relative interpolation on the private custom-reference child veins, treating the upstream parent wall mesh as a fixed known geometric conditioning surface.
2. Map end-stage branch/collision behavior and saved wall-control locations around the central 50% FW anomaly before hypothesizing an exact snap percentage formula.
3. Test multi-vein system termination separately from custom-reference reuse.

The owner's full **private** numerical result bundle (read-only scripts, 9,565 grid-intersection rows, summary JSON, volume audit and private dependency inventory) was supplied in the original conversation. Research branch only, `main` untouched.