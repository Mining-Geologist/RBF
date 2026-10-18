# Native vein pinchout segment generation — 2026-10-18

**Status:** New direct saved-project graph evidence and exact numerical controls. No native source code or licensed software executed. These results are specific to the owner's `02_PINCHOUT_CONTROLS` benchmark.

## The actual native data dependencies

Static parsing of the saved `Vein_Benchmark.aproj` object graph shows two `engine.vein_blocks.pinch_outs.VeinPinchOutSegmentsBlock` objects with direct persisted attributes:
- OFF case `3069`: `collars_selector` -> `VeinPinchOutCollarsSelectorBlock` OID 3068, `_raw_vein` -> `VeinBlock` OID 3066, and `desurveyor` -> `DesurveyBlock` OID 608.
- ON case `3540`: `collars_selector` -> `VeinPinchOutCollarsSelectorBlock` OID 3563, `_raw_vein` -> `VeinBlock` OID 3356, and the same `desurveyor` OID 608.

The collars selector depends on the lithology category column and vein segments, and stores `_pinch_out_on_empty_traces=False` (A) or `True` (B). Both segment blocks retain `_dont_pinch_out_on_empty_intersections=False`, `_distance_to_vein=0.0`.

The resulting synthetic pinchout segments feed a **separate** downstream `VeinWallPointsGeneratorBlock` with `is_pinch_out=True`, also linked to the already-built reference surface. Thus a raw-vein/trajectory-based intersection and synthetic reversed-contact generation is consistent with the **actual dependency order**. In particular, the native pinchout segments are not required to be pre-existing lithology intervals.

## Quantitative evidence from the saved native benchmark

| Native property | Empty traces OFF | Empty traces ON |
|---|---:|---:|
| Selected pinchout drillholes | 17 | 20 |
| Synthetic MD support for each selected hole | 96–104 m | 96–104 m |
| Width | 8.0 m | 8.0 m |
| Center | MD 100 m | MD 100 m |
| Synthetic segments inside one logged WASTE interval | 16 | 16 |
| Synthetic segments in unlogged regions | 1 | 4 |

All observed VEIN intervals in this synthetic case are MD96–104, so the 8m synthetic interval width is consistent with using the preliminary raw vein wall intersections. The observed hangingwall/footwall points are respectively Z104/Z96 (collar Z200, vertical drilling); synthetic reversed controls are **HW Z96/FW Z104**.

One concrete WASTE example is `VH_X-040_Y-080`, logged entirely WASTE at MD0–200 yet receiving artificial MD96–104. An additional hole `VH_X+000_Y+040` is logged only at MD0–35 and MD160–200; its artificial segment lies **inside the unlogged MD35–160 gap**. Three wholly unlogged holes become newly eligible when the empty-traces switch is enabled.

The persisted default reference is Z101.78 while the HW/FW arithmetic center is Z100.0. Therefore reference-midpoint coincidence cannot be assumed, even in this simple flat vein. The reference need not be the geometric midwall in other cases, either.

### How to interpret these data

The hypothesis of putting reversed constraints directly on lithological sample intervals is incompatible with these saved native depths. The actual flow is more plausibly: eligible collar selection -> provisional raw-vein and desurveyed-trace geometric intersection -> artificial interval along drillhole -> reversed contact-point generation -> new walls/final volume. The specific algorithm determining **the interval depth endpoints or width in oblique or curved cases remains unknown**.

The [official 2026.1 Veins help](https://help.seequent.com/Geo/2026.1/en-GB/Content/geo-models/veins.htm) independently describes outside intervals and says empty trace triggers can be defined by intersection with the **HW and/or FW** walls and logged-hole distance thresholds, not a requirement that the reference intersect the same lithology sample.

## Controlled next experiment, import-ready offline pack

Created privately in conversation: `Leapfrog_Pinchout_Reverse_Engineering_Test_20261018.zip` (three folders, each with collar, survey and geology CSVs):
1. A control: original MD96–104 five VEIN intervals, 8m wide.
2. B expansion: **MD94–106** five VEIN intervals, 12m wide; all **midpoints remain MD100** and adjoining WASTE contacts are adjusted.
3. C segmentation control: same vein data as A, but one full WASTE MD0–200 interval is split into WASTE MD0–98.3, MD98.3–101.7, and MD101.7–200. This artificially introduces two sample boundaries within the expected pinchout without changing any categorical geology.

All cases preserve 25 collars, 50 survey rows, and drilling coordinates; all have no overlapping intervals. They are built from the saved project's **valid 33-row geology subset**, not the invalid/filtered six rows of its historic original import. Compare **new A** against new B/C in the same model configurations.

**Decisive expected tests**:
- If B's artificial interval expands to ~12m while its midpoint and reference-input midpoints stay fixed, native width is linked to raw vein thickness/walls rather than a fixed chosen sample interval. Also compare reference surfaces to test wall geometry's influence on reference construction.
- If C's artificial interval remains ~96–104 across the newly split WASTE sample contacts, the native segment geometry is independent of WASTE sample boundary locations for this test. If it moves, further inspect geological logging/aggregation logic.
- For a full numeric check, save and zip each resulting `.aproj` **with its `.aproj_data` folder**; a Reference Surface OBJ alone cannot encode virtual pinchout interval endpoints.

**Still unknown:** exact native ray/cylinder tracing, interval construction width formula, branch picking with repeated crossings, logged-hole distance gating, and whether reference surface changes for different wall widths. Do not generalize numerical values to every vein or merge heuristic assumptions to main.
