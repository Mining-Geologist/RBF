# Vein multiple-intersection and HW/FW orientation audit — 2026-10-18

Offline, owner-owned saved Leapfrog Geo 2026.1.1 benchmark. No native software executed; no proprietary file bytes committed. Research branch only.

## Direct native evidence

Saved `04_MULTIPLE_INTERSECTIONS` has **multiple_intersections=true**. Nine drillholes contain 15 VEIN intersections; six have two vein intervals at MD 78–85 and 118–125, separated by 33 m WASTE. Native ON output contains **15 HW and 15 FW contact points**. Across those six double-hit drillholes, three lower intersections have reversed top/bottom HW/FW labels.

## Multiple-intersections OFF remains unverified

The user's hypothesis is **outermost contacts**: treat MD 78 as the first contact and MD 125 as the last contact, dropping both internal contact points. This yields a **47 m envelope** instead of 14 m of actual vein in each double-hit hole and bridges **33 m of intervening waste**. The alternate **largest interval** hypothesis selects one 7 m vein only. Both policies would yield nine HW/FW pairs instead of the 15 ON pairs, so contact **count cannot distinguish them**. The saved project has no native OFF counterpart; neither hypothesis is proved. Seequent documents the control but not the exact OFF rule: https://help.seequent.com/Geo/2026.1/en-GB/Content/geo-models/veins.htm

## Reference-normal orientation experiment

Independent predictor: sample a face normal from the saved 3D reference mesh at the interval's start (first) contact, calculate normalized downhole tangent, assign that start contact to HW when `normal dot downhole > 0`, else FW; end contact receives opposite polarity. Explicit reference inversion flips assignments. It uses only observed interval depths, collar/traces, and the **saved reference mesh**; native HW/FW labels are read only after prediction.

**100/100** native start-contact assignments were matched across all six saved model cases: flat 25/25, pinchout A+B 10/10, curved 25/25, multiple-hit 15/15, and end-hole 25/25 (one additional end-hole interval has no native saved start contact and was excluded).

The **multiple-hit model's entire 15 HW + 15 FW 3D coordinate sets** were reconstructed from the stored drillhole and reference data, with maximum bidirectional nearest-coordinate difference **7.7e−15 m**, no native contact labels used in prediction.

By contrast, using midpoint reference normals matched 99/100 and 14/15 on multiple-hit. The exception at X=0,Y=0 on the lower hit had a 45.68m midpoint-to-reference distance and selected the wrong branch. A naïve always-top-is-HW classification matched only 12/15 on the multiple-hit case.

**Caveats:** choosing the start-contact sampling rule followed inspection of this same benchmark's midpoint error, so 100/100 is an **in-sample** match, not independently validated across unseen folds. The benchmark provides a *native saved reference surface*, which is not independently reconstructed. No proof of Leapfrog's exact internal orientation rule follows from numerical coincidence; it is a strong useful hypothesis. The multiple-hits-OFF option remains untested in native outputs.

## Deliverables

Original independent code: `benchmarks/vein_parity/vein_segment_orientation.py`; synthetic source-independent test: `test_vein_segment_orientation.py`; CI `.github/workflows/offline-vein-segment-orientation.yml`.

Owner-data comparisons and full 100-row numerical evidence are available as separate conversation ZIP `Leapfrog_Vein_Multiple_Intersection_Orientation_Audit_20261018.zip` (not published in this public repository).

**Next experiment:** construct reference mesh independently from saved midpoints and geological orientation; then revalidate wall assignment and offset interpolation on a truly new geometry. For OFF semantics, a *single duplicate project* with multiple intersections disabled and all other options held constant would distinguish outermost vs largest. Keep both in research code, not production, until validated.
