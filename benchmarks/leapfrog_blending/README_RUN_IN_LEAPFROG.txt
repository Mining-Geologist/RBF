LEAPFROG LVA / SUBDOMAIN BLENDING BENCHMARKS

PURPOSE
-------
These synthetic files are designed to reveal how Leapfrog blends local structural
RBF domains and how the field behaves far from data.

FILES
-----
folded_trend_mesh.obj
    One connected folded structural surface. Its left and right limbs have opposite dip.

01_two_domain_blend_points.csv
    Left contact sheet:  Z = -0.5 X
    Right contact sheet: Z =  0.5 X + 100
    Inside points are 20 m below each sheet, Contacts are on the sheet, and Outside
    points are 20 m above.

02_background_decay_points.csv
    One continuous folded contact sheet: Z = 0.5 |X|.
    This isolates Leapfrog's far-field/background behaviour.

IMPORTANT
---------
Use the same Leapfrog workflow you used to create your S5_R100 benchmark from your
current Inside / Outside / Contact CSV. Do not change hidden or advanced defaults.
Only Strength and Range should be changed as normal structural controls.

TEST 1 - TWO-DOMAIN BLEND
-------------------------
1. Import 01_two_domain_blend_points.csv as point data.
2. Map X, Y and Z to the coordinate columns.
3. Use Role as the categorical role column:
       Inside  -> Inside
       Outside -> Outside
       Contact -> Contact / zero constraint
   The Indicator column is only a check:
       Inside=-1, Contact=0, Outside=+1
4. Import folded_trend_mesh.obj as the structural/LVA input mesh.
5. Build the same categorical RBF / intrusion / indicator volume workflow used for
   the S5_R100 benchmark.
6. Structural trend mode: Strongest along inputs.
7. Strength: 5
8. Range: 100
9. Leave every other Leapfrog setting at its default.
10. Use this exact model extent:
       X min = -300    X max = 300
       Y min = -150    Y max = 150
       Z min = -150    Z max = 350
11. Use surface resolution 5 m, or the closest available Leapfrog setting.
12. Export the zero/contact surface as:
       LF_test1_tight_S5_R100.obj

TEST 1B - SAME MODEL WITH DEEP EXTENT
-------------------------------------
Duplicate Test 1. Change only the model extent:
       X min = -300    X max = 300
       Y min = -150    Y max = 150
       Z min = -500    Z max = 350
Export as:
       LF_test1_deep_S5_R100.obj

TEST 2 - BACKGROUND / FAR-FIELD DECAY
--------------------------------------
1. Import 02_background_decay_points.csv.
2. Use the same folded_trend_mesh.obj.
3. Use the same Inside / Outside / Contact mapping.
4. Structural trend mode: Strongest along inputs.
5. Strength: 5
6. Range: 100
7. Leave every other setting at the Leapfrog default.
8. Use this exact model extent:
       X min = -300    X max = 300
       Y min = -150    Y max = 150
       Z min = -500    Z max = 350
9. Export as:
       LF_test2_background_S5_R100.obj

SEND BACK
---------
Send these three OBJ files:
    LF_test1_tight_S5_R100.obj
    LF_test1_deep_S5_R100.obj
    LF_test2_background_S5_R100.obj

Also send one section-view screenshot of Test 1 at Y = 0 with the points and generated
surface visible.

WHY THESE RUNS ARE ENOUGH
-------------------------
Test 1 reveals the transition between two conflicting local zero surfaces.
Comparing Test 1 tight and deep reveals whether blending depends on model extent.
Test 2 separates local-domain blending from global/background closure.
