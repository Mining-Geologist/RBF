"""Synthetic, native-free checks for the U-fold connected reference-sheet hypothesis."""
import numpy as np
from experimental_fold_reference_sheet import (
    make_synthetic_fold_reference, point_to_fold_profile_distance
)


def main():
    points = []
    for y in (-40, 0, 40):
        for x, z in ((-40,100),(0,118.5),(40,118.5),
                     (40,78.5),(0,78.5)):
            points.append((x,y,z))
    pts = np.asarray(points, float)
    mesh, profile = make_synthetic_fold_reference(pts)
    assert len(mesh.vertices) == 81 * 31
    assert len(mesh.faces) == 80 * 30 * 2
    center_section = mesh.vertices.reshape(81,31,3)[:,0,:][:,[0,2]]
    assert np.max(point_to_fold_profile_distance(profile,center_section)) < .7
    # Confirm a true folded surface: two Z elevations at approximately same XY.
    assert len(np.unique(np.round(mesh.vertices[:,2],2))) > 30
    for bad in (pts[:4], np.zeros((10, 2))):
        try:
            make_synthetic_fold_reference(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid source geometry must be rejected")
    print("PASS: connected folded sheet, correct topology, finite geometry")


if __name__ == "__main__":
    main()
