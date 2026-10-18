"""Independent numerical test of candidate segment policies and local face normals."""
import numpy as np
import trimesh
from vein_segment_orientation import Segment, select_segments, predict_orientations


def main():
    intervals = [Segment(78., 85.), Segment(118., 125.)]
    assert select_segments(intervals, "all") == intervals
    assert select_segments(intervals, "outermost") == [Segment(78., 125.)]
    assert select_segments(intervals, "largest") == [Segment(78., 85.)]
    trace = np.array([[0, 0, 0, 200], [200, 0, 0, 0]], float)
    vertices = np.array([[-1, -1, 100], [-1, 1, 100],
                         [1, 1, 100], [1, -1, 100]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert np.all(mesh.face_normals[:,2] < 0)
    result = predict_orientations(mesh, trace, intervals)
    assert len(result) == 2
    assert all(x["start_wall"]=="HW" and x["end_wall"]=="FW" for x in result)
    reverse = predict_orientations(mesh, trace, intervals, invert_reference=True)
    assert all(x["start_wall"]=="FW" and x["end_wall"]=="HW" for x in reverse)
    assert predict_orientations(mesh, trace, intervals, mode="outermost")[0]["end_md"] == 125
    try:
        select_segments([Segment(0, 10), Segment(9, 20)])
    except ValueError:
        pass
    else:
        raise AssertionError("Overlapping segments must be rejected")
    print("PASS: independent segment policies, HW/FW polarity and inversion")


if __name__=="__main__":
    main()
