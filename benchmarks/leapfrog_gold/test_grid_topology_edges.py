"""Check structured centroid neighbor graph edge counts and uniqueness."""
from grid_topology_edges import grid_edges


def run():
    cases = [
        ((1, 1, 1), (0, 0, 0)),
        ((2, 2, 2), (12, 24, 28)),
        ((16, 25, 15), (16985, 49037, 69197)),
        ((32, 14, 14), (17724, 51108, 72064)),
    ]
    for shape, expected in cases:
        sets = []
        for mode, count in zip((6, 18, 26), expected):
            arr = grid_edges(shape, mode)
            assert arr.shape == (count, 2), (shape, mode, arr.shape, count)
            s = set(map(tuple, arr))
            assert len(s) == count
            assert all(0 <= a < b < shape[0]*shape[1]*shape[2] for a,b in s)
            sets.append(s)
        assert sets[0] <= sets[1] <= sets[2]
    print("PASS: grid topology 6/18/26, unique C-order edges, all shapes")


if __name__ == "__main__":
    run()
