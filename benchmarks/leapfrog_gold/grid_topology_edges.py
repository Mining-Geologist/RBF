"""Neighbour graphs for a structured 3-D centroid grid (C-order)."""
from __future__ import annotations
import numpy as np


def grid_edges(shape: tuple[int, int, int], connectivity: int = 6) -> np.ndarray:
    """Unique pairs for 6-face, 18-face/edge or 26-face/edge/corner topology."""
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18 or 26")
    if len(shape) != 3 or any(d < 1 for d in shape):
        raise ValueError("shape must have three positive dimensions")
    allowed = {6: 1, 18: 2, 26: 3}[connectivity]
    grid = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    parts = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                move = (di, dj, dk)
                axes = sum(v != 0 for v in move)
                if axes == 0 or axes > allowed:
                    continue
                if next(v for v in move if v != 0) < 0:
                    continue
                left = tuple(slice(max(0, -v), min(n, n - v)) for v, n in zip(move, shape))
                right = tuple(slice(max(0, v), min(n, n + v)) for v, n in zip(move, shape))
                if any(s.start >= s.stop for s in left):
                    continue
                parts.append(np.column_stack((grid[left].ravel(), grid[right].ravel())))
    return np.vstack(parts) if parts else np.empty((0, 2), dtype=np.int64)
