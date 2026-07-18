"""Stable LVA-field launcher for the Polatory PyQt application.

Place this file beside ``polatory_lva_pyqt_app_v3_safe.py`` and
``polatory_lva_pyqt_app_v2.py``, then run::

    python polatory_lva_pyqt_app_v4_lva_safe.py

The v2 display path used VTK's interpolating ``slice_orthogonal`` filter on a
three-dimensional StructuredGrid.  Some low grid dimensions can terminate the
VTK/Qt process inside that native filter.  This launcher keeps the same model and
LVA samples, but displays the three centre planes by extracting exact structured
grid index planes.  No interpolating slice filter is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv

import polatory_lva_pyqt_app_v3_safe as v3

app = v3.app


def _nearest_grid_index(
    coordinate: float | None,
    minimum: float,
    maximum: float,
    size: int,
) -> int:
    if size < 1:
        raise ValueError("Structured-grid dimensions must be positive.")
    if size == 1 or not maximum > minimum:
        return 0
    if coordinate is None or not np.isfinite(coordinate):
        coordinate = 0.5 * (minimum + maximum)
    fraction = (float(coordinate) - minimum) / (maximum - minimum)
    return int(np.clip(np.rint(fraction * (size - 1)), 0, size - 1))


def safe_slice_orthogonal(
    self: pv.StructuredGrid,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    *args: Any,
    **kwargs: Any,
) -> pv.PolyData:
    """Return three exact centre planes without invoking VTK's slice filter."""

    dimensions = tuple(int(value) for value in self.dimensions)
    if len(dimensions) != 3 or any(value < 2 for value in dimensions):
        raise ValueError(
            "LVA field display requires at least two samples along every axis."
        )

    bounds = self.bounds
    ix = _nearest_grid_index(x, bounds.x_min, bounds.x_max, dimensions[0])
    iy = _nearest_grid_index(y, bounds.y_min, bounds.y_max, dimensions[1])
    iz = _nearest_grid_index(z, bounds.z_min, bounds.z_max, dimensions[2])

    # VTK VOI order: xmin, xmax, ymin, ymax, zmin, zmax.
    vois = (
        (ix, ix, 0, dimensions[1] - 1, 0, dimensions[2] - 1),
        (0, dimensions[0] - 1, iy, iy, 0, dimensions[2] - 1),
        (0, dimensions[0] - 1, 0, dimensions[1] - 1, iz, iz),
    )

    planes: list[pv.PolyData] = []
    for voi in vois:
        plane = self.extract_subset(voi)
        surface = plane.extract_surface().triangulate().clean()
        if surface.n_points and surface.n_cells:
            planes.append(surface)

    if not planes:
        raise RuntimeError("The LVA field centre planes are empty.")

    combined: pv.DataSet = planes[0]
    for plane in planes[1:]:
        combined = combined.merge(plane, merge_points=False)

    return combined.extract_surface().triangulate().clean()


# Patch only the problematic display operation.  Meshing, automatic domains,
# exported OBJ geometry and LVA values remain unchanged.
pv.StructuredGrid.slice_orthogonal = safe_slice_orthogonal


if __name__ == "__main__":
    raise SystemExit(app.main())
