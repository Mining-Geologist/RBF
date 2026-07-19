"""Process-isolated LVA GUI with Leapfrog-style automatic defaults.

The advanced controls remain visible for diagnostics, but the normal workflow is
to set the structural input, Strength and Trend range. The recovered defaults are
applied at startup, including blend power 7 for local structural-domain blending.

Run:
    python polatory_lva_pyqt_app_v11_leapfrog_defaults.py
"""

from __future__ import annotations

from typing import Any

import polatory_lva_pyqt_app_v10_process_isolated as v10

app = v10.app
_original_window_init = app.MainWindow.__init__


def leapfrog_default_window_init(self: Any) -> None:
    _original_window_init(self)

    # Defaults recovered from the benchmark workflow. Strength and trend range
    # remain the two normal user-facing structural controls.
    defaults = (
        ("alignment_spin", 0.0),
        ("blend_power_spin", 7.0),
        ("centroid_count_spin", 6000),
        ("minimum_fraction_spin", 0.001),
        ("maximum_fraction_spin", 0.10),
        ("consistency_spin", 0.60),
        ("support_multiplier_spin", 5),
        ("minimum_support_spin", 1),
    )
    for attribute, value in defaults:
        widget = getattr(self, attribute, None)
        if widget is not None:
            widget.setValue(value)

    trend_type = getattr(self, "trend_type_combo", None)
    if trend_type is not None:
        index = trend_type.findText("Strongest along inputs")
        if index >= 0:
            trend_type.setCurrentIndex(index)

    blend = getattr(self, "blend_power_spin", None)
    if blend is not None:
        blend.setToolTip(
            "Leapfrog-style automatic default: 7. Normally leave this unchanged; "
            "adjust Strength and Trend range for the structural input."
        )

    self._log(
        "Leapfrog-style defaults loaded: automatic SubDomainer, finite "
        "support-radius-bounded LVA coverage, and blend power 7."
    )


app.MainWindow.__init__ = leapfrog_default_window_init


if __name__ == "__main__":
    raise SystemExit(app.main())
