"""Process-isolated LVA GUI with Leapfrog-style automatic defaults.

The advanced controls remain visible for diagnostics, but the normal workflow is
to set the structural input, Strength and Trend range. The recovered defaults are
applied at startup. Finite local-domain weights now blend smoothly into the outside
field, so the neutral blend exponent is 1 rather than the old compensating value 7.

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

    # Strength and trend range remain the two normal user-facing structural
    # controls. Blend power 1 is the neutral exponent for the recovered smooth
    # local-domain/outside partition of unity.
    defaults = (
        ("alignment_spin", 0.0),
        ("blend_power_spin", 1.0),
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
            "Leapfrog-style automatic default: 1. Finite local-domain influence "
            "is blended smoothly into the Outside field. Normally leave this "
            "unchanged and adjust only Strength and Trend range."
        )

    self._log(
        "Leapfrog-style defaults loaded: automatic SubDomainer, finite "
        "support-radius-bounded LVA coverage, smooth Outside-field blending, "
        "and blend power 1."
    )


app.MainWindow.__init__ = leapfrog_default_window_init


if __name__ == "__main__":
    raise SystemExit(app.main())
