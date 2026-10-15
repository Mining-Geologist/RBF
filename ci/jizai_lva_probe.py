"""Jizai 1.0.0 structural API audit and controlled isotropic invariance test.

No proprietary data. Does not claim structural LVA parity.
"""
import inspect
import json
import time
from pathlib import Path

import numpy as np
import jizai as jz


def public_api(obj):
    names = [n for n in dir(obj) if not n.startswith("_")]
    return names


def signature(obj):
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return str(getattr(obj, "__doc__", ""))[:500]


def fitted_field(train, vals, query):
    model = jz.Model(jz.Biharmonic3D(dim=3))
    interpolant = jz.Interpolant(model)
    t0 = time.perf_counter()
    interpolant.fit(train, vals, tolerance=1e-5, accuracy=1e-7)
    fit_seconds = time.perf_counter() - t0
    pred = np.asarray(interpolant.evaluate(query, accuracy=1e-7), dtype=float).ravel()
    return pred, fit_seconds, model


def main():
    rng = np.random.default_rng(20261015)
    # Training planar SDF values from a smooth, curved synthetic geological field.
    train = rng.uniform([-50, -45, -35], [50, 45, 35], size=(220, 3))
    query = rng.uniform([-35, -30, -25], [35, 30, 25], size=(120, 3))
    def f(p):
        return p[:, 2] - 0.23 * p[:, 0] - 4 * np.sin(p[:, 1] / 22)
    vals = f(train)
    p0, fit0, model = fitted_field(train, vals, query)

    angle = np.deg2rad(37)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]], dtype=float)
    translation = np.array([42312.0, 659210.0, 1250.0])
    # A correct isotropic fit must be invariant under a rigid coordinate transform.
    p1, fit1, _ = fitted_field(train @ rotation.T + translation,
                              vals, query @ rotation.T + translation)
    delta = p1 - p0
    model_names = public_api(model)
    jizai_names = public_api(jz)
    structural_terms = ("structur", "trend", "domain", "anisotrop", "orient", "tensor")
    result = {
        "library": "jizai",
        "reference": "synthetic-only, not Leapfrog",
        "train_count": len(train),
        "query_count": len(query),
        "isotropic_rigid_invariance": {
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "max_abs": float(np.max(np.abs(delta))),
            "fit_seconds_original": fit0,
            "fit_seconds_transformed": fit1,
            "sign_agreement_fraction": float(np.mean(np.sign(p0) == np.sign(p1))),
        },
        "model_public_names": model_names,
        "jizai_structural_candidate_names": [n for n in jizai_names if any(t in n.lower() for t in structural_terms)],
        "model_anisotropies_present": hasattr(model, "anisotropies"),
        "anisotropies_signature_or_doc": signature(getattr(type(model), "anisotropies", None)),
        "model_doc": str(getattr(jz.Model, "__doc__", ""))[:1000],
        "candidate_constructor_signatures": {
            n: signature(getattr(jz, n)) for n in jizai_names
            if any(t in n.lower() for t in structural_terms)
        },
    }
    Path("jizai_lva_probe.json").write_text(json.dumps(result, indent=2))
    np.savetxt("jizai_rigid_invariance.csv",
               np.column_stack([query, p0, p1, delta]),
               delimiter=",", header="x,y,z,pred_original,pred_rigid_transformed,difference",
               comments="")
    print(json.dumps(result, indent=2))
    # Sanity guard - precision should remain within solver precision for this rigid transform.
    assert np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))
    assert result["isotropic_rigid_invariance"]["sign_agreement_fraction"] >= 0.99
    assert result["isotropic_rigid_invariance"]["rmse"] < 1e-2


if __name__ == "__main__":
    main()
