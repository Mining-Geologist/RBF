"""Numerical Jizai 1.0.0 anisotropy baseline and orientation-field prototype.

Only deterministic SYNTHETIC data. This is NOT a Leapfrog parity claim, and
interpolating orientation components is NOT structural LVA/domain subdivision.
"""
import csv
import json
import time
from pathlib import Path
import numpy as np
import jizai as jz

OUT = Path("jizai_lva_baseline")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261015)
RANGE = 65.0
PSILL = 0.8


def fit(xyz, values, matrix=None, span=RANGE):
    model = jz.Model(jz.CovSpheroidal3(psill=PSILL, range=span, dim=3), poly_degree=0)
    if matrix is not None:
        model.anisotropies = [matrix]
    interp = jz.Interpolant(model)
    tic = time.perf_counter()
    interp.fit(xyz, values, tolerance=1e-8, accuracy=1e-8, max_iter=200)
    elapsed = time.perf_counter() - tic
    return interp, elapsed


def evaluate(interp, points):
    return np.asarray(interp.evaluate(points, accuracy=1e-8), dtype=float).reshape(-1)


def covariance(a, b, transform):
    # Independent, dense implementation of the upstream Jizai 1.0.0
    # CovSpheroidal3 kernel and constant-polynomial interpolation.
    r = np.linalg.norm((a[:, None, :] - b[None, :, :]) @ transform.T, axis=2)
    rho = r / RANGE
    return PSILL * np.where(
        rho < 0.18657871684006438,
        1.0 - 2.009875543958482 * rho,
        0.8734640537108553 / (1.0 + 7.181510581693163 * rho**2)**1.5
    )


def exact_predict(x, values, q, transform):
    K = covariance(x, x, transform)
    n = len(x)
    lhs = np.zeros((n + 1, n + 1))
    lhs[:n, :n] = K
    lhs[:n, n] = 1.0
    lhs[n, :n] = 1.0
    rhs = np.r_[values, 0.0]
    coeff = np.linalg.solve(lhs, rhs)
    return covariance(q, x, transform) @ coeff[:n] + coeff[n]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b)**2)))


def save_csv(filename, columns, arrays):
    with (OUT / filename).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*arrays))


def anisotropy_experiment():
    train = RNG.uniform(-26, 26, size=(130, 3))
    query = RNG.uniform(-28, 28, size=(180, 3))
    values = np.sin(train[:, 0] / 11) + .35 * np.cos(train[:, 1] / 9) - .08 * train[:, 2] / 8
    angle = np.deg2rad(36.0)
    R = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                  [np.sin(angle), np.cos(angle), 0.0],
                  [0.0, 0.0, 1.0]])
    # Deliberately nonsymmetric: tests A vs A^T convention.
    A = np.diag([0.55, 1.0, 1.8]) @ R
    assert np.linalg.det(A) > 0
    interp_aniso, time_a = fit(train, values, A)
    interp_iso, time_i = fit(train, values)
    actual = evaluate(interp_aniso, query)
    isotropic = evaluate(interp_iso, query)
    exact_A = exact_predict(train, values, query, A)
    exact_AT = exact_predict(train, values, query, A.T)
    err_A = rmse(actual, exact_A)
    err_AT = rmse(actual, exact_AT)
    save_csv("anisotropy_predictions.csv",
             ["X","Y","Z","jizai_anisotropic","dense_A","dense_A_transpose","jizai_isotropic"],
             [query[:,0],query[:,1],query[:,2],actual,exact_A,exact_AT,isotropic])
    result = {
      "synthetic_training_n": len(train), "heldout_n": len(query),
      "matrix": A.tolist(), "range": RANGE, "psill": PSILL,
      "jizai_vs_independent_dense_A_rmse": err_A,
      "jizai_vs_dense_A_transpose_rmse": err_AT,
      "jizai_aniso_vs_jizai_iso_rmse": rmse(actual, isotropic),
      "jizai_aniso_fit_seconds": time_a, "jizai_iso_fit_seconds": time_i,
      "preferred_transform": "A" if err_A < err_AT else "A_transpose"
    }
    # Independent solver is a real numerical regression guard, not just a smoke test.
    assert min(err_A, err_AT) < 2e-3, result
    assert result["jizai_aniso_vs_jizai_iso_rmse"] > 1e-3, result
    return result


def axial_error(pred, actual):
    return 0.5 * np.arctan2(np.sin(2 * (pred - actual)), np.cos(2 * (pred - actual)))


def truth_angle(points):
    x, y = points[:, 0], points[:, 1]
    return .65 * np.tanh(x / 30.0) + .18 * np.sin(y / 36.0)


def orientation_experiment():
    # Synthetic folded trend, fixed no-noise samples. This tests standard RBF
    # orientation interpolation, not Jizai native spatially varying anisotropy.
    train = RNG.uniform([-75., -70., -12.], [75., 70., 12.], size=(170, 3))
    query = RNG.uniform([-73., -68., -10.], [73., 68., 10.], size=(400, 3))
    theta_t = truth_angle(train)
    theta_q = truth_angle(query)
    ic, tc = fit(train, np.cos(2*theta_t), span=78.0)
    is_, ts = fit(train, np.sin(2*theta_t), span=78.0)
    c_pred = evaluate(ic, query)
    s_pred = evaluate(is_, query)
    theta_hat = 0.5 * np.arctan2(s_pred, c_pred)
    theta_global = 0.5 * np.arctan2(np.mean(np.sin(2*theta_t)), np.mean(np.cos(2*theta_t)))
    model_err = np.abs(np.rad2deg(axial_error(theta_hat, theta_q)))
    global_err = np.abs(np.rad2deg(axial_error(theta_global, theta_q)))
    save_csv("orientation_predictions.csv",
             ["X","Y","Z","truth_degrees","jizai_degrees","abs_error_degrees","global_abs_error_degrees"],
             [query[:,0], query[:,1],query[:,2],np.rad2deg(theta_q),
              np.rad2deg(theta_hat),model_err,global_err])
    result = {
      "dataset": "synthetic smooth folded 3D orientation trend; NOT Leapfrog LVA",
      "training_n": len(train), "heldout_n": len(query),
      "orientation_jizai_mae_deg": float(np.mean(model_err)),
      "orientation_jizai_p95_deg": float(np.percentile(model_err,95)),
      "orientation_global_mae_deg": float(np.mean(global_err)),
      "orientation_global_p95_deg": float(np.percentile(global_err,95)),
      "orientation_improvement_over_global": float(np.mean(global_err)/np.mean(model_err)),
      "fit_seconds": tc + ts
    }
    assert np.all(np.isfinite(model_err))
    return result


def main():
    report = {
       "engine": "Jizai 1.0.0 source v1.0.0 + existing GCC13 compatibility patches",
       "experiment": "global anisotropy convention and smooth structural orientation interpolation",
       "scope": "Synthetic only; no Leapfrog parity or automatic domains yet",
       "anisotropy": anisotropy_experiment(),
       "orientation": orientation_experiment()
    }
    with (OUT / "result.json").open("w") as f:
        json.dump(report, f, indent=2)
    print("JIZAI_LVA_BASELINE_RESULTS_START")
    print(json.dumps(report, indent=2))
    print("JIZAI_LVA_BASELINE_RESULTS_END")


if __name__ == "__main__":
    main()
