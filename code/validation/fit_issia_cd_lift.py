"""Identifiability test on A's two ISSIA arcs: fit (C_D, L) on a 6-frame prefix.

A's handoff v2 (`02_ISSIA数据说明.md` §7.1 note 5) proposes fitting one extra
net-lift scalar on the same 6-frame prefix: with 2 parameters and 18
observations the pair should be identifiable. A's §7 analysis of the same arcs
found a non-gravity *vertical* acceleration of about +3.2 m/s^2, so the lift is
taken to be the component of the vertical direction perpendicular to the
velocity (i.e. a net lift scalar L in m/s^2):

    a = g - k C_D |v| v + L * unit(z_hat - v_hat (z_hat . v_hat)),
    k = rho A / (2 m)

For comparison we also fit a horizontal lift (vertical spin axis,
direction unit(z_hat x v)), which is the assumption used in the earlier
zero-spin extrapolation check; it is kept as a negative control.

Setup mirrors `check_issia_arcs.py`: fit initial state + dynamics parameters on
the first 6 frames (0-0.20 s), score all later frames held out. Reported are
prefix/forecast RMSE, parameter standard errors from the Jacobian, the CD-L
correlation and the Jacobian condition number.

Interpretation boundary (A's review, 2026-11-07): a reduced forecast error only
means that this model extension is worth considering. A small CD-L correlation
or a normal condition number does NOT prove that the physical parameters are
uniquely identified; the vertical force may absorb tracking/reconstruction or
other model error. These results must NOT be used as evidence that C_D(v) is
necessary, and the fitted C_D is not a calibration value.

Usage (from repo root):
    python student_b/fit_issia_cd_lift.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

FPS = 25.0
PREFIX = 6
K = 1.20 * np.pi * 0.11**2 / (2 * 0.430)
DATA = Path(__file__).resolve().parents[2] / "experiments" / "validation" / "issia_A"
ARCS = (("seg_0608_0627.csv", 622), ("seg_1144_1172.csv", 1172))
MODELS = ("gravity", "drag", "drag+lift_perp", "drag+lift_horizontal")


def load(path, last_frame):
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
    data = data[data["frame_id"] <= last_frame]
    frames = data["frame_id"].astype(int)
    times = (frames - frames[0]) / FPS
    observed = np.column_stack([data[n] for n in ("x_m", "y_m", "z_up_m")])
    return frames, times, observed


def accel(v, kind, cd, lift):
    speed = np.linalg.norm(v)
    a = np.array([0.0, 0.0, -9.81]) - K * cd * speed * v
    if lift == 0.0:
        return a
    if kind.endswith("horizontal"):
        direction = np.cross(np.array([0.0, 0.0, 1.0]), v)
    else:  # vertical lift, made perpendicular to the velocity
        v_hat = v / speed
        direction = np.array([0.0, 0.0, 1.0]) - v_hat * v_hat[2]
    norm = np.linalg.norm(direction)
    if norm > 1e-9:
        a = a + lift * direction / norm
    return a


def rollout(x, times, kind):
    r0, v0 = x[:3], x[3:6]
    cd = x[6] if kind != "gravity" else 0.0
    lift = x[7] if kind.endswith("lift_perp") or kind.endswith("lift_horizontal") else 0.0

    def rhs(t, state):
        return np.r_[state[3:], accel(state[3:], kind, cd, lift)]

    result = solve_ivp(rhs, (0.0, times[-1]), np.r_[r0, v0], t_eval=times,
                       rtol=1e-9, atol=1e-10)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:3].T


def fit_arc(path, last_frame, kind):
    frames, times, observed = load(path, last_frame)
    prefix = min(PREFIX, len(times) - 1)
    slope, intercept = np.polyfit(times[:prefix], observed[:prefix], 1)
    x0 = np.r_[intercept, slope]
    if kind != "gravity":
        x0 = np.r_[x0, 0.22]
    if kind.endswith("lift_perp") or kind.endswith("lift_horizontal"):
        x0 = np.r_[x0, 3.2 if kind.endswith("lift_perp") else 0.0]
    lower = np.full(len(x0), -60.0)
    upper = np.full(len(x0), 60.0)
    if kind != "gravity":
        lower[6], upper[6] = 0.0, 1.5
    if kind.endswith("lift_perp"):
        lower[7], upper[7] = -40.0, 40.0
    if kind.endswith("lift_horizontal"):
        lower[7], upper[7] = -3.0, 3.0

    def residual(x):
        return (rollout(x, times[:prefix], kind) - observed[:prefix]).ravel()

    fit = least_squares(residual, x0, bounds=(lower, upper), method="trf",
                        x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    predicted = rollout(fit.x, times, kind)
    error = np.linalg.norm(predicted - observed, axis=1)
    n_obs, n_par = len(residual(fit.x)), len(fit.x)
    rss = float(np.sum(residual(fit.x) ** 2))
    dof = max(n_obs - n_par, 1)
    covariance = np.linalg.inv(fit.jac.T @ fit.jac) * rss / dof
    std = np.sqrt(np.diag(covariance))
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    names = (["x0_m", "y0_m", "z0_m", "vx0_m_s", "vy0_m_s", "vz0_m_s"]
             + ([] if kind == "gravity" else ["Cd"])
             + ([] if not kind.startswith("drag+lift") else ["L_m_s2"]))
    correlation = None
    if kind.startswith("drag+lift"):
        correlation = {"CD_vs_L": float(covariance[6, 7] / (std[6] * std[7]))}
    return dict(
        arc=path.name, last_frame=last_frame, model=kind, frames=len(times),
        prefix_frames=frames[:prefix].tolist(), forecast_frames=frames[prefix:].tolist(),
        prefix_rmse_m=float(np.sqrt(np.mean(error[:prefix] ** 2))),
        forecast_rmse_m=float(np.sqrt(np.mean(error[prefix:] ** 2))),
        final_error_m=float(error[-1]),
        n_observations_prefix=int(n_obs), n_parameters=int(n_par), dof=int(dof),
        jacobian_condition=float(singular[0] / singular[-1]), success=bool(fit.success),
        parameters={n: dict(value=float(v), std=float(s)) for n, v, s in zip(names, fit.x, std)},
        correlation=correlation)


def main():
    result = {"k_rhoA_2m": K, "fps": FPS, "prefix_frames": PREFIX,
              "lift_model": "a = g - k CD |v| v + L * unit(z - v_hat (z . v_hat)); L in m/s^2",
              "fit_scope": "initial state + dynamics parameters fitted on the 6-frame prefix only; later frames held out",
              "interpretation_caveat": (
                  "Forecast-error reduction only means the vertical-force extension is worth considering. "
                  "A near-zero CD-L correlation or a normal Jacobian condition number does not prove unique "
                  "physical identification: the vertical force may absorb tracking/reconstruction or other "
                  "model error. Do not use these fits as evidence that C_D(v) is necessary, and do not quote "
                  "the fitted C_D as a calibration value."),
              "arcs": []}
    for name, last in ARCS:
        path = DATA / name
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        arc = {"file": name, "models": {}}
        for kind in MODELS:
            report = fit_arc(path, last, kind)
            arc["models"][kind] = report
            print(json.dumps({k: report[k] for k in
                              ("arc", "model", "prefix_rmse_m", "forecast_rmse_m",
                               "jacobian_condition", "parameters", "correlation")},
                             ensure_ascii=False), flush=True)
        result["arcs"].append(arc)
    out = DATA / "cd_lift_fit.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
