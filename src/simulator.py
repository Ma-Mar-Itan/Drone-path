"""
src/simulator.py
----------------
RK4 forward simulation of learned symbolic dynamics.

CRITICAL: All integration happens in NORMALISED state space.
The norm_stats dict (from fit_equations) is required to:
  - normalise the initial conditions before integration
  - de-normalise position/velocity outputs for display
  - keep acceleration predictions numerically stable
"""

import numpy as np
from typing import Callable


def _make_accel_fn(model, norm_stats: dict) -> Callable:
    """
    Return a callable (x, y, vx, vy) -> (ax, ay) operating in PIXEL space.

    Internally normalises inputs, calls model, de-normalises outputs.
    """
    mu_x,  sig_x  = norm_stats["x"]
    mu_y,  sig_y  = norm_stats["y"]
    mu_vx, sig_vx = norm_stats["vx"]
    mu_vy, sig_vy = norm_stats["vy"]
    mu_ax, sig_ax = norm_stats["ax"]
    mu_ay, sig_ay = norm_stats["ay"]

    def fn(x, y, vx, vy):
        xn  = (x  - mu_x)  / sig_x
        yn  = (y  - mu_y)  / sig_y
        vxn = (vx - mu_vx) / sig_vx
        vyn = (vy - mu_vy) / sig_vy

        try:
            import pysindy as ps
            if isinstance(model, ps.SINDy):
                U = np.array([[xn, yn, vxn, vyn]])
                pred = model.predict(U)
                ax_n, ay_n = float(pred[0, 0]), float(pred[0, 1])
                return ax_n * sig_ax + mu_ax, ay_n * sig_ay + mu_ay
        except Exception:
            pass

        xs  = np.array([xn]); ys  = np.array([yn])
        vxs = np.array([vxn]); vys = np.array([vyn])
        ax_n_arr, ay_n_arr = model.predict(xs, ys, vxs, vys)
        return float(ax_n_arr[0]) * sig_ax + mu_ax, float(ay_n_arr[0]) * sig_ay + mu_ay

    return fn


def rk4_step(state: np.ndarray, accel_fn: Callable, dt: float) -> np.ndarray:
    """Single RK4 step. state = [x, y, vx, vy]"""
    def f(s):
        x, y, vx, vy = s
        ax, ay = accel_fn(x, y, vx, vy)
        return np.array([vx, vy, ax, ay])

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(model, norm_stats: dict,
             x0: float, y0: float, vx0: float, vy0: float,
             dt: float, N: int,
             clip_val: float = 1e5) -> dict:
    """
    Simulate trajectory forward using RK4 in pixel space.

    Parameters
    ----------
    model      : fitted symbolic model
    norm_stats : normalisation statistics from fit_equations()
    x0..vy0    : initial conditions in pixel units
    dt         : time step in seconds
    N          : number of steps
    clip_val   : safety clip to prevent runaway divergence
    """
    accel_fn = _make_accel_fn(model, norm_stats)

    xs  = np.zeros(N); ys  = np.zeros(N)
    vxs = np.zeros(N); vys = np.zeros(N)
    axs = np.zeros(N); ays = np.zeros(N)

    state = np.array([x0, y0, vx0, vy0], dtype=np.float64)

    for i in range(N):
        x, y, vx, vy = state
        ax, ay = accel_fn(x, y, vx, vy)
        xs[i], ys[i]   = x,  y
        vxs[i], vys[i] = vx, vy
        axs[i], ays[i] = ax, ay

        next_state = rk4_step(state, accel_fn, dt)
        state = np.clip(next_state, -clip_val, clip_val)

    return {"x": xs, "y": ys, "vx": vxs, "vy": vys,
            "ax": axs, "ay": ays, "t": np.arange(N) * dt}


def compute_error_metrics(true_x, true_y, pred_x, pred_y) -> dict:
    N = min(len(true_x), len(pred_x))
    tx, ty = true_x[:N], true_y[:N]
    px, py = pred_x[:N], pred_y[:N]

    rmse_x     = float(np.sqrt(np.mean((tx-px)**2)))
    rmse_y     = float(np.sqrt(np.mean((ty-py)**2)))
    rmse_total = float(np.sqrt(np.mean((tx-px)**2 + (ty-py)**2)))

    def dh(a_x, a_y, b_x, b_y):
        return max(np.sqrt((b_x - ax_i)**2 + (b_y - ay_i)**2).min()
                   for ax_i, ay_i in zip(a_x, a_y))

    hausdorff = float(max(dh(tx, ty, px, py), dh(px, py, tx, ty)))
    return {"rmse_x": rmse_x, "rmse_y": rmse_y,
            "rmse_total": rmse_total, "hausdorff": hausdorff}