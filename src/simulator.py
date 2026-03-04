"""
src/simulator.py
----------------
Forward simulation of the learned symbolic dynamics using RK4 integration.

The symbolic model defines:
    dx/dt   = vx
    dy/dt   = vy
    dvx/dt  = ax(x, y, vx, vy)
    dvy/dt  = ay(x, y, vx, vy)

We integrate this ODE system from given initial conditions.
"""

import numpy as np
from typing import Callable


def _get_accel_fn(model) -> Callable:
    """
    Wrap the symbolic model into a callable:
        (x_scalar, y_scalar, vx_scalar, vy_scalar) → (ax, ay)
    """
    try:
        import pysindy as ps
        if isinstance(model, ps.SINDy):
            def fn(x, y, vx, vy):
                U = np.array([[x, y, vx, vy]], dtype=np.float64)
                accel = model.predict(U)
                return float(accel[0, 0]), float(accel[0, 1])
            return fn
    except Exception:
        pass

    # ManualSINDy
    def fn(x, y, vx, vy):
        xs = np.array([x])
        ys = np.array([y])
        vxs = np.array([vx])
        vys = np.array([vy])
        ax_arr, ay_arr = model.predict(xs, ys, vxs, vys)
        return float(ax_arr[0]), float(ay_arr[0])
    return fn


def rk4_step(state: np.ndarray, accel_fn: Callable, dt: float) -> np.ndarray:
    """
    Single RK4 step.

    state = [x, y, vx, vy]
    """
    def f(s):
        x, y, vx, vy = s
        ax, ay = accel_fn(x, y, vx, vy)
        return np.array([vx, vy, ax, ay])

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    model,
    x0: float,
    y0: float,
    vx0: float,
    vy0: float,
    dt: float,
    N: int,
    clip_val: float = 1e4,
) -> dict:
    """
    Simulate trajectory forward using RK4.

    Parameters
    ----------
    model      : fitted symbolic model (PySINDy or ManualSINDy)
    x0, y0     : initial position
    vx0, vy0   : initial velocity
    dt         : time step
    N          : number of simulation steps
    clip_val   : safety clip for diverging simulations

    Returns
    -------
    dict with keys: x, y, vx, vy, ax, ay, t
    """
    accel_fn = _get_accel_fn(model)

    xs = np.zeros(N)
    ys = np.zeros(N)
    vxs = np.zeros(N)
    vys = np.zeros(N)
    axs = np.zeros(N)
    ays = np.zeros(N)

    state = np.array([x0, y0, vx0, vy0], dtype=np.float64)

    for i in range(N):
        x, y, vx, vy = state
        ax, ay = accel_fn(x, y, vx, vy)

        xs[i], ys[i] = x, y
        vxs[i], vys[i] = vx, vy
        axs[i], ays[i] = ax, ay

        # Clip to prevent divergence
        state = np.clip(rk4_step(state, accel_fn, dt), -clip_val, clip_val)

    t = np.arange(N) * dt
    return {"x": xs, "y": ys, "vx": vxs, "vy": vys, "ax": axs, "ay": ays, "t": t}


def compute_error_metrics(true_x: np.ndarray, true_y: np.ndarray,
                           pred_x: np.ndarray, pred_y: np.ndarray) -> dict:
    """
    Compute trajectory error metrics.

    Returns
    -------
    dict with: rmse_x, rmse_y, rmse_total, hausdorff
    """
    N = min(len(true_x), len(pred_x))
    tx, ty = true_x[:N], true_y[:N]
    px, py = pred_x[:N], pred_y[:N]

    rmse_x = float(np.sqrt(np.mean((tx - px) ** 2)))
    rmse_y = float(np.sqrt(np.mean((ty - py) ** 2)))
    rmse_total = float(np.sqrt(np.mean((tx - px) ** 2 + (ty - py) ** 2)))

    # Directed Hausdorff: max over true points of min dist to simulated path
    def directed_hausdorff(a_x, a_y, b_x, b_y):
        max_min = 0.0
        for ax_i, ay_i in zip(a_x, a_y):
            dists = np.sqrt((b_x - ax_i) ** 2 + (b_y - ay_i) ** 2)
            max_min = max(max_min, dists.min())
        return max_min

    h1 = directed_hausdorff(tx, ty, px, py)
    h2 = directed_hausdorff(px, py, tx, ty)
    hausdorff = float(max(h1, h2))

    return {
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_total": rmse_total,
        "hausdorff": hausdorff,
    }