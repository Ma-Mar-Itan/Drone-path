"""
src/pipeline.py
---------------
Converts raw canvas strokes → resampled, smoothed, differentiated trajectory.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def resample_arc_length(xs: np.ndarray, ys: np.ndarray, N: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a polyline to N points equally spaced by arc-length.

    Parameters
    ----------
    xs, ys : raw pixel coordinates (1-D arrays)
    N      : number of output samples

    Returns
    -------
    xs_r, ys_r : resampled coordinates
    """
    if len(xs) < 3:
        raise ValueError("Need at least 3 points to resample.")

    # Cumulative arc-length
    dx = np.diff(xs)
    dy = np.diff(ys)
    seg_lens = np.sqrt(dx ** 2 + dy ** 2)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = arc[-1]

    if total < 1e-6:
        raise ValueError("Path has zero length.")

    # Remove duplicate arc-length values (keep unique)
    _, unique_idx = np.unique(arc, return_index=True)
    arc = arc[unique_idx]
    xs = xs[unique_idx]
    ys = ys[unique_idx]

    # Uniform arc-length parameterisation
    s_uniform = np.linspace(0, total, N)
    fx = interp1d(arc, xs, kind="linear")
    fy = interp1d(arc, ys, kind="linear")
    return fx(s_uniform), fy(s_uniform)


def smooth_and_differentiate(
    xs: np.ndarray,
    ys: np.ndarray,
    dt: float = 0.05,
    window_len: int = 11,
    poly_order: int = 3,
) -> dict:
    """
    Apply Savitzky-Golay smoothing and compute derivatives up to 2nd order.

    Parameters
    ----------
    xs, ys      : resampled coordinates
    dt          : time step (seconds per sample)
    window_len  : SG filter window (odd integer)
    poly_order  : SG polynomial order

    Returns
    -------
    dict with keys: x, y, vx, vy, ax, ay, t
    """
    N = len(xs)
    # Ensure odd window not larger than data
    wl = min(window_len, N if N % 2 == 1 else N - 1)
    wl = max(wl, poly_order + 2 if (poly_order + 2) % 2 == 1 else poly_order + 3)

    x_s = savgol_filter(xs, wl, poly_order)
    y_s = savgol_filter(ys, wl, poly_order)

    # Derivatives via SG filter (delta=dt for physical units)
    vx = savgol_filter(xs, wl, poly_order, deriv=1, delta=dt)
    vy = savgol_filter(ys, wl, poly_order, deriv=1, delta=dt)
    ax = savgol_filter(xs, wl, poly_order, deriv=2, delta=dt)
    ay = savgol_filter(ys, wl, poly_order, deriv=2, delta=dt)

    t = np.arange(N) * dt

    return {"x": x_s, "y": y_s, "vx": vx, "vy": vy, "ax": ax, "ay": ay, "t": t}


def compute_safety_metrics(data: dict) -> dict:
    """
    Compute speed, acceleration magnitude, and curvature along the trajectory.

    Returns
    -------
    dict with keys: speed, accel_mag, curvature
    """
    vx, vy = data["vx"], data["vy"]
    ax, ay = data["ax"], data["ay"]

    speed = np.sqrt(vx ** 2 + vy ** 2)
    accel_mag = np.sqrt(ax ** 2 + ay ** 2)

    # Signed curvature κ = (vx*ay - vy*ax) / speed^3
    denom = speed ** 3
    denom = np.where(denom < 1e-8, 1e-8, denom)
    curvature = np.abs((vx * ay - vy * ax) / denom)

    return {"speed": speed, "accel_mag": accel_mag, "curvature": curvature}


def safety_mask(metrics: dict, max_speed: float, max_accel: float, max_curvature: float) -> np.ndarray:
    """Return boolean mask: True where *any* threshold is exceeded."""
    unsafe = (
        (metrics["speed"] > max_speed)
        | (metrics["accel_mag"] > max_accel)
        | (metrics["curvature"] > max_curvature)
    )
    return unsafe


def safety_summary(metrics: dict, unsafe: np.ndarray) -> dict:
    """Compute a human-readable safety summary."""
    N = len(unsafe)
    pct_unsafe = 100.0 * unsafe.sum() / N
    violations = np.where(unsafe)[0].tolist()
    return {
        "pct_unsafe": pct_unsafe,
        "max_speed": float(metrics["speed"].max()),
        "max_accel": float(metrics["accel_mag"].max()),
        "max_curvature": float(metrics["curvature"].max()),
        "violation_indices": violations,
    }


def normalize_data(data: dict) -> tuple[dict, dict]:
    """
    Zero-mean / unit-std normalise state features.

    Returns normalised data dict and stats dict for inversion.
    """
    keys = ["x", "y", "vx", "vy", "ax", "ay"]
    stats = {}
    normed = {}
    for k in keys:
        mu = data[k].mean()
        sigma = data[k].std() + 1e-8
        normed[k] = (data[k] - mu) / sigma
        stats[k] = (mu, sigma)
    normed["t"] = data["t"]
    return normed, stats


def denormalize(arr: np.ndarray, stats: tuple) -> np.ndarray:
    """Inverse-transform a normalised array."""
    mu, sigma = stats
    return arr * sigma + mu