"""
src/symbolic.py
---------------
Sparse symbolic regression via SINDy to obtain interpretable equations.

KEY FIX: All fitting and simulation is done in NORMALISED space (zero-mean,
unit-std per feature). The normalisation stats are stored and used to:
  1. Normalise inputs before fitting
  2. Normalise inputs before predicting (in simulator)
  3. De-normalise outputs back to pixel units for display

This prevents the sin/cos features from wrapping at nonsensical pixel values
and keeps all regression coefficients numerically well-conditioned.
"""

import numpy as np
from typing import Optional


FEATURE_NAMES = [
    "1",
    "x", "y", "vx", "vy",
    "x^2", "y^2", "xy",
    "vx^2", "vy^2", "vx*vy",
    "sin(x)", "sin(y)",
    "cos(x)", "cos(y)",
]


def _compute_norm_stats(data: dict) -> dict:
    """Compute mean and std for each state feature."""
    stats = {}
    for k in ["x", "y", "vx", "vy", "ax", "ay"]:
        mu  = float(data[k].mean())
        sig = float(data[k].std()) + 1e-8
        stats[k] = (mu, sig)
    return stats


def _normalise(data: dict, stats: dict) -> dict:
    """Return a copy of data with state features normalised."""
    normed = {}
    for k in ["x", "y", "vx", "vy", "ax", "ay"]:
        mu, sig = stats[k]
        normed[k] = (data[k] - mu) / sig
    normed["t"] = data["t"]
    return normed


def build_feature_matrix(x, y, vx, vy) -> np.ndarray:
    """Build feature library matrix Theta (N x F) from *normalised* state."""
    ones = np.ones_like(x)
    return np.column_stack([
        ones,
        x, y, vx, vy,
        x**2, y**2, x*y,
        vx**2, vy**2, vx*vy,
        np.sin(x), np.sin(y),
        np.cos(x), np.cos(y),
    ])


# ---------------------------------------------------------------------------
# PySINDy wrapper
# ---------------------------------------------------------------------------

def fit_sindy(data_norm: dict, threshold: float = 0.05):
    """Fit SINDy on already-normalised data. Returns model or None."""
    try:
        import pysindy as ps
    except ImportError:
        return None

    x, y   = data_norm["x"],  data_norm["y"]
    vx, vy = data_norm["vx"], data_norm["vy"]
    ax, ay = data_norm["ax"], data_norm["ay"]
    dt     = float(data_norm["t"][1] - data_norm["t"][0])

    feature_lib = ps.CustomLibrary(
        library_functions=[
            lambda x,y,vx,vy: np.ones(len(x)),
            lambda x,y,vx,vy: x,   lambda x,y,vx,vy: y,
            lambda x,y,vx,vy: vx,  lambda x,y,vx,vy: vy,
            lambda x,y,vx,vy: x**2,  lambda x,y,vx,vy: y**2,
            lambda x,y,vx,vy: x*y,
            lambda x,y,vx,vy: vx**2, lambda x,y,vx,vy: vy**2,
            lambda x,y,vx,vy: vx*vy,
            lambda x,y,vx,vy: np.sin(x), lambda x,y,vx,vy: np.sin(y),
            lambda x,y,vx,vy: np.cos(x), lambda x,y,vx,vy: np.cos(y),
        ],
        function_names=[
            lambda *_: "1",
            lambda *_: "x",   lambda *_: "y",
            lambda *_: "vx",  lambda *_: "vy",
            lambda *_: "x^2", lambda *_: "y^2",
            lambda *_: "xy",
            lambda *_: "vx^2",lambda *_: "vy^2",
            lambda *_: "vx*vy",
            lambda *_: "sin(x)", lambda *_: "sin(y)",
            lambda *_: "cos(x)", lambda *_: "cos(y)",
        ],
    )

    optimizer = ps.STLSQ(threshold=threshold, alpha=1e-5)
    U  = np.stack([x, y, vx, vy], axis=1)
    dU = np.stack([ax, ay], axis=1)

    model = ps.SINDy(feature_library=feature_lib, optimizer=optimizer)
    model.fit(U, t=dt, x_dot=dU)
    return model


# ---------------------------------------------------------------------------
# Manual STLSQ fallback
# ---------------------------------------------------------------------------

def _stlsq(Theta, target, threshold, n_iter=20):
    n_feat = Theta.shape[1]
    coeffs = np.linalg.lstsq(Theta, target, rcond=None)[0]
    for _ in range(n_iter):
        active = np.abs(coeffs) > threshold
        if not active.any():
            return np.zeros(n_feat)
        c = np.linalg.lstsq(Theta[:, active], target, rcond=None)[0]
        coeffs = np.zeros(n_feat)
        coeffs[active] = c
    return coeffs


class SparseEquation:
    def __init__(self, coeffs, feature_names, target_name):
        self.coeffs = coeffs
        self.feature_names = feature_names
        self.target_name = target_name

    def __str__(self):
        terms = []
        for c, name in zip(self.coeffs, self.feature_names):
            if abs(c) > 1e-10:
                terms.append(f"{c:+.4f}" if name == "1" else f"{c:+.4f}·{name}")
        return f"{self.target_name} = " + (" ".join(terms) if terms else "0")

    def predict(self, x, y, vx, vy):
        return build_feature_matrix(x, y, vx, vy) @ self.coeffs


class ManualSINDy:
    def __init__(self, eq_x, eq_y):
        self.eq_x = eq_x
        self.eq_y = eq_y

    def equations_str(self):
        return str(self.eq_x), str(self.eq_y)

    def predict(self, x, y, vx, vy):
        return self.eq_x.predict(x, y, vx, vy), self.eq_y.predict(x, y, vx, vy)


def fit_sparse_manual(data_norm, threshold=0.05):
    Theta = build_feature_matrix(data_norm["x"], data_norm["y"],
                                  data_norm["vx"], data_norm["vy"])
    w_ax = _stlsq(Theta, data_norm["ax"], threshold)
    w_ay = _stlsq(Theta, data_norm["ay"], threshold)
    return ManualSINDy(
        SparseEquation(w_ax, FEATURE_NAMES, "d2x/dt2"),
        SparseEquation(w_ay, FEATURE_NAMES, "d2y/dt2"),
    )


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def fit_equations(data: dict, threshold: float = 0.05):
    """
    Normalise data, fit symbolic equations, return model + norm stats + strings.

    Returns
    -------
    model      : fitted model
    norm_stats : dict of (mean, std) per feature — MUST be passed to simulator
    eq_str_x   : human-readable equation string
    eq_str_y   : human-readable equation string
    used_sindy : bool
    """
    norm_stats  = _compute_norm_stats(data)
    data_norm   = _normalise(data, norm_stats)

    sindy_model = fit_sindy(data_norm, threshold=threshold)

    if sindy_model is not None:
        try:
            eqs = sindy_model.equations()
            eq_str_x = f"d2x/dt2 (norm) = {eqs[0]}" if eqs else "d2x/dt2 = (model)"
            eq_str_y = f"d2y/dt2 (norm) = {eqs[1]}" if len(eqs) > 1 else "d2y/dt2 = (model)"
        except Exception:
            eq_str_x = "d2x/dt2 = (PySINDy — see Training tab)"
            eq_str_y = "d2y/dt2 = (PySINDy — see Training tab)"
        return sindy_model, norm_stats, eq_str_x, eq_str_y, True

    manual = fit_sparse_manual(data_norm, threshold=threshold)
    eq_str_x, eq_str_y = manual.equations_str()
    return manual, norm_stats, eq_str_x, eq_str_y, False


def predict_accelerations(model, data: dict, norm_stats: dict):
    """
    Predict accelerations in ORIGINAL pixel units.

    Normalises inputs → model prediction (normalised accel) → de-normalise output.
    """
    mu_x,  sig_x  = norm_stats["x"]
    mu_y,  sig_y  = norm_stats["y"]
    mu_vx, sig_vx = norm_stats["vx"]
    mu_vy, sig_vy = norm_stats["vy"]
    mu_ax, sig_ax = norm_stats["ax"]
    mu_ay, sig_ay = norm_stats["ay"]

    xn  = (data["x"]  - mu_x)  / sig_x
    yn  = (data["y"]  - mu_y)  / sig_y
    vxn = (data["vx"] - mu_vx) / sig_vx
    vyn = (data["vy"] - mu_vy) / sig_vy

    try:
        import pysindy as ps
        if isinstance(model, ps.SINDy):
            U = np.stack([xn, yn, vxn, vyn], axis=1)
            pred_n = model.predict(U)
            return pred_n[:, 0] * sig_ax + mu_ax, pred_n[:, 1] * sig_ay + mu_ay
    except Exception:
        pass

    ax_n, ay_n = model.predict(xn, yn, vxn, vyn)
    return ax_n * sig_ax + mu_ax, ay_n * sig_ay + mu_ay