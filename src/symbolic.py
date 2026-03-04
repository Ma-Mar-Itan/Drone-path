"""
src/symbolic.py
---------------
Sparse symbolic regression via PySINDy to obtain interpretable equations:

    d²x/dt² = f(x, y, vx, vy)
    d²y/dt² = g(x, y, vx, vy)

We use PySINDy with a custom feature library:
    [1, x, y, vx, vy, x², y², xy, vx², vy², vx*vy, sin(x), sin(y), cos(x), cos(y)]

The threshold parameter controls sparsity (larger → fewer terms).
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Feature library helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "1",
    "x", "y", "vx", "vy",
    "x^2", "y^2", "xy",
    "vx^2", "vy^2", "vx*vy",
    "sin(x)", "sin(y)",
    "cos(x)", "cos(y)",
]


def build_feature_matrix(x: np.ndarray, y: np.ndarray,
                          vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """
    Build the feature matrix Θ (N × F) from state variables.

    Columns correspond to FEATURE_NAMES.
    """
    ones = np.ones_like(x)
    Theta = np.column_stack([
        ones,
        x, y, vx, vy,
        x ** 2, y ** 2, x * y,
        vx ** 2, vy ** 2, vx * vy,
        np.sin(x), np.sin(y),
        np.cos(x), np.cos(y),
    ])
    return Theta


# ---------------------------------------------------------------------------
# PySINDy-based fitter (preferred)
# ---------------------------------------------------------------------------

def fit_sindy(data: dict, threshold: float = 0.05) -> Optional[object]:
    """
    Fit a SINDy model using PySINDy.

    Parameters
    ----------
    data      : smoothed trajectory dict (original-scale)
    threshold : STLSQ sparsity threshold (higher → fewer terms)

    Returns
    -------
    sindy_model or None if PySINDy unavailable
    """
    try:
        import pysindy as ps
    except ImportError:
        return None

    x = data["x"]
    y = data["y"]
    vx = data["vx"]
    vy = data["vy"]
    ax = data["ax"]
    ay = data["ay"]
    dt = data["t"][1] - data["t"][0]

    # State matrix: rows = samples, cols = [x, y, vx, vy]
    # We'll treat [ax, ay] as the "derivative" of [vx, vy], so we model the
    # second-order system as first-order on velocity.
    # PySINDy normally differentiates X internally, but here we supply
    # pre-computed derivatives to avoid double-differentiating.

    # Build custom feature library matching FEATURE_NAMES
    feature_lib = ps.CustomLibrary(
        library_functions=[
            lambda x, y, vx, vy: np.ones(len(x)),
            lambda x, y, vx, vy: x,
            lambda x, y, vx, vy: y,
            lambda x, y, vx, vy: vx,
            lambda x, y, vx, vy: vy,
            lambda x, y, vx, vy: x ** 2,
            lambda x, y, vx, vy: y ** 2,
            lambda x, y, vx, vy: x * y,
            lambda x, y, vx, vy: vx ** 2,
            lambda x, y, vx, vy: vy ** 2,
            lambda x, y, vx, vy: vx * vy,
            lambda x, y, vx, vy: np.sin(x),
            lambda x, y, vx, vy: np.sin(y),
            lambda x, y, vx, vy: np.cos(x),
            lambda x, y, vx, vy: np.cos(y),
        ],
        function_names=[
            lambda x, y, vx, vy: "1",
            lambda x, y, vx, vy: "x",
            lambda x, y, vx, vy: "y",
            lambda x, y, vx, vy: "vx",
            lambda x, y, vx, vy: "vy",
            lambda x, y, vx, vy: "x^2",
            lambda x, y, vx, vy: "y^2",
            lambda x, y, vx, vy: "xy",
            lambda x, y, vx, vy: "vx^2",
            lambda x, y, vx, vy: "vy^2",
            lambda x, y, vx, vy: "vx*vy",
            lambda x, y, vx, vy: "sin(x)",
            lambda x, y, vx, vy: "sin(y)",
            lambda x, y, vx, vy: "cos(x)",
            lambda x, y, vx, vy: "cos(y)",
        ],
    )

    # STLSQ optimizer with user-supplied threshold
    optimizer = ps.STLSQ(threshold=threshold, alpha=1e-5)

    # State input for library: (N, 4) — the 4 input args above
    U = np.stack([x, y, vx, vy], axis=1)

    # Targets: acceleration (N, 2)
    dU = np.stack([ax, ay], axis=1)

    model = ps.SINDy(
        feature_library=feature_lib,
        optimizer=optimizer,
    )
    # Fit with pre-computed derivatives (no internal differentiation)
    model.fit(U, t=dt, x_dot=dU)
    return model


# ---------------------------------------------------------------------------
# Fallback: manual sparse regression (STLSQ)
# ---------------------------------------------------------------------------

def _stlsq(Theta: np.ndarray, target: np.ndarray,
            threshold: float, n_iter: int = 20) -> np.ndarray:
    """
    Sequential Thresholded Least Squares (STLSQ).

    Iteratively zeroes out small coefficients and re-fits on active features.
    """
    n_features = Theta.shape[1]
    coeffs = np.linalg.lstsq(Theta, target, rcond=None)[0]

    for _ in range(n_iter):
        active = np.abs(coeffs) > threshold
        if not active.any():
            coeffs = np.zeros(n_features)
            break
        Theta_active = Theta[:, active]
        coeffs_active = np.linalg.lstsq(Theta_active, target, rcond=None)[0]
        coeffs = np.zeros(n_features)
        coeffs[active] = coeffs_active

    return coeffs


class SparseEquation:
    """
    Holds coefficients for one equation: target = Θ · w
    """

    def __init__(self, coeffs: np.ndarray, feature_names: list[str],
                 target_name: str):
        self.coeffs = coeffs
        self.feature_names = feature_names
        self.target_name = target_name

    def __str__(self) -> str:
        terms = []
        for c, name in zip(self.coeffs, self.feature_names):
            if abs(c) > 1e-10:
                if name == "1":
                    terms.append(f"{c:+.4f}")
                else:
                    terms.append(f"{c:+.4f}·{name}")
        if not terms:
            return f"{self.target_name} = 0"
        rhs = " ".join(terms)
        return f"{self.target_name} = {rhs}"

    def predict(self, x, y, vx, vy) -> np.ndarray:
        Theta = build_feature_matrix(x, y, vx, vy)
        return Theta @ self.coeffs


class ManualSINDy:
    """Fallback sparse regression when PySINDy is unavailable."""

    def __init__(self, eq_x: SparseEquation, eq_y: SparseEquation):
        self.eq_x = eq_x
        self.eq_y = eq_y

    def equations_str(self) -> tuple[str, str]:
        return str(self.eq_x), str(self.eq_y)

    def predict(self, x, y, vx, vy) -> tuple[np.ndarray, np.ndarray]:
        return self.eq_x.predict(x, y, vx, vy), self.eq_y.predict(x, y, vx, vy)


def fit_sparse_manual(data: dict, threshold: float = 0.05) -> "ManualSINDy":
    """
    Fit sparse regression equations for ax and ay using STLSQ.
    """
    x, y, vx, vy = data["x"], data["y"], data["vx"], data["vy"]
    ax, ay = data["ax"], data["ay"]

    Theta = build_feature_matrix(x, y, vx, vy)
    w_ax = _stlsq(Theta, ax, threshold=threshold)
    w_ay = _stlsq(Theta, ay, threshold=threshold)

    eq_x = SparseEquation(w_ax, FEATURE_NAMES, "d²x/dt²")
    eq_y = SparseEquation(w_ay, FEATURE_NAMES, "d²y/dt²")
    return ManualSINDy(eq_x, eq_y)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def fit_equations(data: dict, threshold: float = 0.05) -> tuple:
    """
    Fit symbolic dynamics equations.

    Tries PySINDy first; falls back to manual STLSQ if unavailable.

    Returns
    -------
    model      : fitted model (PySINDy model or ManualSINDy)
    eq_str_x   : human-readable equation for d²x/dt²
    eq_str_y   : human-readable equation for d²y/dt²
    use_sindy  : bool — True if PySINDy was used
    """
    sindy_model = fit_sindy(data, threshold=threshold)

    if sindy_model is not None:
        # Extract equation strings from PySINDy
        try:
            eqs = sindy_model.equations()
            # eqs[0] = d/dt(vx), eqs[1] = d/dt(vy)  i.e. ax and ay
            eq_str_x = f"d²x/dt² = {eqs[0]}" if eqs else "d²x/dt² = (see model)"
            eq_str_y = f"d²y/dt² = {eqs[1]}" if len(eqs) > 1 else "d²y/dt² = (see model)"
        except Exception:
            eq_str_x = "d²x/dt² = (PySINDy model, see console)"
            eq_str_y = "d²y/dt² = (PySINDy model, see console)"
        return sindy_model, eq_str_x, eq_str_y, True

    # Fallback
    manual_model = fit_sparse_manual(data, threshold=threshold)
    eq_str_x, eq_str_y = manual_model.equations_str()
    return manual_model, eq_str_x, eq_str_y, False


def predict_accelerations(model, data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Use the fitted symbolic model to predict accelerations.

    Returns ax_pred, ay_pred (original units).
    """
    x, y, vx, vy = data["x"], data["y"], data["vx"], data["vy"]

    try:
        # PySINDy path
        import pysindy as ps
        if isinstance(model, ps.SINDy):
            U = np.stack([x, y, vx, vy], axis=1)
            accel = model.predict(U)
            return accel[:, 0], accel[:, 1]
    except Exception:
        pass

    # ManualSINDy fallback
    return model.predict(x, y, vx, vy)