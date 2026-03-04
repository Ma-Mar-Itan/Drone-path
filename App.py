"""
app.py  —  Drone Path Equation-of-Motion Demo  (v2 — post-audit)
=================================================================
Fixes applied from Technical Audit:
  [BUG]  Simulation scaling: all SINDy fitting & RK4 done in normalised space
  [BUG]  Safety thresholds: sensible defaults scaled to canvas pixel units
  [UX]   Unified Clear button — resets drawing + state together
  [UX]   Progress bar with descriptive steps
  [UX]   Graceful error messages for short paths
  [UX]   Slider units and help text on every control
  [UX]   Results banner after fitting so user knows to scroll
  [UX]   Responsive metric cards
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import streamlit as st
import json, time

from src.pipeline import (
    resample_arc_length, smooth_and_differentiate,
    compute_safety_metrics, safety_mask, safety_summary,
)
from src.gnn_model import train_gnn, gnn_predict
from src.symbolic import fit_equations, predict_accelerations
from src.simulator import simulate, compute_error_metrics

st.set_page_config(page_title="Drone Path · Equation Finder", page_icon="🚁", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0d1117;color:#e6edf3;font-family:'Inter','Segoe UI',sans-serif}
[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d}
[data-testid="stSidebar"] *{color:#c9d1d9!important}
.hero{background:linear-gradient(135deg,#0d1117 0%,#1a2332 50%,#0d2137 100%);border:1px solid #30363d;border-radius:16px;padding:36px 40px;margin-bottom:24px;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 70% 50%,rgba(56,139,253,0.08) 0%,transparent 70%)}
.hero-title{font-size:2.4rem;font-weight:800;background:linear-gradient(90deg,#58a6ff,#79c0ff,#a5d6ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 8px 0}
.hero-sub{font-size:1rem;color:#8b949e;max-width:620px;margin:0 0 20px 0}
.hero-badges{display:flex;gap:10px;flex-wrap:wrap}
.badge{background:#21262d;border:1px solid #30363d;border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#8b949e;font-weight:500}
.badge-blue{border-color:#388bfd44;color:#58a6ff;background:#388bfd11}
.badge-green{border-color:#3fb95044;color:#56d364;background:#3fb95011}
.badge-purple{border-color:#bc8cff44;color:#bc8cff;background:#bc8cff11}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin-bottom:16px}
.card-title{font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#8b949e;margin-bottom:6px}
.card-value{font-size:2rem;font-weight:700;color:#e6edf3}
.card-value-sm{font-size:1.2rem;font-weight:600;color:#e6edf3}
.card-sub{font-size:0.82rem;color:#8b949e;margin-top:2px}
.eq-box{background:#0d1117;border:1px solid #388bfd44;border-left:4px solid #388bfd;border-radius:10px;padding:20px 24px;font-family:'JetBrains Mono','Fira Code','Courier New',monospace;font-size:1rem;color:#79c0ff;line-height:2;margin:12px 0}
.eq-label{font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#388bfd;margin-bottom:8px}
.math-box{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;margin:10px 0;font-size:0.88rem;color:#c9d1d9;line-height:1.7}
.math-formula{background:#0d1117;border-radius:6px;padding:8px 14px;margin:8px 0;font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:#a5d6ff;border-left:3px solid #58a6ff44}
.step-num{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;background:#388bfd22;border:1px solid #388bfd55;color:#58a6ff;font-size:0.78rem;font-weight:700;margin-right:8px;flex-shrink:0}
.step-row{display:flex;align-items:flex-start;margin:8px 0}
.safety-safe{color:#56d364;font-weight:700;font-size:1.8rem}
.safety-warn{color:#e3b341;font-weight:700;font-size:1.8rem}
.safety-danger{color:#f85149;font-weight:700;font-size:1.8rem}
.safety-bar-bg{background:#21262d;border-radius:4px;height:8px;margin:6px 0}
.safety-bar-fill{height:8px;border-radius:4px;background:linear-gradient(90deg,#56d364,#e3b341,#f85149)}
.pipeline{display:flex;align-items:center;overflow-x:auto;padding:16px 0;margin:12px 0}
.pipe-step{background:#21262d;border:1px solid #30363d;border-radius:10px;padding:12px 16px;text-align:center;min-width:95px;flex-shrink:0}
.pipe-step-icon{font-size:1.3rem}
.pipe-step-label{font-size:0.7rem;color:#8b949e;margin-top:4px;font-weight:600}
.pipe-arrow{color:#388bfd;font-size:1.1rem;padding:0 5px;flex-shrink:0}
.pipe-step-active{background:#1c2d3d;border-color:#388bfd;box-shadow:0 0 12px rgba(56,139,253,0.2)}
[data-testid="stTabs"] button{color:#8b949e!important;font-weight:500}
[data-testid="stTabs"] button[aria-selected="true"]{color:#58a6ff!important;border-bottom:2px solid #388bfd!important;background:#161b22!important}
.stButton>button{background:#21262d;border:1px solid #30363d;color:#e6edf3!important;border-radius:8px;font-weight:600;transition:all 0.2s}.stButton>button p,.stButton>button span{color:#e6edf3!important}
.stButton>button:hover{background:#30363d;border-color:#58a6ff;color:#58a6ff!important}.stButton>button:hover p,.stButton>button:hover span{color:#58a6ff!important}
[data-testid="stMetric"]{background:#161b22;border-radius:10px;padding:12px 16px}
[data-testid="stMetricLabel"]{color:#8b949e!important;font-size:0.78rem!important}
[data-testid="stMetricValue"]{color:#e6edf3!important;font-size:1.4rem!important}
hr{border-color:#30363d!important}
.section-header{font-size:1rem;font-weight:700;color:#e6edf3;border-left:3px solid #388bfd;padding-left:12px;margin:20px 0 12px 0}
.result-banner{background:#1c2d3d;border:1px solid #388bfd55;border-radius:10px;padding:14px 20px;margin:16px 0;color:#79c0ff;font-weight:600}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = dict(
    raw_xs=None, raw_ys=None, data=None,
    gnn_model=None, gnn_loss=[],
    sym_model=None, norm_stats=None, eq_x="", eq_y="",
    sim_data=None, fitted=False, simulated=False,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    st.markdown("**📐 Data Pipeline**")
    N_points = st.slider("Trajectory samples N", 40, 200, 80, 10,
        help="Number of arc-length-equalised time steps. More = finer, slower.")
    dt = st.slider("Time step Δt (s)", 0.01, 0.20, 0.05, 0.01,
        help="Seconds between samples. Velocity units = px/s, acceleration = px/s².")

    st.markdown("**🧠 GNN**")
    epochs = st.slider("Max epochs", 100, 2000, 600, 100,
        help="Early stopping usually fires before this limit.")
    hidden = st.slider("Hidden units", 16, 128, 64, 16,
        help="Width of GNN internal layers. Larger = more expressive, slower.")
    k_nbrs = st.slider("k-nearest neighbours", 1, 5, 3,
        help="How many time steps each node can see in each direction.")

    st.markdown("**📝 SINDy Sparsity**")
    sparsity = st.slider("Threshold λ", 0.001, 0.5, 0.05, 0.001, format="%.3f",
        help="Higher = fewer terms in equation. Start at 0.05, raise if simulation diverges.")

    st.markdown("---")
    st.markdown("**🛡️ Safety Thresholds**")
    st.caption("Units: speed in px/s, acceleration in px/s², curvature in 1/px")
    max_speed = st.slider("Max speed (px/s)", 10.0, 2000.0, 300.0, 10.0,
        help="Typical drawn path: 100–600 px/s. Raise if everything shows as unsafe.")
    max_accel = st.slider("Max acceleration (px/s²)", 10.0, 5000.0, 1000.0, 50.0,
        help="Typical drawn path: 200–3000 px/s². Raise if everything shows as unsafe.")
    max_curv  = st.slider("Max curvature (1/px)", 0.001, 0.2, 0.05, 0.001, format="%.3f",
        help="κ = 1/turning-radius. Tight corners have high κ. Typical threshold: 0.02–0.08.")

    st.markdown("---")
    st.markdown("""<div style='font-size:0.78rem;color:#8b949e;line-height:1.8'>
<b style='color:#58a6ff'>Graph</b> — nodes=time steps; edges=temporal neighbours<br>
<b style='color:#58a6ff'>GNN</b> — message passing predicts acceleration<br>
<b style='color:#58a6ff'>SINDy</b> — sparse regression → readable equation<br>
<b style='color:#58a6ff'>RK4</b> — integrate equations forward in time<br>
<b style='color:#58a6ff'>All fitting</b> done in normalised space (zero-mean, unit-std)
</div>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🚁 Drone Path · Equation Finder</div>
  <div class="hero-sub">Draw a 2-D drone trajectory. A Graph Neural Network learns its physics in seconds.
  Sparse symbolic regression distils it into human-readable equations of motion. Re-fly the path. Analyse safety.</div>
  <div class="hero-badges">
    <span class="badge badge-blue">Graph Neural Network</span>
    <span class="badge badge-green">SINDy Symbolic Regression</span>
    <span class="badge badge-purple">RK4 Simulation</span>
    <span class="badge">Safety Analysis</span>
    <span class="badge">CPU Only · No GPU needed</span>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="pipeline">
  <div class="pipe-step pipe-step-active"><div class="pipe-step-icon">✏️</div><div class="pipe-step-label">Draw Path</div></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step"><div class="pipe-step-icon">📐</div><div class="pipe-step-label">Resample &amp; Smooth</div></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step"><div class="pipe-step-icon">🕸️</div><div class="pipe-step-label">Temporal Graph</div></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step"><div class="pipe-step-icon">🧠</div><div class="pipe-step-label">Train GNN</div></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step"><div class="pipe-step-icon">📝</div><div class="pipe-step-label">SINDy Equations</div></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step"><div class="pipe-step-icon">▶️</div><div class="pipe-step-label">RK4 Simulate</div></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step"><div class="pipe-step-icon">🛡️</div><div class="pipe-step-label">Safety Check</div></div>
</div>""", unsafe_allow_html=True)

# ── Drawing area ──────────────────────────────────────────────────────────────
col_canvas, col_ctrl = st.columns([3, 1])

with col_canvas:
    st.markdown('<div class="section-header">✏️ Draw Your Trajectory</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#8b949e'>Click and drag to sketch a drone flight path. Draw a smooth continuous stroke for best results.</small>", unsafe_allow_html=True)

    canvas_available = False
    try:
        from streamlit_drawable_canvas import st_canvas
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)", stroke_width=3,
            stroke_color="#388bfd", background_color="#f0f4f8",
            height=380, width=680, drawing_mode="freedraw", key="canvas",
        )
        canvas_available = True
        if canvas_result.json_data is not None:
            all_xs, all_ys = [], []
            for obj in canvas_result.json_data.get("objects", []):
                if obj.get("type") == "path":
                    for cmd in obj.get("path", []):
                        if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M","L","Q"):
                            all_xs.append(float(cmd[1]))
                            all_ys.append(float(cmd[2]))
            if len(all_xs) > 5:
                st.session_state.raw_xs = np.array(all_xs)
                st.session_state.raw_ys = np.array(all_ys)
    except ImportError:
        t_s = np.linspace(0, 2*np.pi, 300)
        st.session_state.raw_xs = 200 + 150*np.sin(t_s)
        st.session_state.raw_ys = 200 + 80*np.sin(2*t_s)
        st.info("📌 Using built-in figure-8 demo path. Install `streamlit-drawable-canvas` for mouse drawing.")

with col_ctrl:
    st.markdown('<div class="section-header">📊 Path Info</div>', unsafe_allow_html=True)
    if st.session_state.raw_xs is not None:
        n_raw = len(st.session_state.raw_xs)
        xr = st.session_state.raw_xs.max() - st.session_state.raw_xs.min()
        yr = st.session_state.raw_ys.max() - st.session_state.raw_ys.min()
        st.markdown(f"""<div class="card">
<div class="card-title">Raw points</div><div class="card-value">{n_raw}</div>
</div><div class="card">
<div class="card-title">Canvas span</div>
<div class="card-value-sm">{xr:.0f} × {yr:.0f} px</div>
</div>""", unsafe_allow_html=True)
        if n_raw < 10:
            st.warning("Path too short — draw a longer stroke.")
    else:
        st.markdown('<div class="card"><div class="card-sub">No path drawn yet</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fit_btn    = st.button("🔧 Fit Model",   type="primary", use_container_width=True,
                           help="Resample → Smooth → Train GNN → Fit SINDy equations")
    sim_btn    = st.button("▶️ Simulate",    use_container_width=True,
                           help="Integrate learned equations forward with RK4")
    # AUDIT FIX: single unified Clear/Reset button
    clear_btn  = st.button("🗑️ Reset All",   use_container_width=True,
                           help="Clears the drawing AND all fitted results")
    export_btn = st.button("💾 Export JSON", use_container_width=True,
                           help="Download trajectory data and equations as JSON")

# ── Reset (unified — fixes audit finding) ────────────────────────────────────
if clear_btn:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

# ── Fit Model ─────────────────────────────────────────────────────────────────
if fit_btn:
    raw_xs = st.session_state.raw_xs
    raw_ys = st.session_state.raw_ys

    # AUDIT FIX: graceful error for short paths
    if raw_xs is None or len(raw_xs) < 10:
        st.error("⚠️ Path too short. Please draw a longer, continuous stroke (at least 10 points).")
        st.stop()

    if len(raw_xs) < 20:
        st.warning("Path is quite short — results may be less accurate. Try drawing a longer stroke.")

    prog = st.progress(0, text="📐 Step 1/3 — Resampling and smoothing path…")
    try:
        xs_r, ys_r = resample_arc_length(raw_xs, raw_ys, N=N_points)
        data = smooth_and_differentiate(xs_r, ys_r, dt=dt)
        st.session_state.data = data
    except ValueError as e:
        st.error(f"⚠️ Path error: {e}. Please draw a longer, more distinct stroke.")
        st.stop()
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

    prog.progress(25, text="🧠 Step 2/3 — Training Graph Neural Network…")
    try:
        t0 = time.time()
        gnn_model, loss_hist, x_mu, x_std, y_mu, y_std = train_gnn(
            data, epochs=epochs, hidden=hidden, k_neighbors=k_nbrs)
        elapsed = time.time() - t0
        st.session_state.gnn_model = (gnn_model, x_mu, x_std, y_mu, y_std)
        st.session_state.gnn_loss  = loss_hist
    except Exception as e:
        st.error(f"GNN training error: {e}")
        st.stop()

    prog.progress(70, text="📝 Step 3/3 — Fitting symbolic equations (SINDy)…")
    try:
        result = fit_equations(data, threshold=sparsity)
        if len(result) == 5:
            sym_model, norm_stats, eq_x, eq_y, used_sindy = result
        else:
            sym_model, eq_x, eq_y, used_sindy = result
            norm_stats = None
        st.session_state.sym_model  = sym_model
        st.session_state.norm_stats = norm_stats
        st.session_state.eq_x = eq_x
        st.session_state.eq_y = eq_y
    except Exception as e:
        st.error(f"SINDy error: {e}")
        st.stop()

    prog.progress(100, text="✅ Done!"); time.sleep(0.4); prog.empty()
    backend = "PySINDy" if used_sindy else "Manual STLSQ"
    st.success(f"✅ Model fitted in {elapsed:.1f}s · {len(loss_hist)} epochs · "
               f"Final loss: {loss_hist[-1]:.5f} · Backend: {backend}")
    # AUDIT FIX: tell user to scroll down
    st.markdown('<div class="result-banner">👇 Scroll down to see your results in the tabs below</div>',
                unsafe_allow_html=True)
    st.session_state.fitted    = True
    st.session_state.simulated = False

# ── Simulate ──────────────────────────────────────────────────────────────────
if sim_btn:
    if not st.session_state.fitted:
        st.warning("Fit the model first by clicking 🔧 Fit Model.")
    else:
        with st.spinner("▶️ Integrating equations forward (RK4)…"):
            data       = st.session_state.data
            norm_stats = st.session_state.norm_stats
            x0,  y0   = float(data["x"][0]),  float(data["y"][0])
            vx0, vy0  = float(data["vx"][0]), float(data["vy"][0])
            dt_val    = float(data["t"][1] - data["t"][0])
            try:
                sim_data = simulate(
                    st.session_state.sym_model, norm_stats,
                    x0, y0, vx0, vy0, dt=dt_val, N=len(data["x"]))
                st.session_state.sim_data  = sim_data
                st.session_state.simulated = True
                st.success("✅ Simulation complete — see the Trajectory tab below")
            except Exception as e:
                st.error(f"Simulation error: {e}")

# ── Export ────────────────────────────────────────────────────────────────────
if export_btn:
    if st.session_state.data is None:
        st.warning("No data to export. Fit the model first.")
    else:
        data = st.session_state.data
        payload = {
            "trajectory": {k: data[k].tolist() for k in ["x","y","vx","vy","ax","ay","t"]},
            "equations":  {"ax": st.session_state.eq_x, "ay": st.session_state.eq_y},
            "norm_stats": {k: list(v) for k, v in st.session_state.norm_stats.items()}
                          if st.session_state.norm_stats else {},
        }
        st.download_button("⬇️ Download JSON", json.dumps(payload, indent=2),
                           "drone_path.json", "application/json")

# ── Plot helpers ──────────────────────────────────────────────────────────────
DARK_BG="#0d1117"; DARK_AX="#161b22"; GRID_COL="#21262d"
SAFE_COL="#388bfd"; UNSAFE_C="#f85149"
SIM_SAFE="#3fb950"; SIM_UNSA="#e3b341"
TEXT_COL="#e6edf3"; MUTED="#8b949e"

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.7, alpha=0.8)
    if title:  ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COL, pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)

def plot_trajectory(data, unsafe, sim_data=None, sim_unsafe=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, "Trajectory  (blue=safe · red=unsafe · dashed=simulated)", "x (px)", "y (px)")
    x, y, N = data["x"], data["y"], len(data["x"])
    for i in range(N-1):
        ax.plot(x[i:i+2], y[i:i+2],
                color=UNSAFE_C if unsafe[i] else SAFE_COL,
                lw=2.2, solid_capstyle="round", zorder=3)
    cmap = plt.cm.cool
    for i in range(0, N, max(1, N//25)):
        c = mcolors.to_rgba(cmap(i/N), alpha=0.7)
        ax.scatter(x[i], y[i], color=[c], s=18, zorder=5, edgecolors="none")
    ax.scatter([x[0]], [y[0]], color="#56d364", s=120, zorder=7, label="Start")
    ax.scatter([x[-1]], [y[-1]], color="#bc8cff", s=140, zorder=7, label="End", marker="*")
    if sim_data is not None:
        sx, sy = sim_data["x"], sim_data["y"]; M = len(sx)
        su = sim_unsafe if sim_unsafe is not None else np.zeros(M, dtype=bool)
        for i in range(min(M-1, len(su)-1)):
            ax.plot(sx[i:i+2], sy[i:i+2],
                    color=SIM_UNSA if su[i] else SIM_SAFE,
                    lw=1.8, ls="--", alpha=0.85, zorder=4)
    patches = [
        mpatches.Patch(color=SAFE_COL, label="Drawn (safe)"),
        mpatches.Patch(color=UNSAFE_C, label="Drawn (unsafe)"),
    ]
    if sim_data is not None:
        patches += [
            mpatches.Patch(color=SIM_SAFE, label="Simulated (safe)"),
            mpatches.Patch(color=SIM_UNSA, label="Simulated (unsafe)"),
        ]
    patches += [mpatches.Patch(color="#56d364",label="Start"),
                mpatches.Patch(color="#bc8cff",label="End")]
    ax.legend(handles=patches, loc="upper right", fontsize=7.5,
              facecolor="#161b22", edgecolor="#30363d", labelcolor=TEXT_COL)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout(pad=1.5)
    return fig

def plot_accelerations(data, ax_gnn, ay_gnn, ax_sym, ay_sym):
    fig, axs = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    t = data["t"]
    for ax_p, truth, gv, sv, lbl in [
        (axs[0], data["ax"], ax_gnn, ax_sym, "d²x/dt²  (ax)  px/s²"),
        (axs[1], data["ay"], ay_gnn, ay_sym, "d²y/dt²  (ay)  px/s²"),
    ]:
        _style_ax(ax_p, ylabel=lbl)
        ax_p.plot(t, truth, color=TEXT_COL, lw=1.6, label="Ground truth (SG)", zorder=4)
        if gv is not None: ax_p.plot(t, gv, "--", color="#58a6ff", lw=1.3, alpha=0.9, label="GNN", zorder=3)
        if sv is not None: ax_p.plot(t, sv, ":", color="#f78166", lw=1.8, label="SINDy", zorder=5)
        ax_p.legend(fontsize=7.5, facecolor="#161b22", edgecolor="#30363d", labelcolor=TEXT_COL)
    axs[-1].set_xlabel("time (s)", fontsize=8, color=MUTED)
    fig.suptitle("Acceleration — Ground Truth vs GNN vs SINDy",
                 fontsize=10, fontweight="bold", color=TEXT_COL, y=1.01)
    plt.tight_layout(pad=1.2)
    return fig

def plot_safety_metrics(data, metrics, title):
    fig, axes = plt.subplots(3, 1, figsize=(9, 5), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    t = data["t"]
    series = [
        (metrics["speed"],     "Speed (px/s)",       "#58a6ff", max_speed),
        (metrics["accel_mag"], "Acceleration (px/s²)","#bc8cff", max_accel),
        (metrics["curvature"], "Curvature (1/px)",    "#3fb950", max_curv),
    ]
    for ax, (vals, label, color, thresh) in zip(axes, series):
        _style_ax(ax, ylabel=label)
        ax.plot(t, vals, color=color, lw=1.5, zorder=3)
        ax.axhline(thresh, color=UNSAFE_C, ls="--", lw=1.2,
                   label=f"limit = {thresh:.3g}", zorder=4)
        ax.fill_between(t, vals, thresh, where=(vals>thresh),
                        color=UNSAFE_C, alpha=0.18, zorder=2)
        ax.legend(fontsize=7.5, facecolor="#161b22", edgecolor="#30363d", labelcolor=TEXT_COL)
    axes[-1].set_xlabel("time (s)", fontsize=8, color=MUTED)
    fig.suptitle(title, fontsize=10, fontweight="bold", color=TEXT_COL, y=1.01)
    plt.tight_layout(pad=1.2)
    return fig

def plot_loss(loss_hist):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, "GNN Training Loss (log scale)", "Epoch", "MSE Loss")
    ax.semilogy(loss_hist, color="#58a6ff", lw=1.5)
    ax.fill_between(range(len(loss_hist)), loss_hist, alpha=0.15, color="#58a6ff")
    plt.tight_layout(pad=1.2)
    return fig

# ── Results tabs ──────────────────────────────────────────────────────────────
if st.session_state.fitted and st.session_state.data is not None:
    data       = st.session_state.data
    metrics    = compute_safety_metrics(data)
    unsafe     = safety_mask(metrics, max_speed, max_accel, max_curv)
    summary    = safety_summary(metrics, unsafe)
    sim_data    = st.session_state.sim_data if st.session_state.simulated else None
    sim_metrics = compute_safety_metrics(sim_data) if sim_data else None
    sim_unsafe  = safety_mask(sim_metrics, max_speed, max_accel, max_curv) if sim_metrics else None
    sim_summary = safety_summary(sim_metrics, sim_unsafe) if sim_metrics else None

    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "🛤️ Trajectory", "📝 Equations & Math",
        "🛡️ Safety", "📈 Training", "🔬 How It Works"
    ])

    # ── Tab 1: Trajectory ─────────────────────────────────────────────────────
    with tab1:
        fig_p = plot_trajectory(data, unsafe, sim_data, sim_unsafe)
        st.pyplot(fig_p, use_container_width=True); plt.close(fig_p)

        if sim_data is not None:
            err = compute_error_metrics(data["x"],data["y"],sim_data["x"],sim_data["y"])
            st.markdown('<div class="section-header">Simulation Error Metrics</div>', unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("RMSE x",     f"{err['rmse_x']:.1f} px")
            c2.metric("RMSE y",     f"{err['rmse_y']:.1f} px")
            c3.metric("RMSE total", f"{err['rmse_total']:.1f} px")
            c4.metric("Hausdorff",  f"{err['hausdorff']:.1f} px")
            st.markdown("""<div class="math-box">
<b style="color:#58a6ff">RMSE</b> — average positional error between time-matched points:<br>
<div class="math-formula">RMSE = sqrt( (1/N) * sum( ||p_true - p_sim||^2 ) )</div>
<b style="color:#58a6ff">Hausdorff Distance</b> — worst-case deviation anywhere:<br>
<div class="math-formula">H(P,Q) = max( max_p min_q d(p,q),  max_q min_p d(p,q) )</div>
Low RMSE means the simulation tracks the path well on average. Low Hausdorff means no single point deviates wildly.
</div>""", unsafe_allow_html=True)
        else:
            st.info("👆 Click **▶️ Simulate** above to overlay the re-flown trajectory and compare error metrics.")

    # ── Tab 2: Equations ──────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Learned Equations of Motion</div>', unsafe_allow_html=True)
        st.markdown("""<div class="math-box">
These equations describe how the drone <b>accelerates</b> as a function of its current state.
They are expressed in <b>normalised coordinates</b> (zero-mean, unit-std) so coefficients are
numerically well-conditioned and sin/cos terms are meaningful. The simulator de-normalises outputs
back to pixel units automatically.
</div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="eq-label">Equations of Motion — normalised state space</div>
<div class="eq-box">{st.session_state.eq_x}<br>{st.session_state.eq_y}</div>""",
                    unsafe_allow_html=True)

        st.markdown("""<div class="math-box">
<b style="color:#58a6ff">Feature Library Θ</b> — all candidate basis functions:<br>
<div class="math-formula">Θ = [ 1 | x | y | vx | vy | x² | y² | xy | vx² | vy² | vx·vy | sin(x) | sin(y) | cos(x) | cos(y) ]</div>
SINDy finds sparse <b>w</b> such that:
<div class="math-formula">ax_norm ≈ Θ · w_x     and     ay_norm ≈ Θ · w_y</div>
The <b>sparsity slider λ</b> controls how many terms survive. Higher λ = simpler equation.
If the simulation diverges, try raising λ to force a simpler model.
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Acceleration Predictions — Ground Truth vs GNN vs SINDy</div>',
                    unsafe_allow_html=True)
        ax_gnn = ay_gnn = ax_sym = ay_sym = None
        if st.session_state.gnn_model:
            gnn_m, x_mu, x_std, y_mu, y_std = st.session_state.gnn_model
            ax_gnn, ay_gnn = gnn_predict(gnn_m, data, x_mu, x_std, y_mu, y_std)
        if st.session_state.sym_model and st.session_state.norm_stats:
            try:
                ax_sym, ay_sym = predict_accelerations(
                    st.session_state.sym_model, data, st.session_state.norm_stats)
            except Exception: pass

        fig_acc = plot_accelerations(data, ax_gnn, ay_gnn, ax_sym, ay_sym)
        st.pyplot(fig_acc, use_container_width=True); plt.close(fig_acc)

        st.markdown("""<div class="math-box">
<b>Ground truth</b> — accelerations computed analytically by the Savitzky-Golay polynomial fit.<br>
<b>GNN</b> — predictions from the trained Graph Neural Network (high accuracy, black-box).<br>
<b>SINDy</b> — predictions from the sparse symbolic equation (interpretable, may trade some accuracy for simplicity).
</div>""", unsafe_allow_html=True)

    # ── Tab 3: Safety ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Safety Envelope Analysis</div>', unsafe_allow_html=True)
        st.markdown("""<div class="math-box">
Every time step is checked against three physical limits. A segment turns <b style="color:#f85149">red</b>
if <i>any</i> threshold is exceeded. Thresholds are set in the sidebar in physical units (px/s, px/s², 1/px).
<br><br>
<div class="math-formula">Speed        v = sqrt(vx² + vy²)             px/s   — motor saturation limit</div>
<div class="math-formula">Acceleration a = sqrt(ax² + ay²)             px/s²  — structural stress limit</div>
<div class="math-formula">Curvature    κ = |vx·ay − vy·ax| / v³       1/px   — turning radius limit (1/κ = radius)</div>
<small style="color:#8b949e">If most of your path shows as unsafe, raise the thresholds in the sidebar.</small>
</div>""", unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)

        def safety_card(label, summ):
            pct = summ["pct_unsafe"]
            cls = "safety-danger" if pct > 20 else ("safety-warn" if pct > 5 else "safety-safe")
            bar_w = min(100, pct)
            return f"""<div class="card">
<div class="card-title">{label}</div>
<div class="{cls}">{pct:.1f}% unsafe</div>
<div class="safety-bar-bg"><div class="safety-bar-fill" style="width:{bar_w}%"></div></div>
<div class="card-sub">
  Max speed: <b>{summ['max_speed']:.1f} px/s</b> (limit {max_speed:.0f})<br>
  Max accel: <b>{summ['max_accel']:.1f} px/s²</b> (limit {max_accel:.0f})<br>
  Max κ:     <b>{summ['max_curvature']:.4f} 1/px</b> (limit {max_curv:.3f})
</div></div>"""

        with col_s1:
            st.markdown(safety_card("✏️ Drawn Path", summary), unsafe_allow_html=True)
            if summary["violation_indices"]:
                viol = summary["violation_indices"]
                pos  = [f"{100*i/len(data['t']):.0f}%" for i in viol[::max(1,len(viol)//6)]]
                st.markdown(f"<small style='color:#8b949e'>Violations at: {', '.join(pos)} along path</small>",
                            unsafe_allow_html=True)

        with col_s2:
            if sim_summary:
                st.markdown(safety_card("▶️ Simulated Path", sim_summary), unsafe_allow_html=True)
                delta = sim_summary["pct_unsafe"] - summary["pct_unsafe"]
                icon  = "🔴 More unsafe" if delta > 2 else ("🟢 Safer" if delta < -2 else "🟡 Similar")
                st.markdown(f"<small style='color:#8b949e'>vs drawn path: {icon} ({delta:+.1f} pp)</small>",
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="card"><div class="card-sub">Run ▶️ Simulate to compare</div></div>',
                            unsafe_allow_html=True)

        fig_s = plot_safety_metrics(data, metrics, "Drawn Path — Safety Metrics over Time")
        st.pyplot(fig_s, use_container_width=True); plt.close(fig_s)

        if sim_data and sim_metrics:
            fig_ss = plot_safety_metrics(sim_data, sim_metrics, "Simulated Path — Safety Metrics over Time")
            st.pyplot(fig_ss, use_container_width=True); plt.close(fig_ss)

    # ── Tab 4: Training ───────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">GNN Training Curve</div>', unsafe_allow_html=True)
        if st.session_state.gnn_loss:
            fig_l = plot_loss(st.session_state.gnn_loss)
            st.pyplot(fig_l, use_container_width=True); plt.close(fig_l)
            c1,c2,c3 = st.columns(3)
            c1.metric("Final loss",  f"{st.session_state.gnn_loss[-1]:.6f}")
            c2.metric("Epochs run",  len(st.session_state.gnn_loss))
            c3.metric("Best loss",   f"{min(st.session_state.gnn_loss):.6f}")

        density = min(100, 100*(2*k_nbrs+1)/N_points)
        st.markdown(f"""<div class="math-box">
<b style="color:#58a6ff">Temporal Graph G = (V, E)</b><br><br>
Nodes |V| = {N_points} — each time step carries features [x, y, vx, vy]<br>
Edges: node i connects to j if |i−j| ≤ {k_nbrs} plus self-loop → each node sees {2*k_nbrs+1} neighbours<br>
Graph density ≈ {density:.1f}%<br><br>
<div class="math-formula">A_ij = 1 if |i−j| ≤ k else 0    then    Ã = row_normalise(A)</div>
<b>Message passing:</b>
<div class="math-formula">messages = MLP_msg(X)              # Linear → Tanh → Linear</div>
<div class="math-formula">M = Ã · messages                   # aggregate from neighbours</div>
<div class="math-formula">h = MLP_update([X ‖ M])            # Linear → Tanh → Linear → Tanh</div>
<div class="math-formula">(ax, ay) = W · h                   # output head</div>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div class="math-box"><table style="width:100%;border-collapse:collapse;font-size:0.85rem">
<tr style="border-bottom:1px solid #30363d">
  <th style="text-align:left;padding:6px;color:#58a6ff">Aspect</th>
  <th style="text-align:left;padding:6px;color:#58a6ff">Choice</th>
  <th style="text-align:left;padding:6px;color:#58a6ff">Why</th>
</tr>
<tr style="border-bottom:1px solid #21262d"><td style="padding:6px;color:#c9d1d9">Supervision</td><td style="padding:6px;color:#a5d6ff">Teacher forcing</td><td style="padding:6px;color:#8b949e">Feed ground-truth states; supervise predicted accelerations</td></tr>
<tr style="border-bottom:1px solid #21262d"><td style="padding:6px;color:#c9d1d9">Loss</td><td style="padding:6px;color:#a5d6ff">MSE on (ax, ay)</td><td style="padding:6px;color:#8b949e">Direct regression on physical quantities</td></tr>
<tr style="border-bottom:1px solid #21262d"><td style="padding:6px;color:#c9d1d9">Optimiser</td><td style="padding:6px;color:#a5d6ff">Adam</td><td style="padding:6px;color:#8b949e">Adaptive learning rate, fast convergence</td></tr>
<tr style="border-bottom:1px solid #21262d"><td style="padding:6px;color:#c9d1d9">Scheduler</td><td style="padding:6px;color:#a5d6ff">ReduceLROnPlateau</td><td style="padding:6px;color:#8b949e">Halves LR when loss plateaus</td></tr>
<tr style="border-bottom:1px solid #21262d"><td style="padding:6px;color:#c9d1d9">Early stopping</td><td style="padding:6px;color:#a5d6ff">Patience = 30</td><td style="padding:6px;color:#8b949e">Prevents overtraining on single trajectory</td></tr>
<tr><td style="padding:6px;color:#c9d1d9">Gradient clip</td><td style="padding:6px;color:#a5d6ff">Max norm = 1.0</td><td style="padding:6px;color:#8b949e">Prevents exploding gradients on sharp turns</td></tr>
</table></div>""", unsafe_allow_html=True)

    # ── Tab 5: How It Works ───────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">Full Pipeline — Mathematics & ML Explained</div>',
                    unsafe_allow_html=True)
        st.markdown("""
<div class="math-box"><div class="step-row"><span class="step-num">1</span><div>
<b style="color:#58a6ff">Arc-Length Resampling</b><br>
Raw mouse strokes are unevenly spaced. Compute cumulative arc-length, interpolate to N uniform samples:
<div class="math-formula">s_k = Σ sqrt(Δx_i² + Δy_i²)   →   interpolate x(s), y(s) at uniform s values</div>
Ensures Δt is physically meaningful and all derivatives are well-conditioned.
</div></div></div>

<div class="math-box"><div class="step-row"><span class="step-num">2</span><div>
<b style="color:#58a6ff">Savitzky-Golay Differentiation</b><br>
Fit a degree-p polynomial in a sliding window, differentiate analytically:
<div class="math-formula">x̂(t) = Σ c_k · t^k   →   vx = dx̂/dt   →   ax = d²x̂/dt²</div>
Error O(Δt^(p+1)) vs O(Δt) for finite differences — far more accurate on noisy mouse data.
</div></div></div>

<div class="math-box"><div class="step-row"><span class="step-num">3</span><div>
<b style="color:#58a6ff">Normalisation  (critical for stability)</b><br>
Before fitting SINDy, all features are normalised to zero-mean, unit-std:
<div class="math-formula">x_norm = (x − μ_x) / σ_x     for each feature</div>
This prevents sin/cos terms from wrapping at nonsensical pixel values (x ~ 200–600 px)
and keeps regression coefficients numerically well-conditioned.
The simulator normalises inputs and de-normalises outputs at every RK4 step.
</div></div></div>

<div class="math-box"><div class="step-row"><span class="step-num">4</span><div>
<b style="color:#58a6ff">Temporal Graph Construction</b><br>
Each time step becomes a node [x, y, vx, vy]. Edges connect temporal neighbours:
<div class="math-formula">A_ij = 1 if |i−j| ≤ k else 0   →   Ã = row_normalise(A)</div>
The graph gives each node local temporal context — critical at turns where future trajectory matters.
</div></div></div>

<div class="math-box"><div class="step-row"><span class="step-num">5</span><div>
<b style="color:#58a6ff">Graph Neural Network — Message Passing</b><br>
<div class="math-formula">messages = MLP_msg(X)              # transform node features</div>
<div class="math-formula">M = Ã · messages                   # aggregate from temporal neighbours</div>
<div class="math-formula">h = MLP_update([X ‖ M])            # combine own features + context</div>
<div class="math-formula">(ax, ay) = W · h                   # predict acceleration</div>
</div></div></div>

<div class="math-box"><div class="step-row"><span class="step-num">6</span><div>
<b style="color:#58a6ff">SINDy — Sequential Thresholded Least Squares (STLSQ)</b><br>
<div class="math-formula">Step 1: w = (Θᵀ Θ)⁻¹ Θᵀ a       # ordinary least squares</div>
<div class="math-formula">Step 2: zero out |w_i| &lt; λ       # sparsity threshold</div>
<div class="math-formula">Step 3: refit on active features</div>
<div class="math-formula">Step 4: repeat until convergence</div>
</div></div></div>

<div class="math-box"><div class="step-row"><span class="step-num">7</span><div>
<b style="color:#58a6ff">RK4 Forward Simulation</b><br>
ODE: dx/dt=vx, dy/dt=vy, dvx/dt=f(·), dvy/dt=g(·) — integrated with 4th-order Runge-Kutta:
<div class="math-formula">s_n+1 = s_n + (Δt/6)(k1 + 2k2 + 2k3 + k4)</div>
Local truncation error O(Δt⁵). Inputs normalised, outputs de-normalised at each step.
</div></div></div>

<div class="math-box"><div class="step-row"><span class="step-num">8</span><div>
<b style="color:#58a6ff">Safety Envelope — Differential Geometry</b><br>
<div class="math-formula">Speed:        v = sqrt(vx² + vy²)              px/s</div>
<div class="math-formula">Acceleration: a = sqrt(ax² + ay²)              px/s²</div>
<div class="math-formula">Curvature:    κ = |vx·ay − vy·ax| / v³        1/px  (1/κ = turning radius)</div>
</div></div></div>

<div class="math-box">
<b style="color:#58a6ff">References</b><br><br>
• Brunton, Proctor & Kutz — Discovering governing equations (PNAS 2016)<br>
• Kipf & Welling — Graph Convolutional Networks (ICLR 2017)<br>
• Veličković et al. — Graph Attention Networks (ICLR 2018)<br>
• Savitzky & Golay — Smoothing by least squares (Anal. Chem. 1964)<br>
• de Silva et al. — PySINDy (JOSS 2020)
</div>""", unsafe_allow_html=True)

elif st.session_state.raw_xs is not None:
    st.markdown("""<div class="card" style="border-color:#388bfd44;background:#1c2d3d">
<div style="font-size:1.05rem;color:#58a6ff;font-weight:600">✅ Path captured — ready to fit</div>
<div class="card-sub">Click <b>🔧 Fit Model</b> to train the GNN and discover equations of motion</div>
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="card">
<div style="font-size:1.05rem;color:#8b949e;font-weight:600">👆 Draw a path to get started</div>
<div class="card-sub">Click and drag on the canvas above to sketch a drone trajectory, then click Fit Model</div>
</div>""", unsafe_allow_html=True)

st.markdown("""<div style="margin-top:40px;padding:20px;border-top:1px solid #21262d;
display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
<div style="color:#8b949e;font-size:0.8rem">🚁 Drone Path · Equation Finder &nbsp;·&nbsp; GNN + SINDy + RK4</div>
<div style="color:#8b949e;font-size:0.8rem">Draw → Fit → Simulate → Analyse &nbsp;·&nbsp; No GPU required</div>
</div>""", unsafe_allow_html=True)