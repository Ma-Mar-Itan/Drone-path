"""
app.py
------
Drone Path Equation-of-Motion Demo
===================================

Draw a 2-D drone path with your mouse.  The app:
  1. Resamples and smooths the path
  2. Trains a small GNN over the temporal graph
  3. Fits sparse symbolic equations via SINDy
  4. Simulates the learned dynamics forward
  5. Highlights unsafe segments (speed / accel / curvature)

Run:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import json
import time

# ── local modules ──────────────────────────────────────────────────────────
from src.pipeline import (
    resample_arc_length,
    smooth_and_differentiate,
    compute_safety_metrics,
    safety_mask,
    safety_summary,
)
from src.gnn_model import train_gnn, gnn_predict
from src.symbolic import fit_equations, predict_accelerations
from src.simulator import simulate, compute_error_metrics

# ── Streamlit page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Drone Path Dynamics",
    page_icon="🚁",
    layout="wide",
)

# ── Session state initialisation ──────────────────────────────────────────
DEFAULTS = dict(
    raw_xs=None, raw_ys=None,
    data=None,
    gnn_model=None, gnn_loss=[],
    sym_model=None, eq_x="", eq_y="",
    sim_data=None,
    fitted=False,
    simulated=False,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════
# Sidebar – controls & safety panel
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🚁 Drone Path Demo")
    st.markdown("---")

    st.subheader("📐 Resampling")
    N_points = st.slider("Trajectory samples (N)", 40, 200, 80, 10,
                         help="Number of arc-length-equalised samples")
    dt = st.slider("Time step Δt (s)", 0.01, 0.20, 0.05, 0.01,
                   help="Seconds per sample (scales velocity/accel units)")

    st.subheader("🧠 GNN Training")
    epochs = st.slider("Max epochs", 100, 2000, 600, 100)
    hidden = st.slider("Hidden units", 16, 128, 64, 16)
    k_nbrs = st.slider("k-nearest-time neighbours", 1, 5, 3)

    st.subheader("📐 Symbolic Regression")
    sparsity = st.slider("Sparsity threshold", 0.001, 0.5, 0.05, 0.001,
                         format="%.3f",
                         help="Higher → fewer terms in equation")

    st.markdown("---")
    st.subheader("🛡️ Safety Envelope")
    max_speed = st.slider("Max speed", 1.0, 50.0, 15.0, 0.5)
    max_accel = st.slider("Max acceleration", 1.0, 100.0, 30.0, 1.0)
    max_curv  = st.slider("Max curvature", 0.001, 0.5, 0.05, 0.001,
                           format="%.3f")

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown(
        """
        **Graph** = nodes are time-steps; edges connect temporal neighbours.
        
        **GNN** learns acceleration from local spatio-temporal context.
        
        **SINDy** fits sparse polynomial/trig equations to the GNN-identified dynamics.
        
        **Safe envelope** flags where speed, acceleration, or path curvature exceeds thresholds.
        """
    )


# ══════════════════════════════════════════════════════════════════════════
# Main layout
# ══════════════════════════════════════════════════════════════════════════

st.title("✈️ Drone Path Equation-of-Motion Finder")
st.markdown(
    "Draw a drone path below, then hit **Fit Model** to learn its dynamics and "
    "**Simulate** to re-fly it from the learned equations."
)

# ── Drawing canvas ─────────────────────────────────────────────────────────
col_canvas, col_info = st.columns([2, 1])

with col_canvas:
    st.subheader("🖊️ Draw your path")
    st.markdown("Click and drag to draw a continuous drone trajectory.")

    try:
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            stroke_color="#1E88E5",
            background_color="#F8F9FA",
            height=420,
            width=620,
            drawing_mode="freedraw",
            key="canvas",
        )

        # Extract strokes from canvas JSON
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            all_xs, all_ys = [], []
            for obj in objects:
                if obj.get("type") == "path":
                    path_data = obj.get("path", [])
                    for cmd in path_data:
                        if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M", "L", "Q"):
                            all_xs.append(float(cmd[1]))
                            all_ys.append(float(cmd[2]))

            if len(all_xs) > 5:
                st.session_state.raw_xs = np.array(all_xs)
                st.session_state.raw_ys = np.array(all_ys)

        canvas_available = True

    except ImportError:
        canvas_available = False
        st.warning(
            "⚠️ `streamlit-drawable-canvas` not installed. "
            "Using a sample trajectory instead.\n\n"
            "Install with: `pip install streamlit-drawable-canvas`"
        )
        # Provide a sample figure-8 trajectory
        t_sample = np.linspace(0, 2 * np.pi, 300)
        all_xs_s = 200 + 150 * np.sin(t_sample)
        all_ys_s = 200 + 80  * np.sin(2 * t_sample)
        st.session_state.raw_xs = all_xs_s
        st.session_state.raw_ys = all_ys_s
        st.info("Using built-in figure-8 sample path.")


with col_info:
    st.subheader("📊 Path Status")
    if st.session_state.raw_xs is not None:
        n_raw = len(st.session_state.raw_xs)
        st.metric("Raw points", n_raw)
        x_range = st.session_state.raw_xs.max() - st.session_state.raw_xs.min()
        y_range = st.session_state.raw_ys.max() - st.session_state.raw_ys.min()
        st.metric("X range (px)", f"{x_range:.0f}")
        st.metric("Y range (px)", f"{y_range:.0f}")
    else:
        st.info("No path drawn yet.")

    st.markdown("---")
    # Action buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        fit_btn = st.button("🔧 Fit Model", type="primary", use_container_width=True)
    with btn_col2:
        sim_btn = st.button("▶️ Simulate", use_container_width=True)

    clear_btn = st.button("🗑️ Clear", use_container_width=True)
    export_btn = st.button("💾 Export Data", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# Clear
# ══════════════════════════════════════════════════════════════════════════

if clear_btn:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Fit Model
# ══════════════════════════════════════════════════════════════════════════

if fit_btn:
    raw_xs = st.session_state.raw_xs
    raw_ys = st.session_state.raw_ys

    if raw_xs is None or len(raw_xs) < 10:
        st.error("Please draw a path with at least 10 points first.")
    else:
        with st.spinner("📐 Resampling and smoothing…"):
            try:
                xs_r, ys_r = resample_arc_length(raw_xs, raw_ys, N=N_points)
                data = smooth_and_differentiate(xs_r, ys_r, dt=dt)
                st.session_state.data = data
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        with st.spinner(f"🧠 Training GNN ({epochs} max epochs)…"):
            try:
                t0 = time.time()
                gnn_model, loss_hist, x_mu, x_std, y_mu, y_std = train_gnn(
                    data, epochs=epochs, hidden=hidden, k_neighbors=k_nbrs
                )
                elapsed = time.time() - t0
                st.session_state.gnn_model = (gnn_model, x_mu, x_std, y_mu, y_std)
                st.session_state.gnn_loss = loss_hist
                st.success(f"✅ GNN trained in {elapsed:.1f}s ({len(loss_hist)} epochs), "
                           f"final loss: {loss_hist[-1]:.5f}")
            except Exception as e:
                st.error(f"GNN training error: {e}")
                st.stop()

        with st.spinner("📐 Fitting symbolic equations (SINDy)…"):
            try:
                sym_model, eq_x, eq_y, used_sindy = fit_equations(data, threshold=sparsity)
                st.session_state.sym_model = sym_model
                st.session_state.eq_x = eq_x
                st.session_state.eq_y = eq_y
                backend = "PySINDy" if used_sindy else "Manual STLSQ"
                st.success(f"✅ Symbolic regression done ({backend})")
            except Exception as e:
                st.error(f"Symbolic regression error: {e}")
                st.stop()

        st.session_state.fitted = True
        st.session_state.simulated = False


# ══════════════════════════════════════════════════════════════════════════
# Simulate
# ══════════════════════════════════════════════════════════════════════════

if sim_btn:
    if not st.session_state.fitted:
        st.warning("Fit the model first.")
    else:
        data = st.session_state.data
        sym_model = st.session_state.sym_model

        with st.spinner("▶️ Simulating trajectory…"):
            try:
                x0, y0 = float(data["x"][0]), float(data["y"][0])
                vx0, vy0 = float(data["vx"][0]), float(data["vy"][0])
                N = len(data["x"])
                dt_val = float(data["t"][1] - data["t"][0])
                sim_data = simulate(sym_model, x0, y0, vx0, vy0, dt=dt_val, N=N)
                st.session_state.sim_data = sim_data
                st.session_state.simulated = True
                st.success("✅ Simulation complete")
            except Exception as e:
                st.error(f"Simulation error: {e}")


# ══════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════

if export_btn:
    if st.session_state.data is None:
        st.warning("No data to export yet.")
    else:
        data = st.session_state.data
        export = {
            "trajectory": {
                "x": data["x"].tolist(),
                "y": data["y"].tolist(),
                "vx": data["vx"].tolist(),
                "vy": data["vy"].tolist(),
                "ax": data["ax"].tolist(),
                "ay": data["ay"].tolist(),
                "t": data["t"].tolist(),
            },
            "equations": {
                "ax": st.session_state.eq_x,
                "ay": st.session_state.eq_y,
            },
        }
        json_str = json.dumps(export, indent=2)
        st.download_button(
            "⬇️ Download JSON",
            data=json_str,
            file_name="drone_path.json",
            mime="application/json",
        )


# ══════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════

def plot_path_with_safety(
    data,
    unsafe,
    sim_data=None,
    sim_unsafe=None,
    title="Trajectory",
) -> plt.Figure:
    """Render the trajectory with unsafe segments highlighted in red."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#F0F4F8")
    fig.patch.set_facecolor("#FFFFFF")

    x, y = data["x"], data["y"]
    N = len(x)

    # Draw resampled path segment-by-segment
    for i in range(N - 1):
        color = "#E53935" if unsafe[i] else "#1E88E5"
        lw = 2.5 if unsafe[i] else 1.8
        ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=lw, zorder=3)

    # Start / end markers
    ax.scatter([x[0]], [y[0]], c="green", s=100, zorder=5, label="Start")
    ax.scatter([x[-1]], [y[-1]], c="purple", s=100, marker="*", zorder=5, label="End")

    # Simulated path overlay
    if sim_data is not None:
        sx, sy = sim_data["x"], sim_data["y"]
        M = len(sx)
        if sim_unsafe is not None:
            for i in range(min(M - 1, len(sim_unsafe) - 1)):
                color = "#FF6F00" if sim_unsafe[i] else "#43A047"
                ax.plot(sx[i:i+2], sy[i:i+2], color=color, linewidth=1.5,
                        linestyle="--", zorder=4)
        else:
            ax.plot(sx, sy, "--", color="#43A047", linewidth=1.5, zorder=4,
                    label="Simulated")

    # Legend patches
    patches = [
        mpatches.Patch(color="#1E88E5", label="Drawn (safe)"),
        mpatches.Patch(color="#E53935", label="Drawn (unsafe)"),
    ]
    if sim_data is not None:
        patches += [
            mpatches.Patch(color="#43A047", label="Simulated (safe)"),
            mpatches.Patch(color="#FF6F00", label="Simulated (unsafe)"),
        ]
    patches += [
        mpatches.Patch(color="green", label="Start"),
        mpatches.Patch(color="purple", label="End"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    # Time index colouring strip (small dots)
    cmap = plt.cm.viridis
    for i in range(0, N, max(1, N // 20)):
        ax.scatter(x[i], y[i], c=[i / N], cmap=cmap, vmin=0, vmax=1,
                   s=20, zorder=6, alpha=0.6)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def plot_metrics(data, metrics, title="Safety Metrics") -> plt.Figure:
    """Plot speed, accel, curvature over time with threshold lines."""
    t = data["t"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    fig.patch.set_facecolor("#FFFFFF")

    series = [
        (metrics["speed"],     "Speed",        "#1565C0", max_speed),
        (metrics["accel_mag"], "Accel (mag)",   "#6A1B9A", max_accel),
        (metrics["curvature"], "Curvature",     "#2E7D32", max_curv),
    ]
    for ax, (vals, label, color, thresh) in zip(axes, series):
        ax.plot(t, vals, color=color, linewidth=1.5)
        ax.axhline(thresh, color="red", linestyle="--", linewidth=1, label=f"limit={thresh:.3g}")
        ax.fill_between(t, vals, thresh, where=(vals > thresh),
                        color="red", alpha=0.25)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F9F9F9")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_loss(loss_hist) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.semilogy(loss_hist, color="#1E88E5", linewidth=1.5)
    ax.set_title("GNN Training Loss", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Main display area (only shown after fitting)
# ══════════════════════════════════════════════════════════════════════════

if st.session_state.fitted and st.session_state.data is not None:
    data = st.session_state.data

    # Safety metrics for drawn path
    metrics = compute_safety_metrics(data)
    unsafe = safety_mask(metrics, max_speed, max_accel, max_curv)
    summary = safety_summary(metrics, unsafe)

    # Safety metrics for simulated path (if available)
    sim_data = st.session_state.sim_data if st.session_state.simulated else None
    sim_metrics = compute_safety_metrics(sim_data) if sim_data else None
    sim_unsafe = safety_mask(sim_metrics, max_speed, max_accel, max_curv) if sim_metrics else None
    sim_summary = safety_summary(sim_metrics, sim_unsafe) if sim_metrics else None

    # ── Tab layout ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🛤️ Trajectory", "📐 Equations", "🛡️ Safety", "📈 Training"]
    )

    # ── Tab 1: Trajectory ──────────────────────────────────────────────────
    with tab1:
        fig_path = plot_path_with_safety(
            data, unsafe,
            sim_data=sim_data,
            sim_unsafe=sim_unsafe,
            title="Drawn Path (blue=safe, red=unsafe) + Simulated (dashed)",
        )
        st.pyplot(fig_path, use_container_width=True)
        plt.close(fig_path)

        if sim_data is not None:
            err = compute_error_metrics(
                data["x"], data["y"], sim_data["x"], sim_data["y"]
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE (x)", f"{err['rmse_x']:.2f}")
            c2.metric("RMSE (y)", f"{err['rmse_y']:.2f}")
            c3.metric("RMSE (total)", f"{err['rmse_total']:.2f}")
            c4.metric("Hausdorff", f"{err['hausdorff']:.2f}")

    # ── Tab 2: Equations ───────────────────────────────────────────────────
    with tab2:
        st.subheader("Learned Equations of Motion")
        st.markdown("These equations approximate the **acceleration dynamics** of the drawn path:")

        eq_box_style = (
            "background:#1E1E2E;color:#CDD6F4;padding:16px;"
            "border-radius:8px;font-family:monospace;font-size:15px;"
        )
        st.markdown(
            f'<div style="{eq_box_style}">'
            f"{st.session_state.eq_x}<br><br>"
            f"{st.session_state.eq_y}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("GNN vs SINDy Acceleration Predictions")

        # Plot GNN predictions vs ground truth vs SINDy
        gnn_pack = st.session_state.gnn_model
        if gnn_pack is not None:
            gnn_model, x_mu, x_std, y_mu, y_std = gnn_pack
            ax_gnn, ay_gnn = gnn_predict(gnn_model, data, x_mu, x_std, y_mu, y_std)
        else:
            ax_gnn = ay_gnn = None

        sym_model = st.session_state.sym_model
        try:
            ax_sym, ay_sym = predict_accelerations(sym_model, data)
        except Exception:
            ax_sym = ay_sym = None

        t = data["t"]

        fig_acc, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        fig_acc.patch.set_facecolor("#FFFFFF")

        for ax_plot, true_vals, gnn_vals, sym_vals, lbl in [
            (axs[0], data["ax"], ax_gnn, ax_sym, "ax (d²x/dt²)"),
            (axs[1], data["ay"], ay_gnn, ay_sym, "ay (d²y/dt²)"),
        ]:
            ax_plot.plot(t, true_vals, color="#333", linewidth=1.5, label="Ground truth")
            if gnn_vals is not None:
                ax_plot.plot(t, gnn_vals, "--", color="#1E88E5", linewidth=1.2, label="GNN")
            if sym_vals is not None:
                ax_plot.plot(t, sym_vals, ":", color="#E53935", linewidth=1.5, label="SINDy")
            ax_plot.set_ylabel(lbl, fontsize=9)
            ax_plot.legend(fontsize=8)
            ax_plot.grid(True, alpha=0.3)
            ax_plot.set_facecolor("#F9F9F9")

        axs[-1].set_xlabel("time (s)")
        fig_acc.suptitle("Acceleration Predictions: Ground Truth vs GNN vs SINDy",
                         fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_acc, use_container_width=True)
        plt.close(fig_acc)

    # ── Tab 3: Safety ──────────────────────────────────────────────────────
    with tab3:
        st.subheader("🛡️ Safety Envelope Analysis")

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("#### Drawn Path")
            pct = summary["pct_unsafe"]
            color_str = "red" if pct > 20 else ("orange" if pct > 5 else "green")
            st.markdown(
                f'<p style="font-size:28px;font-weight:bold;color:{color_str}">'
                f'{pct:.1f}% unsafe</p>',
                unsafe_allow_html=True,
            )
            st.metric("Max speed",       f"{summary['max_speed']:.2f}")
            st.metric("Max acceleration", f"{summary['max_accel']:.2f}")
            st.metric("Max curvature",    f"{summary['max_curvature']:.4f}")
            if summary["violation_indices"]:
                viol_pct = [
                    f"{100*i/len(data['t']):.0f}%" 
                    for i in summary["violation_indices"][::max(1, len(summary["violation_indices"])//5)]
                ]
                st.markdown(f"**Violations at:** {', '.join(viol_pct)} of path")

        with col_s2:
            if sim_summary:
                st.markdown("#### Simulated Path")
                pct_sim = sim_summary["pct_unsafe"]
                color_sim = "red" if pct_sim > 20 else ("orange" if pct_sim > 5 else "green")
                st.markdown(
                    f'<p style="font-size:28px;font-weight:bold;color:{color_sim}">'
                    f'{pct_sim:.1f}% unsafe</p>',
                    unsafe_allow_html=True,
                )
                st.metric("Max speed",        f"{sim_summary['max_speed']:.2f}")
                st.metric("Max acceleration",  f"{sim_summary['max_accel']:.2f}")
                st.metric("Max curvature",     f"{sim_summary['max_curvature']:.4f}")
            else:
                st.info("Run simulation to compare safety metrics.")

        st.markdown("---")

        # Metric plots
        fig_m = plot_metrics(data, metrics, "Drawn Path Safety Metrics")
        st.pyplot(fig_m, use_container_width=True)
        plt.close(fig_m)

        if sim_data and sim_metrics:
            fig_ms = plot_metrics(sim_data, sim_metrics, "Simulated Path Safety Metrics")
            st.pyplot(fig_ms, use_container_width=True)
            plt.close(fig_ms)

    # ── Tab 4: Training curve ──────────────────────────────────────────────
    with tab4:
        st.subheader("GNN Training Progress")
        if st.session_state.gnn_loss:
            fig_l = plot_loss(st.session_state.gnn_loss)
            st.pyplot(fig_l, use_container_width=True)
            plt.close(fig_l)
            final_loss = st.session_state.gnn_loss[-1]
            n_ep = len(st.session_state.gnn_loss)
            st.metric("Final MSE loss", f"{final_loss:.6f}")
            st.metric("Epochs run", n_ep)

        st.markdown("---")
        st.subheader("Graph Structure")
        st.markdown(
            f"""
            **Temporal graph:**
            - **Nodes:** {N_points} time steps, each with features `[x, y, vx, vy]`
            - **Edges:** each node connects to its ±{k_nbrs} nearest neighbours in time + self-loop
            - **Message passing:** source features → MLP → messages, then sum-aggregate → update MLP → acceleration prediction
            - **Graph density:** ~{min(100, 100*(2*k_nbrs+1)/N_points):.1f}% (each node sees {2*k_nbrs+1} neighbours)
            """
        )

elif st.session_state.raw_xs is not None:
    st.info("✅ Path captured. Click **Fit Model** to learn the dynamics.")
else:
    st.info("👆 Draw a path on the canvas above, then click **Fit Model**.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:gray'>Drone Path Demo · GNN + SINDy · "
    "Draw → Fit → Simulate · No GPU required</small>",
    unsafe_allow_html=True,
)