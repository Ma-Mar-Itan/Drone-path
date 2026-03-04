# 🚁 Drone Path Equation-of-Motion Demo

> Draw a 2-D drone trajectory with your mouse. Watch a Graph Neural Network learn its dynamics in seconds. Get back human-readable equations of motion. Re-fly the path from those equations. All on your CPU, no cloud required.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Quick Start](#quick-start)
3. [Full Pipeline Explained](#full-pipeline-explained)
4. [UI Guide](#ui-guide)
5. [Project Structure](#project-structure)
6. [Module Reference](#module-reference)
7. [Configuration & Tuning](#configuration--tuning)
8. [Requirements & Installation](#requirements--installation)
9. [Known Limitations](#known-limitations)
10. [Background & References](#background--references)

---

## What It Does

This demo bridges three fields — **geometric deep learning**, **sparse system identification**, and **robotics safety** — in a single interactive app:

| Stage | What Happens |
|-------|-------------|
| ✏️ **Draw** | Sketch a drone path on a canvas with your mouse |
| 📐 **Resample** | The stroke is converted to a uniform-time trajectory with clean derivatives |
| 🕸️ **Graph** | The trajectory becomes a temporal graph: nodes = time steps, edges = time neighbours |
| 🧠 **GNN** | A Graph Neural Network learns `state → acceleration` over that graph |
| 📝 **SINDy** | Sparse regression distils the GNN's knowledge into a short, readable equation |
| ▶️ **Simulate** | The equation is integrated forward with RK4 to re-fly the path |
| 🛡️ **Safety** | Speed, acceleration, and curvature are checked against user-defined thresholds |

---

## Quick Start

```bash
# 1. Navigate to the project folder
cd "path/to/Drone-path"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run App.py
```

Your browser opens automatically at `http://localhost:8501`.

> **Windows PowerShell tip:** wrap paths containing spaces in double quotes.

---

## Full Pipeline Explained

### Step 1 — Drawing & Data Capture

The canvas (powered by `streamlit-drawable-canvas`) records every mouse-drag point as a sequence of `(x, y)` pixel coordinates. These are raw, unevenly spaced, and potentially noisy — exactly what you would get from a real sensor log.

If `streamlit-drawable-canvas` is not installed, the app automatically uses a built-in **figure-8 trajectory** so you can still explore all features without the drawing canvas.

---

### Step 2 — Resampling & Smoothing

Raw strokes have uneven point spacing: the mouse moves fast in straight sections and slow around corners. Before any physics analysis we need uniform time steps.

**Arc-length resampling** (`pipeline.py → resample_arc_length`):
1. Compute the cumulative arc-length along the raw polyline.
2. Linearly interpolate to produce `N` points equally spaced by distance.
3. Assign synthetic time `t = 0, Δt, 2Δt, ...`

**Savitzky–Golay smoothing** (`pipeline.py → smooth_and_differentiate`):
- A polynomial is locally fitted over a sliding window and evaluated to produce smooth `x(t)` and `y(t)`.
- The same polynomial is analytically differentiated to give `vx`, `vy`, `ax`, `ay` — far more accurate than finite differences on noisy data.
- Output: a dict with keys `x, y, vx, vy, ax, ay, t`.

---

### Step 3 — The Temporal Graph

This is the "geometric" part. Instead of treating the trajectory as a plain time series, we model it as a **graph**:

```
Node i  →  feature vector [x_i, y_i, vx_i, vy_i]

Edges:  i ↔ j   for  |i − j| ≤ k   (k-nearest neighbours in time)
        i → i   (self-loop so each node includes its own features)
```

The adjacency matrix `A` is `(N × N)`, row-normalised so each row sums to 1.

**Why a graph?** A single time step does not contain enough context to infer acceleration reliably — especially at corners where future trajectory information matters. The graph lets each node "look around" in time before predicting its dynamics, analogous to how a Graph Convolutional Network on a molecular graph aggregates over a local chemical neighbourhood before predicting a bond property.

---

### Step 4 — Graph Neural Network Training

**Architecture** (`gnn_model.py → TemporalGNN`):

```
Input:        X  (N × 4)   — node features [x, y, vx, vy]
              A  (N × N)   — row-normalised adjacency matrix

Message MLP:  X  →  messages              (Linear → Tanh → Linear)
Aggregate:    M  =  A · messages(X)       ← graph convolution step
Concatenate:  [X, M]                      ← own features + context
Update MLP:   [X, M]  →  hidden repr      (Linear → Tanh → Linear → Tanh)
Output head:  hidden  →  (ax, ay)
```

**Training** (`gnn_model.py → train_gnn`):
- **Teacher forcing** — ground-truth states are used as node features at every step (no recurrent rollout during training).
- **Loss** — MSE between predicted and Savitzky–Golay-derived accelerations.
- **Optimiser** — Adam with `ReduceLROnPlateau` learning rate scheduler.
- **Early stopping** — training halts after `patience` epochs with no improvement, restoring the best checkpoint.
- Trains on a single CPU in **5–15 seconds** for typical paths.

---

### Step 5 — Symbolic Regression (SINDy)

The GNN is accurate but opaque. **SINDy** (Sparse Identification of Nonlinear Dynamics) converts it into readable equations.

**How it works:**

1. Collect `(state, acceleration)` pairs from the smoothed trajectory.
2. Build a **feature library** matrix `Θ` where each column is a candidate basis function:

```
Θ = [ 1 | x | y | vx | vy | x² | y² | xy | vx² | vy² | vx·vy | sin(x) | sin(y) | cos(x) | cos(y) ]
```

3. Solve the sparse regression problem for each acceleration component:

```
ax  ≈  Θ · w_x      (minimise ||w_x||₀ subject to fit quality)
ay  ≈  Θ · w_y
```

using **STLSQ** — Sequential Thresholded Least Squares. This iteratively zeroes out coefficients below the sparsity threshold and refits on the remaining active terms.

4. Output the surviving terms as human-readable equations:

```
d²x/dt² = +12.34 − 0.82·vx + 3.11·cos(y)
d²y/dt² = − 5.67·vy + 2.44·x − 1.09·sin(x)
```

The **sparsity slider** in the sidebar directly controls the threshold — higher means fewer terms and simpler equations.

The app uses **PySINDy** when available (more robust, battle-tested) and falls back to a hand-coded STLSQ implementation if PySINDy is not installed.

---

### Step 6 — Forward Simulation (RK4)

The symbolic equations define a complete dynamical system:

```
dx/dt   = vx
dy/dt   = vy
dvx/dt  = f(x, y, vx, vy)      ← SINDy equation for d²x/dt²
dvy/dt  = g(x, y, vx, vy)      ← SINDy equation for d²y/dt²
```

This ODE is integrated forward using **4th-order Runge–Kutta (RK4)** starting from the drawn path's initial conditions `(x₀, y₀, vx₀, vy₀)`.

Error metrics reported:

| Metric | Formula | Meaning |
|--------|---------|---------|
| RMSE x | `√ mean((x_true − x_sim)²)` | Average horizontal deviation |
| RMSE y | `√ mean((y_true − y_sim)²)` | Average vertical deviation |
| RMSE total | `√ mean(‖p_true − p_sim‖²)` | Overall positional error |
| Hausdorff | `max(d_H(true→sim), d_H(sim→true))` | Worst-case deviation anywhere |

---

### Step 7 — Safety Envelope

Real drones have hard physical limits. The safety panel checks three metrics at every time step:

| Metric | Formula | Physical Meaning |
|--------|---------|-----------------|
| **Speed** | `√(vx² + vy²)` | Motor saturation, battery drain |
| **Acceleration** | `√(ax² + ay²)` | Structural stress, propeller stall |
| **Curvature** | `|vx·ay − vy·ax| / speed³` | Minimum turning radius, aerodynamic stability |

Any time step where **at least one** metric exceeds its threshold is flagged **unsafe** and drawn in **red** on the trajectory plot.

The Safety tab reports:
- Percentage of the trajectory that is unsafe
- Peak values of each metric across the whole path
- Where along the path violations occur (as % of total path length)
- Side-by-side comparison of drawn vs simulated paths — does the learned model behave more conservatively or more aggressively?

---

## UI Guide

### Sidebar Controls

| Control | What It Does |
|---------|-------------|
| **Trajectory samples (N)** | Number of arc-length-equalised time steps. More = finer resolution, slower GNN training |
| **Time step Δt** | Seconds per sample. Scales all velocity and acceleration units |
| **Max epochs** | Upper limit on GNN training. Early stopping usually fires before this |
| **Hidden units** | Width of GNN's internal layers. Larger = more expressive but slower |
| **k-nearest neighbours** | How many time steps each node can see in each direction |
| **Sparsity threshold** | Higher = fewer terms in the symbolic equations |
| **Max speed / accel / curvature** | Safety thresholds. Segments above these turn red |

### Main Tabs

| Tab | Content |
|-----|---------|
| 🛤️ **Trajectory** | Drawn path (blue = safe, red = unsafe) with simulated overlay (dashed) and error metrics |
| 📐 **Equations** | The learned symbolic equations + acceleration prediction comparison (ground truth vs GNN vs SINDy) |
| 🛡️ **Safety** | Summary cards, metric time-series plots with threshold lines, drawn vs simulated comparison |
| 📈 **Training** | GNN loss curve, epoch count, graph structure description |

### Buttons

| Button | Action |
|--------|--------|
| 🔧 **Fit Model** | Run the full pipeline: resample → smooth → train GNN → fit SINDy |
| ▶️ **Simulate** | Integrate the symbolic equations forward from initial conditions |
| 🗑️ **Clear** | Reset everything and start fresh |
| 💾 **Export** | Download trajectory data + equations as a JSON file |

---

## Project Structure

```
Drone-path/
│
├── App.py                  # Streamlit entry point — all UI logic
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
└── src/                    # Core library
    ├── __init__.py         # Package marker
    ├── pipeline.py         # Resampling, smoothing, derivatives, safety metrics
    ├── gnn_model.py        # Temporal GNN: architecture, adjacency, training loop
    ├── symbolic.py         # PySINDy wrapper + manual STLSQ fallback
    └── simulator.py        # RK4 integrator + RMSE / Hausdorff metrics
```

---

## Module Reference

### `src/pipeline.py`

| Function | Description |
|----------|-------------|
| `resample_arc_length(xs, ys, N)` | Arc-length resampling to N uniform points |
| `smooth_and_differentiate(xs, ys, dt, window_len, poly_order)` | SG filter → smooth coords + 1st and 2nd derivatives |
| `compute_safety_metrics(data)` | Returns `speed`, `accel_mag`, `curvature` arrays |
| `safety_mask(metrics, max_speed, max_accel, max_curvature)` | Boolean mask: True where any threshold is exceeded |
| `safety_summary(metrics, unsafe)` | Dict with `pct_unsafe`, peak values, violation indices |

### `src/gnn_model.py`

| Function / Class | Description |
|-----------------|-------------|
| `TemporalGNN(in_dim, hidden, out_dim)` | PyTorch GNN module (pure PyTorch, no PyG required) |
| `build_adjacency(N, k_neighbors)` | Builds row-normalised (N×N) adjacency matrix |
| `train_gnn(data, epochs, lr, patience, hidden, k_neighbors)` | Full training loop with early stopping |
| `gnn_predict(model, data, x_mu, x_std, y_mu, y_std)` | Inference returning `(ax_pred, ay_pred)` |

### `src/symbolic.py`

| Function / Class | Description |
|-----------------|-------------|
| `build_feature_matrix(x, y, vx, vy)` | Constructs the Θ feature library (N × 15) |
| `fit_sindy(data, threshold)` | PySINDy-based sparse regression |
| `fit_sparse_manual(data, threshold)` | Pure-numpy STLSQ fallback |
| `fit_equations(data, threshold)` | Unified entry point — tries PySINDy, falls back automatically |
| `predict_accelerations(model, data)` | Runs prediction via whichever backend was used |

### `src/simulator.py`

| Function | Description |
|----------|-------------|
| `rk4_step(state, accel_fn, dt)` | Single RK4 integration step |
| `simulate(model, x0, y0, vx0, vy0, dt, N)` | Full forward simulation returning trajectory dict |
| `compute_error_metrics(true_x, true_y, pred_x, pred_y)` | RMSE (x, y, total) + Hausdorff distance |

---

## Configuration & Tuning

**Getting better equations:**
- Use **smooth, flowing paths** — sharp corners create large derivative noise that confuses SINDy
- **Lower the sparsity threshold** (e.g. 0.01) to retain more terms if the simulation diverges
- **Raise the sparsity threshold** (e.g. 0.2–0.5) to get a clean 2–3 term equation
- Increase **N** (samples) and **epochs** for longer or more complex paths

**Getting better simulations:**
- If the simulated path diverges or flies off-screen, the symbolic model is overfitting — increase sparsity threshold
- Increase **k-nearest neighbours** to give the GNN more temporal context at each node
- Reduce **Δt** for smoother RK4 integration (at the cost of more steps)

**Safety tuning:**
- All units are in **pixels per second** — adjust thresholds to match your canvas size and drawing speed
- A path drawn slowly on a 600 px canvas will have much lower speed values than one drawn quickly
- Curvature is the most sensitive metric — start with a threshold around 0.02–0.05 and adjust from there

---

## Requirements & Installation

```
Python                      3.10+
streamlit                   1.32+
torch                       2.1+      (CPU build is fine, no GPU needed)
pysindy                     1.7+      (optional, falls back to manual STLSQ)
scipy                       1.11+
numpy                       1.24+
matplotlib                  3.7+
scikit-learn                1.3+
streamlit-drawable-canvas   0.9.3+    (optional, falls back to sample path)
```

Install all at once:

```bash
pip install -r requirements.txt
```

No GPU is required. The GNN trains entirely on CPU in under 15 seconds for typical drawn paths.

---

## Known Limitations

- **Single trajectory** — the model is fitted to exactly one drawn path and does not generalise to new paths.
- **Pixel units** — all physics quantities are in screen pixels, not real-world metres. The equations are dimensionally self-consistent but not physically calibrated.
- **Divergent simulations** — if the learned equations have unstable fixed points, RK4 integration can diverge. A safety clip prevents crashes but the simulated path may look unrealistic. Fix: increase the sparsity threshold.
- **Short paths** — paths with fewer than ~20 points may not have enough data to fit meaningful equations.
- **Rapidly reversing paths** — U-turns create large curvature and acceleration spikes that challenge the Savitzky–Golay smoother. Try increasing the smoothing window length.
- **Not for production** — this is an educational demo. Do not use it to plan real drone flights.

---

## Background & References

| Topic | Reference |
|-------|-----------|
| SINDy — Sparse Identification of Nonlinear Dynamics | Brunton, Proctor & Kutz (2016), *PNAS* 113(15) |
| Graph Convolutional Networks | Kipf & Welling (2017), *ICLR* |
| Graph Attention Networks | Veličković et al. (2018), *ICLR* |
| Savitzky–Golay filter | Savitzky & Golay (1964), *Analytical Chemistry* 36(8) |
| PySINDy library | de Silva et al. (2020), *Journal of Open Source Software* |
| Runge–Kutta integration | Burden & Faires, *Numerical Analysis* (any edition) |

---

*Built as a self-contained educational demo. Not intended for real flight systems.*
