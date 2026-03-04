<div align="center">

# 🚁 Drone Path — Equation of Motion Finder

**Draw a drone trajectory. Learn its physics. Get human-readable equations.**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drone-path-7kpgwzdwoih9vayensxuom.streamlit.app/)

*A live demo combining Graph Neural Networks, Sparse Symbolic Regression, and Robotics Safety Analysis — all running in your browser, no installation required.*

---

[🚀 Launch the App](https://drone-path-7kpgwzdwoih9vayensxuom.streamlit.app/) • [What It Does](#-what-it-does) • [The Math](#-the-mathematics) • [The ML](#-the-machine-learning) • [Safety System](#-safety-envelope) • [Run Locally](#-run-locally)

</div>

---

## 🎯 What It Does

You draw a 2-D flight path with your mouse. In seconds, the app:

1. **Learns the physics** of your path using a Graph Neural Network
2. **Discovers the equations of motion** — short, human-readable formulas like:
   ```
   d²x/dt² = −0.82·vx + 3.11·cos(y)
   d²y/dt² = −5.67·vy + 2.44·x
   ```
3. **Re-flies the drone** by integrating those equations forward in time
4. **Flags unsafe segments** where speed, acceleration, or curvature exceed safety limits

No data. No training set. No cloud GPU. Just your drawing and mathematics.

---

## 🔢 The Mathematics

### 1 — Arc-Length Resampling

Raw mouse strokes are unevenly spaced — the mouse moves fast on straight segments and slow at corners. We need uniform time steps for any physical analysis.

Given raw points $(x_i, y_i)$, the cumulative arc-length is:

$$s_k = \sum_{i=1}^{k} \sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}$$

We then interpolate to find $N$ points equally spaced along $s$, producing a trajectory parameterised by uniform time $t = 0, \Delta t, 2\Delta t, \ldots$

---

### 2 — Savitzky–Golay Differentiation

To compute velocity and acceleration we need derivatives of noisy, discrete data. Naive finite differences amplify noise. Instead we use the **Savitzky–Golay filter** (1964):

Within a sliding window of width $w$, fit a polynomial of degree $p$ to the data:

$$\hat{x}(t) = \sum_{k=0}^{p} c_k \, t^k$$

The fitted polynomial is analytically differentiated to yield smooth, accurate derivatives:

$$v_x = \frac{d\hat{x}}{dt}, \qquad a_x = \frac{d^2\hat{x}}{dt^2}$$

This gives us a clean dataset of states and accelerations: $(x, y, v_x, v_y, a_x, a_y)$ at every time step.

---

### 3 — The Temporal Graph

The trajectory is modelled as a **directed graph** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

- **Nodes** $\mathcal{V}$ — each time step $t_i$ is a node with feature vector $\mathbf{h}_i = [x_i,\; y_i,\; v_{x,i},\; v_{y,i}]$
- **Edges** $\mathcal{E}$ — node $i$ connects to node $j$ if $|i - j| \leq k$ (temporal neighbourhood of radius $k$, plus a self-loop)

The adjacency matrix $A \in \mathbb{R}^{N \times N}$ is row-normalised:

$$\tilde{A}_{ij} = \frac{A_{ij}}{\sum_j A_{ij}}$$

**Why a graph?** A single snapshot $(x, y, v_x, v_y)$ at time $t$ is not enough to infer acceleration accurately — especially at a turn, where you need to "see" nearby time steps to understand the curvature. The graph structure gives each node local spatio-temporal context.

---

### 4 — Sparse Identification of Nonlinear Dynamics (SINDy)

After learning dynamics with the GNN, we want a *human-readable* equation. Enter **SINDy** (Brunton, Proctor & Kutz, 2016).

We build a **feature library** $\Theta \in \mathbb{R}^{N \times F}$ of candidate basis functions evaluated at every time step:

$$\Theta = \begin{bmatrix} 1 & x & y & v_x & v_y & x^2 & y^2 & xy & v_x^2 & v_y^2 & v_x v_y & \sin x & \sin y & \cos x & \cos y \end{bmatrix}$$

We then solve two **sparse regression** problems:

$$a_x \approx \Theta\,\mathbf{w}_x, \qquad a_y \approx \Theta\,\mathbf{w}_y$$

subject to $\mathbf{w}$ being **sparse** — most entries should be zero, keeping only the terms that truly matter.

**STLSQ — Sequential Thresholded Least Squares:**

```
1. Fit w = (ΘᵀΘ)⁻¹Θᵀa  (ordinary least squares)
2. Zero out all |w_i| < λ  (threshold λ = sparsity slider)
3. Refit on remaining active features only
4. Repeat until convergence
```

The surviving non-zero coefficients form your equation of motion. The **sparsity slider** in the UI controls $\lambda$ — higher means fewer, simpler terms.

---

### 5 — RK4 Forward Simulation

The symbolic equations define a complete ODE system:

$$\frac{d}{dt}\begin{pmatrix} x \\ y \\ v_x \\ v_y \end{pmatrix} = \begin{pmatrix} v_x \\ v_y \\ f(x, y, v_x, v_y) \\ g(x, y, v_x, v_y) \end{pmatrix}$$

where $f$ and $g$ are the SINDy equations. We integrate this forward using **4th-order Runge–Kutta**:

$$\mathbf{s}_{n+1} = \mathbf{s}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)$$

$$\mathbf{k}_1 = F(\mathbf{s}_n), \quad \mathbf{k}_2 = F\!\left(\mathbf{s}_n + \tfrac{\Delta t}{2}\mathbf{k}_1\right), \quad \mathbf{k}_3 = F\!\left(\mathbf{s}_n + \tfrac{\Delta t}{2}\mathbf{k}_2\right), \quad \mathbf{k}_4 = F(\mathbf{s}_n + \Delta t\,\mathbf{k}_3)$$

RK4 has local truncation error $O(\Delta t^5)$, making it far more accurate than Euler integration for the same step size.

---

## 🧠 The Machine Learning

### Graph Neural Network Architecture

The GNN is implemented in **pure PyTorch** (no external graph libraries required). It follows the message-passing paradigm:

```
┌─────────────────────────────────────────────────────────┐
│                   TemporalGNN                           │
│                                                         │
│  Input X  (N × 4)   ─── [x, y, vx, vy] per node       │
│  Adjacency A  (N × N)  ─── temporal neighbourhood      │
│                                                         │
│  ① Message MLP:   X  ──► messages  (4 → 64 → 64)       │
│     Linear → Tanh → Linear                              │
│                                                         │
│  ② Aggregate:     M = Ã · messages(X)                  │
│     Each node receives weighted sum of neighbour msgs   │
│                                                         │
│  ③ Concatenate:   [X ‖ M]   (4 + 64 = 68 dims)         │
│                                                         │
│  ④ Update MLP:    [X‖M] ──► h  (68 → 64 → 64)          │
│     Linear → Tanh → Linear → Tanh                      │
│                                                         │
│  ⑤ Output head:   h ──► (ax, ay)  (64 → 2)             │
└─────────────────────────────────────────────────────────┘
```

### Training Strategy

| Aspect | Choice | Why |
|--------|--------|-----|
| **Supervision** | Teacher forcing | Feed ground-truth states; supervise predicted accelerations |
| **Loss** | MSE on $(a_x, a_y)$ | Direct regression on physical quantities |
| **Optimiser** | Adam | Adaptive learning rate, fast convergence |
| **Scheduler** | ReduceLROnPlateau | Halves LR when loss plateaus |
| **Early stopping** | Patience = 30 epochs | Prevents overtraining on a single trajectory |
| **Gradient clipping** | Max norm = 1.0 | Prevents exploding gradients on sharp turns |
| **Hardware** | CPU only | Trains in 5–15 seconds on any laptop |

### Why a GNN and Not Just an MLP?

An MLP predicting acceleration from a single state vector $[x, y, v_x, v_y]$ sees only the instantaneous snapshot. It cannot distinguish between:
- A point at the *beginning* of a sharp turn (where acceleration is about to spike)
- A point at the *middle* of that same turn (where acceleration is at maximum)

The GNN aggregates features from neighbouring time steps, giving it the context to resolve this ambiguity — exactly as a Graph Convolutional Network uses local molecular structure to predict bond energies.

### GNN → SINDy Pipeline

The GNN is a black box. SINDy converts it into glass:

```
Drawn path
    │
    ▼
Smooth derivatives  ──►  (state, accel) pairs
    │                           │
    │                           ▼
    │                    Build Θ feature library
    │                           │
    ▼                           ▼
Train GNN  ──────►  Validate GNN predictions
                           │
                           ▼
                    STLSQ sparse regression
                           │
                           ▼
                    d²x/dt² = ...  ✓ Human-readable
                    d²y/dt² = ...  ✓ Simulatable
                    
```

---

## 🛡️ Safety Envelope

Real drones have hard physical constraints. The app checks three at every time step:

### Speed
$$v = \sqrt{v_x^2 + v_y^2}$$
Exceeding max speed risks motor saturation and battery overload.

### Acceleration Magnitude
$$a = \sqrt{a_x^2 + a_y^2}$$
High acceleration causes structural stress and propeller stall.

### Path Curvature
$$\kappa = \frac{|v_x \, a_y - v_y \, a_x|}{(v_x^2 + v_y^2)^{3/2}}$$

This is the **signed curvature** formula from differential geometry. High curvature means a tight turn — which demands both high lateral acceleration and risks aerodynamic instability. $1/\kappa$ gives the instantaneous turning radius.

A time step is flagged **unsafe** (shown in red) if *any* of the three metrics exceeds its threshold. The Safety tab reports:
- Percentage of the total trajectory that is unsafe
- Peak values of speed, acceleration, and curvature
- Side-by-side comparison: is the re-simulated path safer or more aggressive than what you drew?

---

## 📊 Error Metrics

After simulation, the app computes:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **RMSE x** | $\sqrt{\frac{1}{N}\sum(x_i - \hat{x}_i)^2}$ | Average horizontal error |
| **RMSE y** | $\sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$ | Average vertical error |
| **RMSE total** | $\sqrt{\frac{1}{N}\sum\|\mathbf{p}_i - \hat{\mathbf{p}}_i\|^2}$ | Overall positional error |
| **Hausdorff** | $\max\bigl(d_H(P,Q),\, d_H(Q,P)\bigr)$ | Worst-case deviation anywhere on path |

The **Hausdorff distance** is the most conservative metric — it finds the single point where the two paths disagree the most, making it ideal for safety-critical path comparison.

---

## 🖥️ Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/drone-path.git
cd drone-path

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run App.py
```

**Requirements:** Python 3.10+, runs entirely on CPU.

---

## 📁 Project Structure

```
drone-path/
│
├── App.py                  # Streamlit UI — drawing, controls, plots
├── requirements.txt
├── README.md
│
└── src/
    ├── pipeline.py         # Arc-length resampling, SG smoothing, safety metrics
    ├── gnn_model.py        # Temporal GNN (pure PyTorch)
    ├── symbolic.py         # SINDy / STLSQ sparse regression
    └── simulator.py        # RK4 integrator, RMSE, Hausdorff
```

---

## 📚 References

| Paper | Authors | Venue |
|-------|---------|-------|
| Discovering governing equations from data by sparse identification of nonlinear dynamics | Brunton, Proctor & Kutz | *PNAS* 2016 |
| Semi-supervised classification with graph convolutional networks | Kipf & Welling | *ICLR* 2017 |
| Graph attention networks | Veličković et al. | *ICLR* 2018 |
| Smoothing and differentiation of data by simplified least squares procedures | Savitzky & Golay | *Analytical Chemistry* 1964 |
| PySINDy: A comprehensive Python package for robust sparse system identification | de Silva et al. | *JOSS* 2020 |

---

<div align="center">

**Built with PyTorch · SINDy · Streamlit**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drone-path-7kpgwzdwoih9vayensxuom.streamlit.app/)

</div>
