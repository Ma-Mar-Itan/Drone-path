<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/SINDy-Symbolic_Regression-8A2BE2?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>

<br><br>

# 🚁 Drone Path · Equation Finder

**Draw a drone trajectory. A Graph Neural Network learns its physics.<br>
Sparse regression writes the equations. RK4 re-flies the drone.**

<br>

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drone-path-7kpgwzdwoih9vayensxuom.streamlit.app/)

<sub>No installation · No GPU · No data · Just draw.</sub>

<br>

[🚀 Live App](https://drone-path-7kpgwzdwoih9vayensxuom.streamlit.app/) · [What It Does](#-what-it-does) · [Quick Start](#-quick-start) · [The Math](#-the-mathematics) · [The ML](#-the-machine-learning) · [Safety System](#-safety-envelope) · [Contributing](#-contributing)

</div>

---

## 🎯 What It Does

This project is an **educational demo** for researchers, students, and engineers curious about system identification and graph learning. It shows how a single user-drawn curve can be automatically turned into a physics equation using modern machine learning — with no training data, no labels, and no pre-training.

**Typical use cases:**
- Learning how SINDy and GNNs work by playing with a visual interface
- Demonstrating symbolic regression to non-technical audiences
- Exploring how trajectory curvature, speed, and acceleration relate
- A starting point for robotics system-identification research

### The 7-Step Pipeline

```
✏️  Draw        →  sketch a 2-D flight path with your mouse
📐  Resample    →  arc-length resampling to uniform time steps
🔬  Smooth      →  Savitzky–Golay filter computes clean derivatives
🕸️  Graph       →  trajectory becomes a temporal graph
🧠  GNN         →  Graph Neural Network predicts accelerations
📝  SINDy       →  sparse regression writes human-readable equations
▶️  Simulate    →  RK4 integrates the equations to re-fly the drone
🛡️  Safety      →  speed, acceleration and curvature are flagged
```

**Output example — equations discovered from a figure-8 path:**
```
d²x/dt² = −0.83·vx + 2.14·cos(y)
d²y/dt² = −4.91·vy + 1.73·x
```

These are real differential equations your drone would need to satisfy to follow that exact path.

---

## ⚡ Quick Start

### Option A — Use the Live App (recommended)

Click the badge below. No setup required.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drone-path-7kpgwzdwoih9vayensxuom.streamlit.app/)

### Option B — Run Locally

```bash
# Clone
git clone https://github.com/Ma-Mar-Itan/Drone-path.git
cd Drone-path

# Install
pip install -r requirements.txt

# Run
streamlit run App.py
```

Opens at `http://localhost:8501`. Requires **Python 3.10+**, CPU only.

### Usage in 4 clicks

1. **Draw** a continuous path on the canvas (spirals, figure-8s, zigzags all work)
2. Click **🔧 Fit Model** — GNN trains in ~10 seconds
3. Click **▶️ Simulate** — RK4 re-flies the drone from the equations
4. Explore the **Equations, Safety, and Training** tabs

> **Tip:** Draw smooth, continuous strokes. Sharp disconnected dots give poor results.  
> **Tip:** If simulation diverges, raise the **Sparsity λ** slider to force a simpler equation.

---

## 🔢 The Mathematics

### Step 1 · Arc-Length Resampling

Raw mouse strokes are unevenly spaced — fast on straight sections, slow at corners. We need uniform time steps for any physical analysis.

Compute cumulative arc-length, then interpolate to **N uniformly-spaced points**:

```
s_k = Σ √(Δxᵢ² + Δyᵢ²)

Interpolate x(s), y(s) at   s = 0,  L/(N−1),  2L/(N−1), ...
```

---

### Step 2 · Savitzky–Golay Differentiation

Within a sliding window of width *w*, fit a polynomial of degree *p* and differentiate it analytically:

```
x̂(t) = Σ cₖ tᵏ     →     vx = dx̂/dt     →     ax = d²x̂/dt²
```

**Why not finite differences?**  
Finite differences amplify noise with error **O(Δt)**. Savitzky–Golay achieves **O(Δt^(p+1))** — the difference between clean, usable derivatives and noise.

---

### Step 3 · Normalisation *(the stability fix)*

Before any regression, every feature is normalised to zero-mean, unit-std:

```
x_norm = (x − μₓ) / σₓ        for each of [x, y, vx, vy, ax, ay]
```

**Why this is critical:**  
The feature library contains `sin(x)` and `cos(x)`. At raw pixel scale (x ≈ 200–600 px), these wrap hundreds of times and produce useless coefficients that explode during simulation. In normalised space (x ∈ [−2, +2] approximately), they are well-defined and physically meaningful.

The normalisation statistics are stored and applied at **every RK4 step** during simulation — inputs are normalised in, accelerations are de-normalised out.

---

### Step 4 · SINDy — Sparse Identification of Nonlinear Dynamics

Build a **feature library** Θ of 15 candidate basis functions (evaluated on normalised state):

```
Θ = [ 1  x  y  vx  vy  x²  y²  xy  vx²  vy²  vx·vy  sin(x)  sin(y)  cos(x)  cos(y) ]
```

Solve two sparse regression problems:

```
ax_norm ≈ Θ · w_x        (find the x-acceleration equation)
ay_norm ≈ Θ · w_y        (find the y-acceleration equation)
```

Subject to **w being sparse** — most coefficients should be zero, keeping only the terms that genuinely matter.

**STLSQ Algorithm — Sequential Thresholded Least Squares:**

```
1.  w ← (ΘᵀΘ)⁻¹ Θᵀ a          ordinary least squares
2.  set wᵢ = 0  for all |wᵢ| < λ    apply sparsity threshold
3.  refit on active (non-zero) features only
4.  repeat until no more terms are eliminated
```

The **λ slider** in the sidebar controls how aggressively terms are removed. Higher λ = simpler equation with fewer terms.

---

### Step 5 · RK4 Forward Simulation

The discovered equations define a complete ODE system:

```
d/dt [x, y, vx, vy] = [vx,  vy,  f(x,y,vx,vy),  g(x,y,vx,vy)]
```

Integrated with **4th-order Runge–Kutta**:

```
s_{n+1} = sₙ + (Δt/6)(k₁ + 2k₂ + 2k₃ + k₄)

k₁ = F(sₙ)                      k₂ = F(sₙ + Δt/2 · k₁)
k₃ = F(sₙ + Δt/2 · k₂)         k₄ = F(sₙ + Δt · k₃)
```

Local truncation error **O(Δt⁵)** — 4 orders of magnitude more accurate than Euler for the same step size.

---

## 🧠 The Machine Learning

### Graph Neural Network

The GNN is built in **pure PyTorch** (no PyTorch Geometric dependency).

```
Input:   X ∈ ℝᴺˣ⁴     node features [x, y, vx, vy] at each time step
         Ã ∈ ℝᴺˣᴺ     row-normalised adjacency (temporal neighbourhood)

① Message:    m = MLP_msg(X)              [Linear → Tanh → Linear]
② Aggregate:  M = Ã · m                   neighbourhood average
③ Concat:     Z = [X ‖ M]                 own features + context
④ Update:     h = MLP_update(Z)           [Linear → Tanh → Linear → Tanh]
⑤ Output:     (ax, ay) = W · h            predict acceleration
```

**Why graph structure?**  
A standard MLP sees only the instantaneous snapshot `[x, y, vx, vy]` — it cannot distinguish a point at the *start* of a sharp turn from a point at the *peak*. The graph lets each node look **±k steps** in time before predicting, giving it the context it needs to understand curvature and anticipate acceleration changes.

### Training Strategy

| Aspect | Choice | Why |
|--------|--------|-----|
| Supervision | Teacher forcing | Feed ground-truth states, supervise accelerations |
| Loss function | MSE on (ax, ay) | Direct regression on physical quantities |
| Optimiser | Adam | Adaptive learning rate, fast on small datasets |
| LR scheduler | ReduceLROnPlateau | Halves LR when progress stalls |
| Early stopping | Patience = 30 epochs | Prevents overfitting to one trajectory |
| Gradient clipping | Max norm = 1.0 | Stabilises training on sharp turns |
| Hardware | CPU only | Full training in 5–15 s on any laptop |

### GNN → SINDy Handoff

The GNN is a black box — high accuracy, zero interpretability. SINDy converts it to glass:

```
GNN predicts (ax, ay) at every time step
       ↓
These predictions form the regression targets
       ↓
STLSQ finds the sparse equation that best explains them
       ↓
Result: 2–6 term differential equation  ✓  interpretable  ✓  simulatable
```

---

## 🛡️ Safety Envelope

Three metrics are computed at every time step:

| Metric | Formula | Units | Physical meaning |
|--------|---------|-------|-----------------|
| Speed | `v = √(vx² + vy²)` | px/s | Motor saturation / battery |
| Acceleration | `a = √(ax² + ay²)` | px/s² | Structural stress / stall |
| Curvature | `κ = \|vx·ay − vy·ax\| / v³` | 1/px | Turning radius (1/κ = radius) |

A time step is flagged **unsafe** (red) if *any* threshold is exceeded. The Safety tab shows:
- Percentage of the path that is unsafe — drawn vs simulated
- Peak values of all three metrics
- Where along the path violations occur
- Whether the simulation produces a safer or more aggressive trajectory

**Setting thresholds** — all units are in pixels per second, scaled to your drawing speed:

| Drawing style | Speed limit | Acceleration limit |
|--------------|------------|-------------------|
| Slow, careful | 150–300 px/s | 500–1 000 px/s² |
| Normal pace | 300–600 px/s | 1 000–2 500 px/s² |
| Fast, sweeping | 600–1 500 px/s | 2 500–5 000 px/s² |

---

## 📊 Error Metrics

After simulation, two metrics compare the simulated path to the drawn path:

**RMSE** — average positional error across all time-matched point pairs:
```
RMSE = √( (1/N) · Σ ||p_true − p_sim||² )
```

**Hausdorff Distance** — worst-case deviation anywhere along the paths:
```
H(P, Q) = max( max_{p∈P} min_{q∈Q} d(p,q),   max_{q∈Q} min_{p∈P} d(p,q) )
```

Hausdorff is the most conservative metric — it catches the single worst point of disagreement, making it ideal for safety-critical comparison.

---

## 📁 Project Structure

```
Drone-path/
│
├── App.py                  ← Streamlit UI: drawing, controls, tabs, plots
├── requirements.txt        ← Python dependencies
├── LICENSE                 ← MIT
├── README.md
│
└── src/
    ├── __init__.py
    ├── pipeline.py         ← Arc-length resampling · SG smoothing · safety metrics
    ├── gnn_model.py        ← Temporal GNN in pure PyTorch
    ├── symbolic.py         ← SINDy wrapper + manual STLSQ · normalisation logic
    └── simulator.py        ← RK4 integrator · normalise/de-normalise per step
```

---

## 🤝 Contributing

Contributions are welcome. Here is how to get started:

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/<your-username>/Drone-path.git
cd Drone-path
pip install -r requirements.txt
streamlit run App.py       # verify everything works
```

**Before submitting a pull request:**
- Test your change with at least two different drawn paths
- Verify that simulation still produces sensible trajectories
- Keep changes focused — one fix or feature per PR

**Good first issues to tackle:**
- Add a screenshot/GIF to this README
- Improve error messages for edge-case inputs
- Add unit tests for `pipeline.py` and `symbolic.py`
- Support drawing multiple strokes (currently only the last stroke is used)

**Reporting bugs:**  
Open a GitHub Issue with: the path you drew, the slider values used, and the error message or unexpected behaviour you saw.

---

## 🗺️ Roadmap

Planned improvements (contributions welcome):

- [ ] GPU support for larger networks
- [ ] 3-D trajectory support (add z-axis)
- [ ] Jerk (third derivative) as an additional safety metric
- [ ] Export discovered equations as LaTeX
- [ ] Side-by-side comparison of GNN vs plain MLP
- [ ] Save and load trajectories between sessions
- [ ] Jupyter notebook walkthroughs of each pipeline step

---

## ⚠️ Known Limitations

- **Single trajectory** — the model learns the physics of *one* drawn path, not a general model
- **Pixel units** — all distances are in canvas pixels, not real-world metres
- **Simulation divergence** — highly nonlinear paths can still produce divergent simulations; try raising λ
- **Not for production** — this is a research and educational demo, not a flight controller

---

## 📚 References

| Paper | Authors | Where |
|-------|---------|-------|
| Discovering governing equations from data by sparse identification of nonlinear dynamics | Brunton, Proctor & Kutz | PNAS 2016 |
| Semi-supervised classification with graph convolutional networks | Kipf & Welling | ICLR 2017 |
| Graph attention networks | Veličković et al. | ICLR 2018 |
| Smoothing and differentiation of data by simplified least squares procedures | Savitzky & Golay | Analytical Chemistry 1964 |
| PySINDy: A comprehensive Python package for robust sparse system identification | de Silva et al. | JOSS 2020 |

---

## 📄 License

MIT — see [LICENSE](LICENSE). Use freely, attribution appreciated.

---

<div align="center">

Built with PyTorch · SINDy · Streamlit · Pure Python

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drone-path-7kpgwzdwoih9vayensxuom.streamlit.app/)

*Draw → Fit → Simulate → Analyse*

</div>
