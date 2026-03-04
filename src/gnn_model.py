"""
src/gnn_model.py
----------------
A lightweight Graph Neural Network (GNN) over the time-series graph.

Architecture
------------
- Nodes  : time steps t_i, each with feature vector [x, y, vx, vy]
- Edges  : (i → i+1) directed temporal edges + optional k-nearest-time edges
- Message passing aggregates neighbour features and updates node representations
- Output head: 2-D acceleration (ax, ay)

We implement message-passing **from scratch in pure PyTorch** (no PyG dependency)
so the demo runs anywhere without heavy optional installs.

Conceptually this is equivalent to a single-layer Graph Attention Network
(Veličković et al., 2018) with uniform weights, applied over a temporal graph.
"""

import torch
import torch.nn as nn
import numpy as np


class TemporalGNN(nn.Module):
    """
    Single-layer GNN for trajectory dynamics.

    For each node i we aggregate features from neighbours N(i) defined by
    the adjacency matrix A.  The aggregated message is concatenated with the
    node's own features and passed through an MLP to predict acceleration.

    Parameters
    ----------
    in_dim   : input feature dimension per node  (default 4: x,y,vx,vy)
    hidden   : hidden layer width
    out_dim  : output dimension (default 2: ax, ay)
    """

    def __init__(self, in_dim: int = 4, hidden: int = 64, out_dim: int = 2):
        super().__init__()

        # Message MLP: transforms source-node features into messages
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )

        # Update MLP: combines aggregated message + own features → hidden repr
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden + in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Output head: hidden → acceleration
        self.out_head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (N, in_dim) node features
        adj : (N, N) row-normalised adjacency matrix (float)

        Returns
        -------
        acc : (N, out_dim) predicted accelerations
        """
        # Compute messages for all nodes
        messages = self.msg_mlp(x)          # (N, hidden)

        # Aggregate: m_i = Σ_j A[i,j] * msg_j
        agg = adj @ messages                 # (N, hidden)

        # Concatenate own features with aggregated message
        combined = torch.cat([x, agg], dim=-1)   # (N, hidden + in_dim)

        # Update
        h = self.update_mlp(combined)        # (N, hidden)

        # Predict acceleration
        acc = self.out_head(h)               # (N, out_dim)
        return acc


def build_adjacency(N: int, k_neighbors: int = 2, device: str = "cpu") -> torch.Tensor:
    """
    Build a row-normalised adjacency matrix for a temporal graph.

    Edges:
      - i → i+1  (forward temporal)
      - i → i-1  (backward temporal, for smoothness)
      - i → j  for j in [i-k .. i+k] (k-nearest in time)

    Self-loops are included so each node also passes its own feature.

    Parameters
    ----------
    N           : number of nodes
    k_neighbors : half-window for k-nearest-time edges (default 2)

    Returns
    -------
    A : (N, N) row-normalised adjacency (torch.float32)
    """
    A = torch.zeros(N, N, device=device)

    for i in range(N):
        # Self-loop
        A[i, i] = 1.0
        # Temporal neighbourhood
        for delta in range(-k_neighbors, k_neighbors + 1):
            j = i + delta
            if 0 <= j < N:
                A[i, j] = 1.0

    # Row-normalise (avoid division by zero)
    row_sums = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
    A = A / row_sums
    return A


def train_gnn(
    data: dict,
    epochs: int = 500,
    lr: float = 1e-3,
    patience: int = 30,
    hidden: int = 64,
    k_neighbors: int = 3,
    device: str = "cpu",
) -> tuple["TemporalGNN", list[float], float, float]:
    """
    Train the GNN on a single trajectory using teacher forcing.

    Teacher forcing: feed ground-truth states as node features;
    supervise with ground-truth accelerations.

    Parameters
    ----------
    data     : output of pipeline.smooth_and_differentiate (normalised)
    epochs   : max training epochs
    lr       : learning rate
    patience : early-stopping patience (epochs without improvement)

    Returns
    -------
    model        : trained TemporalGNN
    loss_history : list of per-epoch losses
    x_mu, x_std  : normalisation stats for state features
    """
    # ---- Prepare tensors ----
    x = np.stack([data["x"], data["y"], data["vx"], data["vy"]], axis=1).astype(np.float32)
    y = np.stack([data["ax"], data["ay"]], axis=1).astype(np.float32)

    # Normalise inputs
    x_mu = x.mean(axis=0)
    x_std = x.std(axis=0) + 1e-8
    x_n = (x - x_mu) / x_std

    # Normalise targets
    y_mu = y.mean(axis=0)
    y_std = y.std(axis=0) + 1e-8
    y_n = (y - y_mu) / y_std

    X = torch.tensor(x_n, device=device)
    Y = torch.tensor(y_n, device=device)

    N = X.shape[0]
    A = build_adjacency(N, k_neighbors=k_neighbors, device=device)

    model = TemporalGNN(in_dim=4, hidden=hidden, out_dim=2).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=10, factor=0.5)
    loss_fn = nn.MSELoss()

    loss_history = []
    best_loss = float("inf")
    best_state = None
    no_improve = 0

    model.train()
    for epoch in range(epochs):
        optimiser.zero_grad()
        pred = model(X, A)
        loss = loss_fn(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step(loss)

        l = loss.item()
        loss_history.append(l)

        if l < best_loss - 1e-7:
            best_loss = l
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, loss_history, x_mu, x_std, y_mu, y_std


def gnn_predict(
    model: "TemporalGNN",
    data: dict,
    x_mu: np.ndarray,
    x_std: np.ndarray,
    y_mu: np.ndarray,
    y_std: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run GNN inference over the full trajectory.

    Returns
    -------
    ax_pred, ay_pred : predicted accelerations in original units
    """
    x = np.stack([data["x"], data["y"], data["vx"], data["vy"]], axis=1).astype(np.float32)
    x_n = (x - x_mu) / x_std
    X = torch.tensor(x_n, device=device)

    N = X.shape[0]
    A = build_adjacency(N, device=device)

    with torch.no_grad():
        pred_n = model(X, A).cpu().numpy()

    pred = pred_n * y_std + y_mu
    return pred[:, 0], pred[:, 1]