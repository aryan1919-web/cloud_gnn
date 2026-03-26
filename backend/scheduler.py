"""Graph Neural Network scheduler – inference wrapper."""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric imports (optional at import time for environments
# where only inference is needed without full pyg install)
try:
    from torch_geometric.nn import GATConv, SAGEConv
    from torch_geometric.data import Data

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# ---------------------------------------------------------------------------
# GAT Model Definition
# ---------------------------------------------------------------------------
class GATScheduler(nn.Module):
    """Two-layer Graph Attention Network for task-to-machine scheduling."""

    def __init__(
        self,
        node_features: int = 4,
        task_features: int = 4,
        hidden_dim: int = 64,
        heads: int = 4,
        num_machines: int = 20,
    ):
        super().__init__()
        self.gat1 = GATConv(node_features, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.task_encoder = nn.Sequential(
            nn.Linear(task_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_machines = num_machines

    def forward(self, x, edge_index, task_feat):
        """
        Parameters
        ----------
        x : Tensor [num_machines, node_features]
        edge_index : Tensor [2, num_edges]
        task_feat : Tensor [batch, task_features]

        Returns
        -------
        scores : Tensor [batch, num_machines]
        """
        # Graph encoding
        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))  # [num_machines, hidden_dim]

        # Task encoding
        t = self.task_encoder(task_feat)  # [batch, hidden_dim]

        # Score each machine for each task
        batch_size = t.size(0)
        num_nodes = h.size(0)
        h_exp = h.unsqueeze(0).expand(batch_size, -1, -1)  # [B, M, H]
        t_exp = t.unsqueeze(1).expand(-1, num_nodes, -1)  # [B, M, H]
        combined = torch.cat([h_exp, t_exp], dim=-1)  # [B, M, 2H]
        scores = self.classifier(combined).squeeze(-1)  # [B, M]
        return scores


# ---------------------------------------------------------------------------
# GraphSAGE Model Definition
# ---------------------------------------------------------------------------
class SAGEScheduler(nn.Module):
    """
    Two-layer GraphSAGE network for task-to-machine scheduling.

    GraphSAGE vs GAT — core architectural difference:
      GAT:  h_i = ELU( sum_j alpha_ij * W * h_j )  — attention-weighted sum
      SAGE: h_i = ReLU( W * CONCAT(h_i, MEAN(h_N(i))) ) — mean aggregation

    GraphSAGE has no attention heads (fewer params), runs faster at inference,
    and generalises inductively to unseen nodes (new machines added to cluster).
    """

    def __init__(
        self,
        node_features: int = 4,
        task_features: int = 4,
        hidden_dim: int = 64,
        num_machines: int = 20,
    ):
        super().__init__()
        self.sage1 = SAGEConv(node_features, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.task_encoder = nn.Sequential(
            nn.Linear(task_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_machines = num_machines

    def forward(self, x, edge_index, task_feat):
        """
        Parameters
        ----------
        x          : Tensor [num_machines, node_features]
        edge_index : Tensor [2, num_edges]
        task_feat  : Tensor [batch, task_features]

        Returns
        -------
        scores : Tensor [batch, num_machines]
        """
        h = F.relu(self.sage1(x, edge_index))   # [num_machines, hidden_dim]
        h = F.relu(self.sage2(h, edge_index))   # [num_machines, hidden_dim]

        t = self.task_encoder(task_feat)         # [batch, hidden_dim]

        batch_size = t.size(0)
        num_nodes  = h.size(0)
        h_exp = h.unsqueeze(0).expand(batch_size, -1, -1)  # [B, M, H]
        t_exp = t.unsqueeze(1).expand(-1, num_nodes, -1)   # [B, M, H]
        combined = torch.cat([h_exp, t_exp], dim=-1)        # [B, M, 2H]
        scores = self.classifier(combined).squeeze(-1)      # [B, M]
        return scores


# ---------------------------------------------------------------------------
# Scheduler Wrapper
# ---------------------------------------------------------------------------
class GNNScheduler:
    """Loads a trained GAT model and provides scheduling predictions."""

    def __init__(self, model_path: str = "model/scheduler_model.pt", num_machines: int = 20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_machines = num_machines
        self.model: GATScheduler | None = None
        self._machine_features: np.ndarray | None = None
        self._edge_index: torch.Tensor | None = None

        if not PYG_AVAILABLE:
            print("[WARN] torch_geometric not installed – GNN inference disabled.")
            return

        if os.path.isfile(model_path):
            self.model = GATScheduler(num_machines=num_machines).to(self.device)
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"[OK] GNN model loaded from {model_path}")
        else:
            print(f"[WARN] Model file not found at {model_path} – using heuristic fallback.")

    # ----- graph helpers -----
    def set_graph(self, machine_features: np.ndarray, edge_index: np.ndarray):
        """Set the current cloud infrastructure graph."""
        self._machine_features = machine_features
        self._edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        self.num_machines = machine_features.shape[0]

    def _build_default_graph(self, machines: list[dict]) -> None:
        n = len(machines)
        # Node features: normalized available_cpu, available_ram, load, bandwidth
        feats = np.array(
            [
                [
                    m["available_cpu"] / max(m["total_cpu"], 1.0),   # fraction of CPU still free
                    m["available_ram"] / max(m["total_ram"], 1.0),   # fraction of RAM still free
                    m["load"],                                         # 0-1 utilization
                    m["bandwidth"] / 10.0,                            # normalize to ~0-1 (max 10 Gbps)
                ]
                for m in machines
            ],
            dtype=np.float32,
        )
        # Fully connected edges (small graph)
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = np.array([src, dst])
        self.set_graph(feats, edge_index)

    # ----- prediction -----
    def predict(self, task: dict, machines: list[dict]) -> int:
        """Return index into *machines* of the best machine for the given task.

        The graph is always rebuilt from the current machine state so that
        available_cpu / available_ram / load reflect real-time allocations.
        """
        # Always rebuild with current state (dynamic resource updates)
        self._build_default_graph(machines)

        task_feat = np.array(
            [[task["cpu_request"], task["memory_request"], task["priority"], 0.0]],
            dtype=np.float32,
        )

        if self.model is None:
            return self._heuristic_schedule(task, machines)

        with torch.no_grad():
            x = torch.tensor(self._machine_features, dtype=torch.float32).to(self.device)
            t = torch.tensor(task_feat, dtype=torch.float32).to(self.device)
            scores = self.model(x, self._edge_index, t)  # [1, M]
            best_idx = int(scores.argmax(dim=1).item())
        return best_idx

    @staticmethod
    def _heuristic_schedule(task: dict, machines: list[dict]) -> int:
        """Heuristic fallback: pick machine with most remaining capacity."""
        best, best_score = 0, -1e9
        for i, m in enumerate(machines):
            remaining_cpu = m["available_cpu"] - task["cpu_request"]
            remaining_mem = m["available_ram"] - task["memory_request"]
            if remaining_cpu >= 0 and remaining_mem >= 0:
                score = remaining_cpu + remaining_mem * 0.1 + m["bandwidth"]
                if score > best_score:
                    best, best_score = i, score
        return best


# ---------------------------------------------------------------------------
# GraphSAGE Scheduler Wrapper
# ---------------------------------------------------------------------------
class SAGEGNNScheduler:
    """
    Loads a trained GraphSAGE model and provides scheduling predictions.

    Used as a comparison algorithm alongside the GAT-based GNNScheduler.
    The model is loaded from model/scheduler_model_sage.pt (trained via
    training/train_kaggle_sage.py).

    Key difference from GNNScheduler:
      - Uses SAGEScheduler (SAGEConv layers, mean aggregation, no attention)
      - Fewer parameters (~12K vs ~31K for GAT)
      - Faster inference, inductively generalises to new machines
    """

    def __init__(self, model_path: str = "model/scheduler_model_sage.pt", num_machines: int = 20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_machines = num_machines
        self.model: SAGEScheduler | None = None
        self._machine_features: np.ndarray | None = None
        self._edge_index: torch.Tensor | None = None

        if not PYG_AVAILABLE:
            print("[WARN] torch_geometric not installed – GraphSAGE inference disabled.")
            return

        if os.path.isfile(model_path):
            self.model = SAGEScheduler(num_machines=num_machines).to(self.device)
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"[OK] GraphSAGE model loaded from {model_path}")
        else:
            print(f"[WARN] GraphSAGE model not found at {model_path} – using heuristic fallback.")

    def set_graph(self, machine_features: np.ndarray, edge_index: np.ndarray):
        self._machine_features = machine_features
        self._edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        self.num_machines = machine_features.shape[0]

    def _build_default_graph(self, machines: list[dict]) -> None:
        n = len(machines)
        feats = np.array(
            [
                [
                    m["available_cpu"] / max(m["total_cpu"], 1.0),
                    m["available_ram"] / max(m["total_ram"], 1.0),
                    m["load"],
                    m["bandwidth"] / 10.0,
                ]
                for m in machines
            ],
            dtype=np.float32,
        )
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = np.array([src, dst])
        self.set_graph(feats, edge_index)

    def predict(self, task: dict, machines: list[dict]) -> int:
        """Return index into *machines* of the best machine for the given task."""
        self._build_default_graph(machines)

        task_feat = np.array(
            [[task["cpu_request"], task["memory_request"], task["priority"], 0.0]],
            dtype=np.float32,
        )

        if self.model is None:
            return GNNScheduler._heuristic_schedule(task, machines)

        with torch.no_grad():
            x = torch.tensor(self._machine_features, dtype=torch.float32).to(self.device)
            t = torch.tensor(task_feat, dtype=torch.float32).to(self.device)
            scores = self.model(x, self._edge_index, t)  # [1, M]
            best_idx = int(scores.argmax(dim=1).item())
        return best_idx
