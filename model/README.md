# Model Weights Directory

Place your trained model files here after training on Kaggle.
The backend automatically loads both files on startup.

---

## scheduler_model.pt — GAT (Graph Attention Network)

Trained by: `training/train_kaggle.py`

Architecture: 2-layer GATConv with 4 attention heads
Parameters: ~31,169
Aggregation: Attention-weighted sum — learns which neighbor machines matter more
Activation: ELU

Used as: **Primary scheduler** (GAT picks the actual machine for every task)

```
Download from Kaggle output → place at:
  cloud_scheduler/model/scheduler_model.pt
```

---

## scheduler_model_sage.pt — GraphSAGE (Graph Sample and Aggregate)

Trained by: `training/train_kaggle_sage.py`

Architecture: 2-layer SAGEConv with mean aggregation
Parameters: ~12,480 (60% fewer than GAT)
Aggregation: Mean of all neighbors — uniform weight, no attention softmax
Activation: ReLU

Used as: **Comparison algorithm** (shown in the dashboard alongside GAT,
Round Robin, Random, and First Fit)

```
Download from Kaggle output → place at:
  cloud_scheduler/model/scheduler_model_sage.pt
```

---

## GAT vs GraphSAGE — Side-by-Side

| Aspect              | GAT (scheduler_model.pt)          | GraphSAGE (scheduler_model_sage.pt) |
|---------------------|-----------------------------------|-------------------------------------|
| Layer type          | GATConv                           | SAGEConv                            |
| Aggregation         | Attention-weighted sum            | Mean of neighbors                   |
| Attention weights   | Learned per-edge (4 heads)        | None                                |
| Parameters          | ~31,169                           | ~12,480                             |
| Inference speed     | Slower (softmax over edges)       | Faster (simple mean)                |
| Topology awareness  | High — pays attention to important links | Moderate — treats all neighbors equally |
| Inductive ability   | Transductive (fixed graph)        | Inductive (generalises to new nodes)|
| Best for            | Fixed cluster, max accuracy       | Dynamic cluster (machines added)    |
| Update rule         | h = ELU(Σ αᵢⱼ · W · hⱼ)         | h = ReLU(W · CONCAT(hᵢ, MEAN(Nᵢ)))|

---

## Fallback Behaviour

If either model file is missing, the backend does NOT crash.
- `scheduler_model.pt` missing → GAT falls back to Best-Fit heuristic
- `scheduler_model_sage.pt` missing → GraphSAGE falls back to Best-Fit heuristic

The `/comparison` endpoint will still return data; the GraphSAGE entry
will use heuristic results rather than learned predictions.

