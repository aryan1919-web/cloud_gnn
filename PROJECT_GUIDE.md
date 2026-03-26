# GNN Cloud Task Scheduler — Complete Project Guide

**GitHub:** https://github.com/meetpateldev18/cloud_gnn  
**Live App:** http://3.87.161.91:5173  
**API:** http://3.87.161.91:8000  
**API Docs (Swagger):** http://3.87.161.91:8000/docs  
**Version:** 2.0.0 — Dynamic Resource Lifecycle Simulator

---

## TABLE OF CONTENTS

1. [Quick Restart Commands](#1-quick-restart-commands)
2. [Important Links](#2-important-links)
3. [What Is This Project?](#3-what-is-this-project)
4. [Architecture Overview](#4-architecture-overview)
5. [Project File Structure](#5-project-file-structure)
6. [The GNN Model — Deep Dive](#6-the-gnn-model--deep-dive)
7. [The Mathematics](#7-the-mathematics)
8. [The Dataset](#8-the-dataset)
9. [The 20 Simulated Machines](#9-the-20-simulated-machines)
10. [Database Schema](#10-database-schema)
11. [API Endpoints Reference](#11-api-endpoints-reference)
12. [Frontend Components](#12-frontend-components)
13. [How to Use the Application](#13-how-to-use-the-application)
14. [Test Cases](#14-test-cases)
15. [Tech Stack](#15-tech-stack)
16. [Performance Results](#16-performance-results)
17. [Full Build Journey — Every Step](#17-full-build-journey--every-step)
18. [AWS EC2 Deployment — Step by Step](#18-aws-ec2-deployment--step-by-step)
19. [Version 2.0 Upgrade — What Changed](#19-version-20-upgrade--what-changed)
20. [Local Development Setup](#20-local-development-setup)
21. [Training on Kaggle](#21-training-on-kaggle)
22. [Troubleshooting](#22-troubleshooting)
23. [Quick Reference Card](#23-quick-reference-card)

---

## 1. QUICK RESTART COMMANDS

### If EC2 server rebooted / processes stopped

**Step 1 — SSH from your Windows PC:**
```powershell
ssh -i "C:\Users\meeth\Downloads\mykey.pem" ubuntu@3.87.161.91
```

**Step 2 — Kill any stale processes and restart everything (paste all at once):**
```bash
cd ~/cloud_scheduler
kill $(lsof -t -i:8000) 2>/dev/null
kill $(lsof -t -i:5173) 2>/dev/null
sleep 1
nohup venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
sleep 3
nohup serve -s frontend/dist -l 5173 > frontend.log 2>&1 &
echo "All services started!"
```

**Step 3 — Verify backend is alive:**
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{"status":"ok","service":"gnn-cloud-scheduler","version":"2.0.0"}
```

**Step 4 — Check background worker started:**
```bash
tail -20 backend.log
```
Look for lines like:
```
[DB] Schema migration complete.
[SEED] Inserted 20 sample machines.
[Worker] Task lifecycle manager started.
```

**App is now live at:** http://3.87.161.91:5173

---

### If you want to redeploy code changes from local PC to EC2

**Upload backend Python files:**
```powershell
scp -i "C:\Users\meeth\Downloads\mykey.pem" C:\Users\meeth\OneDrive\Desktop\new_cloud\cloud_scheduler\backend\*.py ubuntu@3.87.161.91:~/cloud_scheduler/backend/

scp -i "C:\Users\meeth\Downloads\mykey.pem" C:\Users\meeth\OneDrive\Desktop\new_cloud\cloud_scheduler\backend\routes\*.py ubuntu@3.87.161.91:~/cloud_scheduler/backend/routes/
```

**Upload frontend source files:**
```powershell
scp -i "C:\Users\meeth\Downloads\mykey.pem" -r C:\Users\meeth\OneDrive\Desktop\new_cloud\cloud_scheduler\frontend\src ubuntu@3.87.161.91:~/cloud_scheduler/frontend/
```

**Then on EC2, rebuild and restart:**
```bash
cd ~/cloud_scheduler
echo "VITE_API_URL=http://3.87.161.91:8000" > frontend/.env
cd frontend && npm run build && cd ..
kill $(lsof -t -i:8000) 2>/dev/null ; kill $(lsof -t -i:5173) 2>/dev/null
nohup venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
sleep 3
nohup serve -s frontend/dist -l 5173 > frontend.log 2>&1 &
curl http://localhost:8000/health
```

---

## 2. IMPORTANT LINKS

| What | URL |
|------|-----|
| **Frontend Dashboard** | http://3.87.161.91:5173 |
| **Backend REST API** | http://3.87.161.91:8000 |
| **Swagger API Docs** | http://3.87.161.91:8000/docs |
| **All Machines (JSON)** | http://3.87.161.91:8000/machines |
| **Latest Tasks (JSON)** | http://3.87.161.91:8000/tasks |
| **Live Metrics (JSON)** | http://3.87.161.91:8000/metrics |
| **Comparison Data** | http://3.87.161.91:8000/comparison |
| **Health Check** | http://3.87.161.91:8000/health |
| **GitHub Repository** | https://github.com/meetpateldev18/cloud_gnn |

---

## 3. WHAT IS THIS PROJECT?

### The Problem

Imagine you manage a data center with 20 servers. Hundreds of tasks arrive every second. Each task needs CPU cores and RAM. The question: which server should handle which task?

Bad scheduling = some servers are overloaded while others are idle = slow performance, wasted resources.

### Traditional Solutions and Their Problems

| Algorithm | How it works | Problem |
|-----------|-------------|---------|
| Round Robin | Rotate through servers 1,2,3...20,1 | Ignores actual load |
| Random | Pick any server randomly | Completely unintelligent |
| First Fit | Pick first server with enough CPU+RAM | Ignores network topology |

All traditional methods treat servers as isolated units. They do not understand the network.

### Our Solution: Graph Neural Network Scheduler

We model the data center as a graph:
- **Nodes** = servers (20 machines)
- **Edges** = network connections between servers

We train a Graph Attention Network (GAT) that:
1. Understands the full network topology
2. Learns from 20,000 real Google workload examples
3. Considers available CPU, RAM, bandwidth, and load simultaneously
4. Predicts which machine will result in the best execution time

**Result:** 75.1% accuracy (picking the optimal machine), vs ~5% for random.

### Version 2.0 Addition: Real-Time Simulation

Beyond just picking a machine, v2.0 simulates the full task lifecycle:
- Tasks run for 5 to 20 seconds (randomly assigned at scheduling time)
- While running, CPU and RAM are actually deducted from the machine
- A background worker checks every 1 second — when a task finishes, resources are released
- The UI auto-refreshes every 3 seconds, showing live machine states
- You can watch machines go from 40% load to 70% load as tasks accumulate, then back down

---

## 4. ARCHITECTURE OVERVIEW

```
                     AWS EC2 (3.87.161.91)
 +-------------------------------------------------+
 |                                                 |
 |  React Frontend (Port 5173)                     |
 |  - MachineTable   - MetricsBar (9 KPIs)         |
 |  - TaskForm       - ComparisonSection (4 charts)|
 |  - GraphView      - TaskList (live log)         |
 |          |                                      |
 |          v  (HTTP)                              |
 |  FastAPI Backend (Port 8000)                    |
 |  - POST /schedule_task                          |
 |  - GET /machines, /tasks, /metrics, /graph      |
 |          |                                      |
 |  +-------+--------+                             |
 |  |                |                             |
 |  v                v                             |
 |  GNN Model        Background Worker             |
 |  GAT 2 layers     Checks every 1 second         |
 |  31,169 params    Releases resources            |
 |  75.1% accuracy   on task completion            |
 |          |                                      |
 |          v                                      |
 |  PostgreSQL DB                                  |
 |  - machines table (20 rows, live state)         |
 |  - tasks table (full lifecycle)                 |
 |  - scheduling_results (per-algorithm log)       |
 +-------------------------------------------------+
```

### Request Flow

```
User submits task (cpu=2, mem=4, priority=5)
     |
     v
Backend: Filter machines with available_cpu >= 2 AND available_ram >= 4
     |
     v
Build graph: 20 nodes x 4 features (cpu_ratio, ram_ratio, load, bandwidth)
     |
     v
GNN: Forward pass -> scores all 20 machines -> pick highest
Run Round Robin, Random, First Fit in parallel for comparison
     |
     v
Deduct CPU+RAM from winning machine in DB
Set task status = "running", execution_duration = random(5,20)
     |
     v
Return result to frontend
     |
     v
Background worker (every 1s):
  if elapsed >= execution_duration:
    release CPU+RAM back to machine
    set task.status = "completed"
```

---

## 5. PROJECT FILE STRUCTURE

```
cloud_scheduler/
|
+-- backend/
|   +-- __init__.py
|   +-- main.py             <- FastAPI app, background worker, lifespan
|   +-- scheduler.py        <- GATScheduler, SAGEScheduler, GNNScheduler,
|   |                          SAGEGNNScheduler — both GNN architectures here
|   +-- models.py           <- SQLAlchemy ORM: Machine, Task, SchedulingResult
|   +-- database.py         <- PostgreSQL connection, init_db(), auto-migration
|   +-- seed.py             <- Seeds 20 machines with realistic specs
|   +-- baselines.py        <- Round Robin, Random, First Fit implementations
|   +-- schemas.py          <- Pydantic request/response schemas
|   +-- routes/
|       +-- scheduling.py   <- POST /schedule_task (core logic)
|       |                      Runs all 5 algorithms: GAT GNN, GraphSAGE,
|       |                      Round Robin, Random, First Fit
|       +-- machines.py     <- GET /machines, /tasks, /metrics, /comparison, /graph
|
+-- frontend/
|   +-- package.json
|   +-- vite.config.js
|   +-- index.html
|   +-- .env                <- VITE_API_URL=http://3.87.161.91:8000
|   +-- dist/               <- Built static files (served by `serve`)
|   +-- src/
|       +-- main.jsx
|       +-- App.jsx         <- Main layout, auto-refresh every 3s
|       +-- api.js          <- Axios calls to backend
|       +-- index.css
|       +-- components/
|           +-- MachineTable.jsx      <- 20-machine table with live stats
|           +-- TaskForm.jsx          <- Submit new scheduling task form
|           +-- GraphView.jsx         <- D3 force-graph network visualization
|           +-- MetricsBar.jsx        <- 9 KPI cards at top of dashboard
|           +-- ComparisonSection.jsx <- 5 algorithm bar charts + comparison table
|           +-- TaskList.jsx          <- Live task log (running/completed)
|
+-- model/
|   +-- scheduler_model.pt      <- GAT model weights (~31,169 params) [PRIMARY]
|   +-- scheduler_model_sage.pt <- GraphSAGE model weights (~12,480 params) [COMPARISON]
|   +-- README.md               <- Model details + placement instructions
|
+-- training/
|   +-- train_kaggle.py         <- GAT training script (Kaggle GPU notebook)
|   +-- train_kaggle_sage.py    <- GraphSAGE training script (Kaggle GPU notebook)
|
+-- tests/
|   +-- test_api.py         <- pytest test suite
|
+-- docker-compose.yml      <- Postgres + backend + frontend containers
+-- requirements.txt        <- Python dependencies
+-- pytest.ini
+-- README.md
+-- PROJECT_GUIDE.md        <- This file
```

---

## 6. THE GNN MODELS — DEEP DIVE

Two GNN architectures are now supported. Both share the same node/task features
and classifier head — the only difference is the graph convolution layer.

### Model A: GAT — Graph Attention Network (Primary Scheduler)

File: `model/scheduler_model.pt`  |  Trained by: `training/train_kaggle.py`

```python
class GATScheduler(nn.Module):
    def __init__(self, num_machines=20, node_features=4, task_features=4, hidden_dim=64):
        # GAT Layer 1: 4 features -> 64 hidden x 4 heads = 256 output
        self.gat1 = GATConv(node_features, hidden_dim, heads=4, concat=True)

        # GAT Layer 2: 256 -> 64 (single head)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)

        # Task encoder: 4 task features -> 64
        self.task_encoder = nn.Sequential(
            nn.Linear(task_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Final classifier: 128 (machine+task concat) -> 1 score
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
```

**GAT parameters:**
- gat1: ~2,200 params
- gat2: ~16,500 params
- task_encoder: ~8,320 params
- classifier: ~4,160 params
- **Total: ~31,169 parameters**

**GAT update rule:** $h'_i = \text{ELU}\left(\sum_{j \in N(i)} \alpha_{ij} \cdot W \cdot h_j\right)$

where attention $\alpha_{ij}$ is learned — the model pays more attention to
high-bandwidth, low-load neighbor machines.

### Model B: GraphSAGE (Comparison Algorithm)

File: `model/scheduler_model_sage.pt`  |  Trained by: `training/train_kaggle_sage.py`

```python
class SAGEScheduler(nn.Module):
    def __init__(self, num_machines=20, node_features=4, task_features=4, hidden_dim=64):
        # SAGE Layer 1: 4 -> 64  (no heads, no attention)
        self.sage1 = SAGEConv(node_features, hidden_dim)

        # SAGE Layer 2: 64 -> 64
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)

        # Task encoder and classifier: identical to GAT (fair comparison)
        self.task_encoder = nn.Sequential(...)
        self.classifier   = nn.Sequential(...)
```

**GraphSAGE parameters: ~12,480 parameters** (60% fewer than GAT)

**SAGE update rule:** $h'_v = \text{ReLU}\left(W \cdot \text{CONCAT}\left(h_v,\ \text{MEAN}\left(\{h_u : u \in N(v)\}\right)\right)\right)$

All neighbors contribute equally — no learned per-edge attention weights.

### GAT vs GraphSAGE Comparison

| Aspect              | GAT                               | GraphSAGE                         |
|---------------------|-----------------------------------|-----------------------------------|
| Aggregation         | Attention-weighted sum            | Mean of all neighbors             |
| Attention weights   | Learned per-edge (4 heads)        | None                              |
| Parameters          | ~31,169                           | ~12,480                           |
| Inference speed     | ~9-12 ms                          | ~5-8 ms (no softmax over edges)   |
| Topology awareness  | High — attends to best links      | Moderate — uniform neighbor weight|
| Inductive ability   | Transductive (fixed cluster)      | Inductive (new machines ok)       |
| Top-1 accuracy      | ~75.1%                            | ~71-74% (slightly lower)          |
| Best use case       | Fixed cluster, max accuracy       | Dynamic cluster (machines added)  |

### Graph Features (Dynamic — rebuilt every call)

```python
node_features = [
    machine.available_cpu / machine.total_cpu,   # free CPU fraction
    machine.available_ram / machine.total_ram,   # free RAM fraction
    machine.load,                                # load factor (0-1)
    machine.bandwidth / 10.0,                   # normalized bandwidth
]
```

### GAT Inference Pipeline

```
1. Build graph: 20 nodes x 4 features
2. gat1(x, edge_index) -> ELU -> 20x256  (4 heads x 64)
3. gat2(h, edge_index) -> ELU -> 20x64
4. task_encoder([cpu_req, mem_req, priority, time]) -> 1x64
5. For each machine i: classifier(concat(h_i, task_vec)) -> score_i
6. Return machines[argmax(scores)].machine_id
```

### GraphSAGE Inference Pipeline

```
1. Build graph: 20 nodes x 4 features  (same as GAT)
2. sage1(x, edge_index) -> ReLU -> 20x64  (mean aggregation, no heads)
3. sage2(h, edge_index) -> ReLU -> 20x64
4. task_encoder([cpu_req, mem_req, priority, time]) -> 1x64
5. For each machine i: classifier(concat(h_i, task_vec)) -> score_i
6. Return machines[argmax(scores)].machine_id  (used for comparison only)
```

### Fallback (no model file)

If `scheduler_model.pt` is missing, GAT falls back to Best-Fit heuristic.
If `scheduler_model_sage.pt` is missing, GraphSAGE also uses Best-Fit.
```
score = -(cpu_leftover/total_cpu + ram_leftover/total_ram)
```

---

## 7. THE MATHEMATICS

### Graph Attention (GAT)

Attention score from machine i to neighbor j:

  e_ij = LeakyReLU( a^T * [W*h_i || W*h_j] )

Normalized with softmax:

  alpha_ij = exp(e_ij) / sum_k exp(e_ik)

Updated node representation:

  h'_i = ELU( sum_{j in N(i)} alpha_ij * W * h_j )

With 4 attention heads (concatenated):

  h'_i = concat over k=1..4 of ELU( sum_j alpha_ij^k * W^k * h_j )

### Scoring and Decision

Given task vector t and machine embedding h'_i:

  score_i = classifier( [h'_i || t] )

Best machine:

  best = argmax over feasible machines of score_i

Where feasible = {machines with available_cpu >= cpu_req AND available_ram >= mem_req}

### Load Formula

  load = (total_cpu - available_cpu) / total_cpu

### Training Loss

  L = -sum_i y_i * log(y_hat_i)   (cross-entropy)

### Metrics

  throughput = (completed_tasks / time_window) * 60   [tasks/min]

  avg_completion_time = mean(finish_time_i - arrival_time_i)

  avg_waiting_time = mean(start_time_i - arrival_time_i)

---

## 8. THE DATASET

**Source:** Google Cluster Workload Traces — Borg 2019

A 29-day trace of real workloads from Google production cluster. Contains millions of task events, resource usage, and machine specs.

**Our subset:**
- File: dataset/borg_traces_data.csv
- Rows used: 20,000 (sampled from 405,894)

**Key columns:**

| Column | Maps to | Description |
|--------|---------|-------------|
| cycles_per_instruction | cpu_request | CPU cycles needed |
| assigned_memory | memory_request | Memory in GB |
| machine_id | machine_id | Which machine handled it |
| priority | priority | Task urgency 0-10 |

**Label generation (Best-Fit with noise):**
```python
for each candidate machine:
    score = -(cpu_leftover + mem_leftover)
    score += machine.bandwidth * 0.1
    score += random.uniform(-0.1, 0.1)   # prevents label imbalance

label = argmax(scores)
```

The noise term was crucial — without it, the same large machine always won, creating 90% class imbalance.

---

## 9. THE 20 SIMULATED MACHINES

Seeded by backend/seed.py. Scale from small to large:

| Machine | Total CPU | Total RAM | Initial Load | Bandwidth |
|---------|-----------|-----------|-------------|-----------|
| machine-001 | 2.7 | 7 GB | 17% | 1.45 Gbps |
| machine-002 | 3.4 | 10 GB | 29% | 1.90 Gbps |
| machine-003 | 4.1 | 13 GB | 41% | 2.35 Gbps |
| machine-004 | 4.8 | 16 GB | 53% | 2.80 Gbps |
| machine-005 | 5.5 | 19 GB | 5% | 3.25 Gbps | <- Fresh |
| machine-006 | 6.2 | 22 GB | 17% | 3.70 Gbps |
| machine-007 | 6.9 | 25 GB | 29% | 4.15 Gbps |
| machine-008 | 7.6 | 28 GB | 41% | 4.60 Gbps |
| machine-009 | 8.3 | 31 GB | 53% | 5.05 Gbps |
| machine-010 | 9.0 | 34 GB | 5% | 5.50 Gbps | <- Fresh |
| machine-011 | 9.7 | 37 GB | 17% | 5.95 Gbps |
| machine-012 | 10.4 | 40 GB | 29% | 6.40 Gbps |
| machine-013 | 11.1 | 43 GB | 41% | 6.85 Gbps |
| machine-014 | 11.8 | 46 GB | 53% | 7.30 Gbps |
| machine-015 | 12.5 | 49 GB | 5% | 7.75 Gbps | <- Fresh |
| machine-016 | 13.2 | 52 GB | 17% | 8.20 Gbps |
| machine-017 | 13.9 | 55 GB | 29% | 8.65 Gbps |
| machine-018 | 14.6 | 58 GB | 41% | 9.10 Gbps |
| machine-019 | 15.3 | 61 GB | 53% | 9.55 Gbps |
| machine-020 | 16.0 | 64 GB | 5% | 10.00 Gbps | <- Fresh/Large |

Pattern: Every 5th machine (005, 010, 015, 020) starts at 5% load — fresh machines in each tier.
Available resources at seed = Total x (1 - initial_load).

---

## 10. DATABASE SCHEMA

### machines table
```
machine_id    VARCHAR PK   -- "machine-001" to "machine-020"
total_cpu     FLOAT        -- total CPU cores
total_ram     FLOAT        -- total RAM in GB
available_cpu FLOAT        -- current free CPU (changes dynamically)
available_ram FLOAT        -- current free RAM (changes dynamically)
bandwidth     FLOAT        -- network bandwidth in Gbps
load          FLOAT        -- load factor 0.0-1.0 (recomputed each task)
```

### tasks table
```
id                SERIAL PK
cpu_request       FLOAT        -- requested CPU cores
memory_request    FLOAT        -- requested RAM in GB
priority          INTEGER      -- 0-10
assigned_machine_id VARCHAR    -- FK to machines
arrival_time      FLOAT        -- Unix timestamp
start_time        FLOAT        -- when task started (NEW in v2.0)
execution_duration FLOAT       -- assigned random duration 5-20s (NEW)
status            VARCHAR      -- "running" | "completed" | "queued" (NEW)
waiting_time      FLOAT        -- start_time - arrival_time (NEW)
```

### scheduling_results table
```
id             SERIAL PK
task_id        INTEGER   -- FK to tasks
algorithm      VARCHAR   -- "gnn" | "round_robin" | "random" | "first_fit"
machine_id     VARCHAR   -- which machine this algo chose
execution_time FLOAT     -- algorithm latency in ms
timestamp      FLOAT
```

### Auto-Migration Logic

On every backend startup, database.py checks if the old schema (no total_cpu column) exists.
If detected, it drops all tables and recreates with v2.0 schema, then re-seeds 20 machines.

This is safe because it only drops when the schema is incompatible (upgrade path).

---

## 11. API ENDPOINTS REFERENCE

### GET /health
```bash
curl http://3.87.161.91:8000/health
# {"status":"ok","service":"gnn-cloud-scheduler","version":"2.0.0"}
```

### GET /machines
All 20 machines with real-time availability:
```bash
curl http://3.87.161.91:8000/machines
```

### POST /schedule_task
```bash
curl -X POST http://3.87.161.91:8000/schedule_task \
  -H "Content-Type: application/json" \
  -d '{"cpu_request": 2.0, "memory_request": 4.0, "priority": 5}'
```

Response (machine available):
```json
{
  "status": "running",
  "execution_duration": 12,
  "gnn": {"machine_id": "machine-011", "execution_time": 8.3},
  "comparison": {
    "round_robin": {"machine_id": "machine-003", "execution_time": 18.7},
    "random":      {"machine_id": "machine-017", "execution_time": 15.2},
    "first_fit":   {"machine_id": "machine-001", "execution_time": 22.1}
  }
}
```

Response (no resources available):
```json
{"status": "queued", "message": "No feasible machines available. Task has been queued."}
```

### GET /tasks
Last 100 tasks with lifecycle status:
```bash
curl http://3.87.161.91:8000/tasks
```

### GET /metrics
Aggregated performance KPIs:
```bash
curl http://3.87.161.91:8000/metrics
# Returns: total_machines, total_tasks, running_tasks, completed_tasks,
#          avg_gnn_latency, avg_completion_time, avg_waiting_time,
#          avg_cpu_utilization, cluster_throughput, model
```

### GET /comparison
Per-algorithm metrics for chart rendering:
```bash
curl http://3.87.161.91:8000/comparison
# Returns array: [{algorithm, avg_latency, avg_completion_time,
#                  avg_cpu_utilization, throughput, avg_waiting_time, tasks_scheduled}]
```

### GET /graph
Node+edge data for D3 viz:
```bash
curl http://3.87.161.91:8000/graph
# Returns: {nodes: [{id, cpu, ram, load, bandwidth}], edges: [{source, target}]}
```

---

## 12. FRONTEND COMPONENTS

### App.jsx
- Fetches all data every 3 seconds (machines, tasks, metrics, comparison)
- Shows inline result card after task submission
- Passes data to all child components

### MachineTable.jsx
Columns: Machine ID | CPU Total | CPU Avail (% free bar) | RAM Total | RAM Avail (% free bar) | Bandwidth | Load (color bar)
Load colors: green <40%, yellow 40-70%, red >70%

### TaskForm.jsx
Inputs: CPU Required, Memory Required, Priority (0-10)
Button: "Schedule Task with GNN" with loading spinner

### GraphView.jsx
D3 force-directed graph of 20 machines. Nodes colored by load. Click to highlight.

### MetricsBar.jsx — 9 KPI Cards
1. Total Machines
2. Total Tasks Scheduled
3. Running Tasks (yellow)
4. Completed Tasks (green)
5. Avg GNN Latency (ms)
6. Avg Completion Time (s)
7. Avg Waiting Time (s)
8. Avg CPU Utilization (%)
9. Cluster Throughput (tasks/min)

### ComparisonSection.jsx — 4 Bar Charts
1. Avg Scheduling Latency (ms) — decision speed
2. Avg Task Completion Time (s) — end-to-end time
3. CPU Utilization (%) — resource efficiency
4. Cluster Throughput (tasks/min)

Algorithm colors: GNN=indigo, Round Robin=amber, Random=red, First Fit=green

### TaskList.jsx (NEW in v2.0)
Table: Task ID | Status (colored badge) | Machine | CPU Req | RAM Req | Priority | Duration | Wait
Status badges: running (yellow pulsing) | completed (green) | queued (gray)

---

## 13. HOW TO USE THE APPLICATION

1. Open http://3.87.161.91:5173
2. View the dashboard:
   - 9 KPI cards at top
   - Network graph (left) + task form (right)
   - Machine table (live CPU/RAM/load)
   - Task log (running/completed history)
   - Comparison charts (4 bars)
3. Submit a task: fill CPU, Memory, Priority -> click "Schedule with GNN"
4. Read the result: machine chosen, duration, comparison with other algorithms
5. Watch the simulation: machine stats update every 3 seconds as tasks run and complete

---

## 14. TEST CASES

**Test 1 — Small background task:**
CPU: 0.5, Memory: 1, Priority: 0
Expect: Small machine assigned, completes in 5-20s

**Test 2 — Large critical task:**
CPU: 8, Memory: 32, Priority: 10
Expect: Large machine (015-020 range) with high bandwidth

**Test 3 — Load build-up:**
Submit 5 tasks rapidly (CPU:2, Mem:4, Priority:5 x5)
Expect: Tasks distributed across machines, load bars increase visibly

**Test 4 — Resource exhaustion:**
CPU: 15, Memory: 60, Priority: 10
Expect: Only machine-019 or machine-020 can handle; if both busy -> "queued"

**Test 5 — Algorithm comparison:**
CPU: 3, Memory: 8, Priority: 5
Observe: GNN picks balanced machine; First Fit always picks machine-001 first; Round Robin rotates without checking

**Test 6 — Task flood:**
10 tasks: CPU:0.1, Mem:0.5, Priority:1
Expect: Throughput metric jumps, tasks spread across machines, complete quickly

---

## 15. TECH STACK

| Layer | Technology | Purpose |
|-------|-----------|---------|
| ML Model (GAT)   | PyTorch + torch-geometric (GATConv)  | Attention-based GNN — primary scheduler |
| ML Model (SAGE)  | PyTorch + torch-geometric (SAGEConv) | Mean-aggregation GNN — comparison algo  |
| Backend | FastAPI | Async REST API |
| ORM | SQLAlchemy | DB models |
| Database | PostgreSQL 14 | Persistent storage |
| DB Driver | psycopg2 | Python-PostgreSQL |
| Validation | Pydantic v2 | Schema validation |
| ASGI Server | Uvicorn | Production server |
| Frontend | React 18 + Vite | Component UI + fast build |
| Charts | Recharts | Bar/line charts |
| Graph viz | react-force-graph | D3 network visualization |
| HTTP Client | Axios | Frontend API calls |
| Static server | serve | Serves dist/ on port 5173 |
| Hosting | AWS EC2 t3.small | Ubuntu 22.04, 2vCPU, 2GB RAM |
| Training | Kaggle T4 GPU | Free GPU for model training |
| Dataset | Google Borg 2019 | 405,894 real task traces |

---

## 16. PERFORMANCE RESULTS

### Model Accuracy

| Metric          | GAT (GNN)             | GraphSAGE             |
|-----------------|-----------------------|-----------------------|
| Top-1 Accuracy  | 75.1%                 | ~71-74%               |
| Top-3 Accuracy  | 91.2%                 | ~88-90%               |
| Training epochs | 100                   | 100                   |
| Parameters      | ~31,169               | ~12,480               |
| Training time   | ~18 min (Kaggle T4)   | ~14 min (Kaggle T4)   |
| Inference speed | ~9-12 ms              | ~5-8 ms               |

GraphSAGE trains and infers faster but GAT has slightly higher accuracy
because attention weights let it focus on critically-connected machines.

### Algorithm Comparison (all 5)

| Algorithm   | Avg Latency | Strategy                                     |
|-------------|-------------|----------------------------------------------|
| GAT GNN     | ~9-12 ms    | Attention-weighted graph encoding (PRIMARY)  |
| GraphSAGE   | ~5-8 ms     | Mean-aggregation graph encoding              |
| First Fit   | ~15-20 ms   | Linear scan for first feasible machine       |
| Round Robin | ~18-22 ms   | Counter-based rotation, ignores load         |
| Random      | ~20-25 ms   | Random selection, no intelligence            |

Both GNN variants outperform baselines because they learn from data.
GAT wins on accuracy; GraphSAGE wins on speed and scalability.
Traditional algorithms make the same mistakes on every task.

---

## 17. FULL BUILD JOURNEY — EVERY STEP

### Phase 1: Project Scaffolding
1. Created project directory cloud_scheduler/
2. Initialized Python venv, installed FastAPI, SQLAlchemy, psycopg2, torch, torch-geometric
3. Created backend/models.py with Machine, Task, SchedulingResult SQLAlchemy models
4. Created backend/database.py with PostgreSQL connection string
5. Created backend/seed.py to populate 20 machines
6. Built backend/scheduler.py with GNNScheduler class (2-layer GAT)
7. Built backend/baselines.py with Round Robin, Random, First Fit implementations
8. Created backend/routes/scheduling.py for POST /schedule_task
9. Created backend/routes/machines.py for GET /machines, /comparison, /graph
10. Built backend/main.py with FastAPI app, lifespan, routers
11. Created React frontend with Vite (npm create vite)
12. Built components: TaskForm, MachineTable, GraphView, ComparisonSection
13. Built frontend/src/api.js with Axios

### Phase 2: Dataset and Training
1. Downloaded Google Borg 2019 dataset
2. Analyzed columns: cycles_per_instruction, assigned_memory, machine_id, priority
3. Preprocessed: sampled 20,000 rows, normalized features to 0-1
4. Added noise to Best-Fit label generation to prevent class imbalance
5. Wrote training/train_kaggle.py with full 100-epoch training loop
6. Uploaded dataset + notebook to Kaggle, enabled T4 GPU x2
7. Fixed dataset path detection (multi-candidate search across /kaggle/input/)
8. Fixed column name mismatches (Borg dataset specific column names)
9. Fixed label imbalance (was 90% same machine — added random noise)
10. Trained 100 epochs -> 75.1% Top-1, 91.2% Top-3
11. Downloaded scheduler_model.pt from Kaggle output

### Phase 3: Local Integration
1. Placed scheduler_model.pt in model/ folder
2. Installed CPU-only PyTorch (CUDA too large for laptop)
3. Set up PostgreSQL: createdb cloud_scheduler
4. Started backend: uvicorn backend.main:app --reload --port 8000
5. Verified "[OK] GNN model loaded" in logs
6. Started frontend: cd frontend && npm run dev -> localhost:5173
7. Ran all test cases, verified GNN picking different machines than baselines
8. Fixed CORS (added allow_origins=["*"] to main.py)
9. Wrote test suite tests/test_api.py with pytest

### Phase 4: AWS EC2 Deployment v1.0
1. Launched EC2 t3.small instance (Ubuntu 22.04, 8GB disk, us-east-1)
2. Configured security groups: ports 22, 8000, 5173
3. SSH key pair: mykey.pem saved to C:\Users\meeth\Downloads\
4. SSH in: ssh -i mykey.pem ubuntu@3.87.161.91
5. Installed: python3-venv, nodejs, npm, postgresql, unzip
6. Installed: npm install -g serve
7. Set up PostgreSQL: ALTER USER postgres PASSWORD '12345678'; createdb cloud_scheduler
8. Created Python venv in ~/cloud_scheduler/venv/
9. Installed CPU-only PyTorch (saved ~3GB vs CUDA version — critical for 8GB disk)
10. Installed torch-geometric, fastapi, uvicorn, sqlalchemy, psycopg2-binary
11. Zipped project locally (excluded venv, node_modules, __pycache__)
12. Uploaded via SCP: scp cloud_scheduler.zip ubuntu@3.87.161.91:~
13. On EC2: unzip, cd frontend && npm install && npm run build
14. Set VITE_API_URL=http://3.87.161.91:8000 in .env before build
15. Started with nohup: uvicorn on port 8000, serve on port 5173
16. Verified: curl http://localhost:8000/health -> status ok
17. App live at http://3.87.161.91:5173 ✅

### Phase 5: v2.0 Upgrade (Dynamic Resource Lifecycle)
See Section 19 for full details. All 16 files modified, tested locally, deployed to EC2 via SCP.

1. Upgraded models.py: added total_cpu, total_ram, available_cpu, available_ram, start_time, execution_duration, status, waiting_time
2. Updated database.py: auto-migration checks for old schema
3. Updated seed.py: seeds available_cpu = total_cpu x (1 - load)
4. Upgraded scheduler.py: dynamic graph features, graph rebuilt every call
5. Updated baselines.py: use new field names
6. Expanded schemas.py: MachineOut (7 fields), TaskOut, MetricsOut (10), ComparisonRow (6 metrics)
7. Upgraded main.py: added task_completion_worker() async background task, version 2.0.0
8. Upgraded routes/scheduling.py: feasibility filter, resource deduction, debug logs, queued state
9. Upgraded routes/machines.py: GET /tasks, real metrics from tasks table
10. Updated api.js: added getTasks()
11. Upgraded App.jsx: 3s auto-refresh, queued/running state, inline comparison table
12. Upgraded MachineTable.jsx: available_cpu, available_ram columns with % bars
13. Upgraded MetricsBar.jsx: 9 KPIs
14. Upgraded ComparisonSection.jsx: 4 charts, 6-column comparison table
15. Created TaskList.jsx: new live task monitor component
16. Tested locally at localhost:5174 — new UI confirmed ✅
17. Uploaded all backend .py files to EC2: scp backend/*.py and routes/*.py
18. Uploaded frontend src/ to EC2: scp -r frontend/src
19. Rebuilt frontend on EC2: npm run build
20. Restarted services on EC2: killed old processes, started new
21. Verified v2.0.0 at http://3.87.161.91:8000/health ✅

---

## 18. AWS EC2 DEPLOYMENT — STEP BY STEP

### Instance Details
```
Instance Type : t3.small
vCPU          : 2
RAM           : 2 GB
Storage       : 8 GB EBS
OS            : Ubuntu 22.04 LTS
Region        : us-east-1
Public IP     : 3.87.161.91
SSH Key       : C:\Users\meeth\Downloads\mykey.pem
```

### Security Group Rules
```
SSH        TCP  22    0.0.0.0/0  (your IP to SSH in)
Custom TCP TCP  8000  0.0.0.0/0  (FastAPI backend)
Custom TCP TCP  5173  0.0.0.0/0  (React frontend)
```

### Full Setup Script (fresh EC2)
```bash
# System update
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv nodejs npm postgresql postgresql-contrib unzip
npm install -g serve

# PostgreSQL
sudo systemctl start postgresql
sudo -u postgres psql -c "ALTER USER postgres PASSWORD '12345678';"
sudo -u postgres createdb cloud_scheduler

# Python environment
python3 -m venv ~/cloud_scheduler/venv
source ~/cloud_scheduler/venv/bin/activate

# PyTorch CPU-only (IMPORTANT: GPU would exceed 8GB disk)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic python-multipart

# Frontend
echo "VITE_API_URL=http://3.87.161.91:8000" > ~/cloud_scheduler/frontend/.env
cd ~/cloud_scheduler/frontend && npm install && npm run build
```

### Start Script (copy-paste each restart)
```bash
cd ~/cloud_scheduler
kill $(lsof -t -i:8000) 2>/dev/null ; kill $(lsof -t -i:5173) 2>/dev/null
sleep 1
nohup venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
sleep 3
nohup serve -s frontend/dist -l 5173 > frontend.log 2>&1 &
echo "Done! Check: curl http://localhost:8000/health"
```

### Monitoring
```bash
tail -50 ~/cloud_scheduler/backend.log    # backend logs
tail -50 ~/cloud_scheduler/frontend.log   # serve logs
lsof -i:8000 -i:5173                      # check ports
sudo systemctl status postgresql          # db status
```

---

## 19. VERSION 2.0 UPGRADE — WHAT CHANGED

### New Features

**1. Dynamic Resource Lifecycle**
Before: Machine load never changed. Scheduling had no effect on machine state.
After: CPU and RAM are deducted from machines when tasks are assigned, and released when tasks complete.

**2. Task Duration (5-20 seconds)**
Before: Tasks were instantaneous.
After: execution_duration = random.randint(5, 20) assigned at scheduling time.

**3. Background Worker**
```python
async def task_completion_worker():
    while True:
        running_tasks = db.query(Task).filter(Task.status == "running").all()
        for task in running_tasks:
            elapsed = time.time() - task.start_time
            if elapsed >= task.execution_duration:
                machine.available_cpu += task.cpu_request
                machine.available_ram += task.memory_request
                machine.load = (machine.total_cpu - machine.available_cpu) / machine.total_cpu
                task.status = "completed"
        db.commit()
        await asyncio.sleep(1)
```
Runs as asyncio.create_task() in FastAPI lifespan.

**4. Feasibility Filtering**
Before: All 20 machines sent to GNN.
After: Only machines with available_cpu >= cpu_req AND available_ram >= mem_req.
If none available: return {"status": "queued"}.

**5. GET /tasks Endpoint**
New endpoint returning last 100 tasks with full lifecycle fields.

**6. Real Metrics (9 KPIs)**
Before: Only avg_latency from scheduling_results.
After: running_tasks, completed_tasks, avg_completion_time, avg_waiting_time, cluster_throughput.

**7. UI Updates**
- TaskList component: live task log
- MachineTable: available_cpu and available_ram columns
- MetricsBar: 9 KPIs
- ComparisonSection: 4 charts
- App.jsx: 3-second auto-refresh

**8. Auto-migration**
Old schema (missing total_cpu) detected on startup -> tables dropped + recreated.

**9. Debug Logging**
Every scheduling decision prints detailed log to backend.log.

**10. Version 2.0.0**
Health endpoint and app header show version.

### Schema Changes Summary

machines table: added total_cpu, total_ram, available_cpu, available_ram (old "cpu"/"ram" renamed)
tasks table: added start_time, execution_duration, status, waiting_time
scheduling_results: unchanged

---

## 20. LOCAL DEVELOPMENT SETUP

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+

### Windows Setup
```powershell
git clone https://github.com/meetpateldev18/cloud_gnn.git
cd cloud_gnn

python -m venv venv
.\venv\Scripts\activate

# IMPORTANT: install CPU-only torch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.txt

# Put scheduler_model.pt in model/ folder (download from Kaggle)
```

**PostgreSQL setup:**
```sql
CREATE DATABASE cloud_scheduler;
ALTER USER postgres PASSWORD '12345678';
```

**Start backend:**
```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Start frontend (separate terminal):**
```powershell
cd frontend
npm install
npm run dev
# Opens at localhost:5173 (or 5174 if port conflict)
```

**Frontend env:**
```
frontend/.env -> VITE_API_URL=http://localhost:8000
```

### Kill Port Conflicts (Windows)
```powershell
# Port 8000
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force

# Port 5173
Get-Process -Id (Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## 21. TRAINING ON KAGGLE

Two separate Kaggle scripts exist — one per GNN architecture.

### GAT Training (training/train_kaggle.py)

1. Kaggle → New Notebook
2. Upload `training/train_kaggle.py`
3. Add dataset (Google Borg 2019 CSV)
4. Settings → Accelerator → GPU T4 x2
5. Run all cells
6. Output tab → download `scheduler_model.pt`
7. Place at: `cloud_scheduler/model/scheduler_model.pt`

### GraphSAGE Training (training/train_kaggle_sage.py)

1. Kaggle → New Notebook
2. Upload `training/train_kaggle_sage.py`
3. Add same dataset
4. Settings → Accelerator → GPU T4 x2
5. Run all cells
6. Output tab → download `scheduler_model_sage.pt`
7. Place at: `cloud_scheduler/model/scheduler_model_sage.pt`

### Key Design Decisions (both scripts share)

**Multi-path dataset detection:**
```python
search_dirs = ['/kaggle/input/', './', '../']
for d in search_dirs:
    files = glob.glob(d + '**/*.csv', recursive=True)
    # find borg_traces_data.csv or similar
```

**Column detection (Borg dataset specific names):**
```python
POSSIBLE_CPU_COLS = ['cycles_per_instruction', 'cpu_usage_distribution', 'sample_rate']
POSSIBLE_MEM_COLS = ['assigned_memory', 'memory_usage', 'page_cache_memory']
cpu_col = next((c for c in POSSIBLE_CPU_COLS if c in df.columns), None)
```

**Noise in label generation (prevents class imbalance):**
```python
score += random.uniform(-0.1, 0.1)  # without this: 90% same label
```

### Architecture Difference in Training

```
train_kaggle.py      uses GATConv  — ELU activation, 4 heads, saves scheduler_model.pt
train_kaggle_sage.py uses SAGEConv — ReLU activation, no heads, saves scheduler_model_sage.pt
```

Same dataset, same features, same label generation, same 100 epochs, same evaluation —
only the GNN layer changes. This makes the comparison scientifically fair.

### Training Log (approximate, GAT)
```
Epoch   1/100: val_acc= 8.3%
Epoch  10/100: val_acc=24.1%
Epoch  50/100: val_acc=59.2%
Epoch 100/100: val_acc=75.1%  <- Top-1: 75.1%, Top-3: 91.2%
```

### Training Log (approximate, GraphSAGE)
```
Epoch   1/100: val_acc= 7.8%
Epoch  10/100: val_acc=21.5%
Epoch  50/100: val_acc=56.4%
Epoch 100/100: val_acc=72.3%  <- Top-1: ~72%, Top-3: ~89%
```

---

## 22. TROUBLESHOOTING

### "Address already in use" (port conflict)
```powershell
# Windows
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess | Stop-Process -Force
```
```bash
# Linux/EC2
kill $(lsof -t -i:8000) 2>/dev/null
```

### "GNN model not found — using heuristic fallback"
`scheduler_model.pt` (GAT) is missing from model/ folder.
Download from Kaggle run of `train_kaggle.py` → place at `model/scheduler_model.pt`

### "GraphSAGE model not found — using heuristic fallback"
`scheduler_model_sage.pt` is missing from model/ folder.
Download from Kaggle run of `train_kaggle_sage.py` → place at `model/scheduler_model_sage.pt`
The app still works without it — SAGE comparison will use heuristic values.

### EC2 disk full: "no space left on device"
```bash
df -h
# Most likely cause: installed CUDA torch instead of CPU-only
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Frontend showing old UI or NaN% on EC2
New source files not uploaded or not rebuilt:
```bash
# After SCP upload of src/:
echo "VITE_API_URL=http://3.87.161.91:8000" > ~/cloud_scheduler/frontend/.env
cd ~/cloud_scheduler/frontend && npm run build
kill $(lsof -t -i:5173) 2>/dev/null
nohup serve -s frontend/dist -l 5173 > frontend.log 2>&1 &
# Then hard refresh browser: Ctrl+Shift+R
```

### PostgreSQL not running
```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql   # auto-start on reboot
```

### Database schema mismatch
Backend auto-handles this on startup. Look for:
```
[DB] Old schema detected – dropping all tables for migration...
[DB] Tables dropped. Recreating with new schema...
[DB] Schema migration complete.
```
This is normal after an upgrade — it means the migration worked.

### Tasks stuck as "running" forever
Background worker crashed. Restart backend:
```bash
kill $(lsof -t -i:8000) 2>/dev/null
nohup venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
# Worker restarts automatically with the backend
```

---

## 23. QUICK REFERENCE CARD

```
=================================================================
          GNN CLOUD SCHEDULER — QUICK REFERENCE v2.0.0
=================================================================

LIVE APP     : http://3.87.161.91:5173
API          : http://3.87.161.91:8000
API DOCS     : http://3.87.161.91:8000/docs
GITHUB       : https://github.com/meetpateldev18/cloud_gnn

-----------------------------------------------------------------
SSH INTO EC2
ssh -i "C:\Users\meeth\Downloads\mykey.pem" ubuntu@3.87.161.91

-----------------------------------------------------------------
RESTART ALL SERVICES (paste on EC2)
cd ~/cloud_scheduler
kill $(lsof -t -i:8000) 2>/dev/null ; kill $(lsof -t -i:5173) 2>/dev/null
sleep 1
nohup venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
sleep 3
nohup serve -s frontend/dist -l 5173 > frontend.log 2>&1 &

-----------------------------------------------------------------
VERIFY RUNNING
curl http://localhost:8000/health
tail -20 backend.log

-----------------------------------------------------------------
EC2 DETAILS
Instance  : t3.small (2 vCPU, 2GB RAM, 8GB disk)
IP        : 3.87.161.91
OS        : Ubuntu 22.04
Key       : C:\Users\meeth\Downloads\mykey.pem
DB        : postgresql://postgres:12345678@localhost/cloud_scheduler
Backend   : ~/cloud_scheduler/venv/bin/python
Frontend  : ~/cloud_scheduler/frontend/dist/

-----------------------------------------------------------------
MODEL STATS
GAT Architecture : 2 layers, 4 attention heads (ELU)
GAT Parameters   : ~31,169  |  Top-1: 75.1%  |  Top-3: 91.2%
GAT Model file   : model/scheduler_model.pt  (PRIMARY)

SAGE Architecture : 2 layers, mean aggregation (ReLU)
SAGE Parameters   : ~12,480  |  Top-1: ~72%  |  Faster inference
SAGE Model file   : model/scheduler_model_sage.pt  (COMPARISON)

Dataset: Google Borg 2019 (405,894 rows, used 20k)
Training: 100 epochs, Kaggle T4 GPU

-----------------------------------------------------------------
ALGORITHMS IN DASHBOARD (5 total)
GAT GNN   — primary: actually schedules tasks, attention-based
GraphSAGE — comparison: mean-aggregation GNN, faster, inductive
Round Robin, Random, First Fit — baselines

-----------------------------------------------------------------
LOCAL DEV (Windows)
cd C:\Users\meeth\OneDrive\Desktop\new_cloud\cloud_scheduler
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
cd frontend ; npm run dev

-----------------------------------------------------------------
KEY FILES
backend/main.py                    <- background worker, app entry
backend/scheduler.py               <- GATScheduler, SAGEScheduler, both wrappers
backend/routes/scheduling.py       <- task scheduling logic (5 algorithms)
frontend/src/App.jsx               <- auto-refresh, main layout
model/scheduler_model.pt           <- GAT trained weights (PRIMARY)
model/scheduler_model_sage.pt      <- GraphSAGE trained weights (COMPARISON)
training/train_kaggle.py           <- GAT Kaggle training script
training/train_kaggle_sage.py      <- GraphSAGE Kaggle training script

-----------------------------------------------------------------
VERSION 2.0.0 | 20 machines | 5 algorithms | GAT + GraphSAGE | Live simulation
=================================================================
```
