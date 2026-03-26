"""
Live demo: shows real-time resource deduction and release on a machine.
Run with: python demo_realtime.py
"""
import time
import requests

BASE = "http://127.0.0.1:8000"

def get_machine(machine_id):
    machines = requests.get(f"{BASE}/machines").json()
    # handle both list and dict with 'value' key
    items = machines if isinstance(machines, list) else machines.get("value", machines)
    for m in items:
        if m["machine_id"] == machine_id:
            return m
    return None

# ── Step 1: Find a low-load machine to watch ─────────────────────────────
machines = requests.get(f"{BASE}/machines").json()
items = machines if isinstance(machines, list) else machines.get("value", [])
watch = sorted(items, key=lambda m: m["load"])[0]
watch_id = watch["machine_id"]

print(f"\n{'='*55}")
print(f" BEFORE submitting task")
print(f"{'='*55}")
print(f" Machine     : {watch_id}")
print(f" Total CPU   : {watch['total_cpu']} cores")
print(f" Total RAM   : {watch['total_ram']} GB")
print(f" Available CPU: {watch['available_cpu']} cores")
print(f" Available RAM: {watch['available_ram']} GB")
print(f" Load        : {round(watch['load']*100,1)}%")

# ── Step 2: Submit a task ─────────────────────────────────────────────────
cpu_req, mem_req = 2.0, 6.0
print(f"\n Submitting task: cpu={cpu_req}, ram={mem_req} GB ...")
resp = requests.post(f"{BASE}/schedule_task", json={
    "cpu_required": cpu_req,
    "memory_required": mem_req,
    "priority": 7
}).json()

assigned = resp.get("assigned_machine") or resp.get("machine_id", "?")
duration = resp.get("execution_duration", 10)
task_id  = resp.get("task_id", "?")
print(f" → Assigned to : {assigned}")
print(f" → Task ID     : {task_id}")
print(f" → Will run for: {duration}s")

# ── Step 3: Immediately check the assigned machine ────────────────────────
m_after = get_machine(assigned)
print(f"\n{'='*55}")
print(f" IMMEDIATELY AFTER task assigned (resources locked)")
print(f"{'='*55}")
if m_after:
    print(f" Machine     : {m_after['machine_id']}")
    print(f" Available CPU: {m_after['available_cpu']} cores  ← was {watch['available_cpu'] if assigned == watch_id else '?'}")
    print(f" Available RAM: {m_after['available_ram']} GB     ← was {watch['available_ram'] if assigned == watch_id else '?'}")
    print(f" Load        : {round(m_after['load']*100,1)}%")

# ── Step 4: Poll every 2s until task completes ────────────────────────────
print(f"\n Waiting for task to complete ({duration}s)...")
wait = int(duration) + 4
for i in range(wait):
    time.sleep(1)
    tasks = requests.get(f"{BASE}/tasks").json()
    task_list = tasks if isinstance(tasks, list) else tasks.get("value", [])
    task = next((t for t in task_list if t["id"] == task_id), None)
    status = task["status"] if task else "unknown"
    print(f"  [{i+1:2d}s] Task {task_id} status: {status}", end="")
    if status == "completed":
        print(" ✓")
        break
    print()

# ── Step 5: Check machine after completion ────────────────────────────────
m_done = get_machine(assigned)
print(f"\n{'='*55}")
print(f" AFTER task completed (resources freed back)")
print(f"{'='*55}")
if m_done and m_after:
    print(f" Machine      : {m_done['machine_id']}")
    print(f" Available CPU: {m_done['available_cpu']} cores  ({m_after['available_cpu']} → {m_done['available_cpu']}, +{cpu_req} freed)")
    print(f" Available RAM: {m_done['available_ram']} GB  ({m_after['available_ram']} → {m_done['available_ram']}, +{mem_req} freed)")
    print(f" Load         : {round(m_done['load']*100,1)}%")

print(f"\n{'='*55}")
print(" SUMMARY: Real-time resource tracking")
print(f"{'='*55}")
print(f" 1. Task submitted  → machine CPU/RAM immediately reduced")
print(f" 2. Task running    → load increases, resources locked")
print(f" 3. Task completes  → CPU/RAM returned, load decreases")
print(f"{'='*55}\n")
