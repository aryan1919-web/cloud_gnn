"""
FastAPI main application – GNN Cloud Task Scheduler
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db, SessionLocal
from .seed import seed_machines
from .routes import scheduling, machines


# ---------------------------------------------------------------------------
# Background Worker – Task Lifecycle Manager
# ---------------------------------------------------------------------------
async def task_completion_worker():
    """Runs every second. Completes tasks whose duration has elapsed and
    releases their CPU + RAM back to the assigned machine."""
    # Import inside function to avoid circular imports at module level
    from .models import Task, Machine  # noqa: F401 (models needed for DB queries)

    print("[Worker] Task lifecycle manager started.")
    while True:
        await asyncio.sleep(1)
        try:
            db = SessionLocal()
            try:
                # Use naive local time — PostgreSQL TIMESTAMP WITHOUT TIME ZONE
                # stores datetimes in the DB server's local timezone (naive).
                # Comparing two naive local datetimes gives a correct elapsed time.
                now = datetime.now()
                running = db.query(Task).filter(Task.status == "running").all()

                released = []
                for task in running:
                    if task.start_time is None or task.execution_duration is None:
                        continue
                    start = task.start_time
                    # Strip timezone info if present so subtraction stays naive
                    if start.tzinfo is not None:
                        start = start.replace(tzinfo=None)
                    elapsed = (now - start).total_seconds()

                    if elapsed >= task.execution_duration:
                        machine = (
                            db.query(Machine)
                            .filter(Machine.machine_id == task.assigned_machine_id)
                            .first()
                        )
                        if machine:
                            machine.available_cpu = round(
                                min(machine.available_cpu + task.cpu_request, machine.total_cpu), 4
                            )
                            machine.available_ram = round(
                                min(machine.available_ram + task.memory_request, machine.total_ram), 4
                            )
                            machine.load = round(
                                (machine.total_cpu - machine.available_cpu)
                                / max(machine.total_cpu, 0.01),
                                4,
                            )
                        task.status = "completed"
                        released.append((task.id, task.assigned_machine_id))

                if released:
                    db.commit()
                    for tid, mid in released:
                        print(
                            f"[Worker] Task {tid} completed \u2192 resources released from {mid}"
                        )
            finally:
                db.close()
        except Exception as exc:
            print(f"[Worker Error] {exc}")


# ---------------------------------------------------------------------------
# App Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create/migrate tables, seed data, launch background worker."""
    init_db()
    db = SessionLocal()
    try:
        seed_machines(db)
    finally:
        db.close()

    # Launch background worker as a concurrent asyncio task
    worker_task = asyncio.create_task(task_completion_worker())
    yield
    # Graceful shutdown
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GNN Cloud Task Scheduler",
    description="Graph Neural Network-based task scheduling for distributed cloud systems",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (EC2 + local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(scheduling.router, tags=["Scheduling"])
app.include_router(machines.router, tags=["Machines & Metrics"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "gnn-cloud-scheduler", "version": "2.0.0"}
