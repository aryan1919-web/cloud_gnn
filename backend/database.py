"""SQLAlchemy database connection and session management."""

import os
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:12345678@localhost:5432/cloud_scheduler",
)
# Render uses 'postgres://' but SQLAlchemy requires 'postgresql://'
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency – yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables, auto-migrating schema if old columns are detected."""
    inspector = inspect(engine)
    needs_reset = False

    if inspector.has_table("machines"):
        cols = {c["name"] for c in inspector.get_columns("machines")}
        # Old schema used cpu_capacity / ram_capacity; new schema uses total_cpu / total_ram
        if "total_cpu" not in cols:
            needs_reset = True
            print("[DB] Old schema detected – dropping all tables for migration...")

    if needs_reset:
        Base.metadata.drop_all(bind=engine)
        print("[DB] Tables dropped. Recreating with new schema...")

    Base.metadata.create_all(bind=engine)
    if needs_reset:
        print("[DB] Schema migration complete. Machines will be re-seeded.")
