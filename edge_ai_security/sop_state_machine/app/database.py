"""
database.py — SQLAlchemy models and session factory for sop_state_machine.
Database file: /data/security.db
"""

import json
import os
from datetime import datetime

from sqlalchemy import (
    Column, Integer, Text, Float, ForeignKey,
    UniqueConstraint, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATA_DIR = os.getenv("DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "security.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ── ORM Models ─────────────────────────────────────────────────────────────────

class Person(Base):
    __tablename__ = "persons"

    id          = Column(Integer, primary_key=True)
    name        = Column(Text, nullable=False)
    role        = Column(Text, nullable=False)       # Manager/Security/Staff/Customer
    embedding   = Column(Text, nullable=False)       # JSON float array
    enrolled_at = Column(Text, default=lambda: datetime.utcnow().isoformat())
    active      = Column(Integer, default=1)

    def get_embedding(self):
        return json.loads(self.embedding)


class Zone(Base):
    __tablename__ = "zones"

    id               = Column(Integer, primary_key=True)
    name             = Column(Text, nullable=False)
    cam_id           = Column(Text, nullable=False)
    polygon_points   = Column(Text, nullable=False)  # JSON [[x,y],...]
    restricted_roles = Column(Text, default="[]")    # JSON list of role strings
    created_at       = Column(Text, default=lambda: datetime.utcnow().isoformat())

    def get_polygon(self):
        return json.loads(self.polygon_points)

    def get_restricted_roles(self):
        return json.loads(self.restricted_roles)


class ZoneAccess(Base):
    __tablename__ = "zone_access"

    person_id = Column(Integer, ForeignKey("persons.id"), primary_key=True)
    zone_id   = Column(Integer, ForeignKey("zones.id"),   primary_key=True)

    __table_args__ = (UniqueConstraint("person_id", "zone_id"),)


class Event(Base):
    __tablename__ = "events"

    id              = Column(Integer, primary_key=True)
    timestamp       = Column(Text)
    cam_id          = Column(Text)
    zone_name       = Column(Text)
    person_id       = Column(Integer, nullable=True)
    person_name     = Column(Text)
    person_role     = Column(Text)
    global_id       = Column(Text)
    status          = Column(Text)   # AUTHORIZED/WRONG_ZONE/RESTRICTED/UNKNOWN/MASKED
    confidence      = Column(Float, default=0.0)
    message         = Column(Text)
    frame_snapshot  = Column(Text)   # base64 JPEG


# ── Init ───────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Return a new database session."""
    return SessionLocal()


# ── CRUD helpers ───────────────────────────────────────────────────────────────

def get_all_persons(db: Session):
    return db.query(Person).filter(Person.active == 1).all()


def get_all_zones(db: Session):
    return db.query(Zone).all()


def get_zones_for_camera(db: Session, cam_id: str):
    return db.query(Zone).filter(Zone.cam_id == cam_id).all()


def has_zone_access(db: Session, person_id: int, zone_id: int) -> bool:
    row = db.query(ZoneAccess).filter(
        ZoneAccess.person_id == person_id,
        ZoneAccess.zone_id == zone_id,
    ).first()
    return row is not None


def log_event(db: Session, **kwargs):
    event = Event(**kwargs)
    db.add(event)
    db.commit()
    return event


def seed_default_zone(db: Session, cam_id: str):
    """Create a full-frame default zone if none exist for this camera."""
    existing = get_zones_for_camera(db, cam_id)
    if existing:
        return
    zone = Zone(
        name="Default Zone",
        cam_id=cam_id,
        # Full 1920×1080 frame — covers any standard CCTV resolution
        polygon_points=json.dumps([[0, 0], [1920, 0], [1920, 1080], [0, 1080]]),
        restricted_roles=json.dumps([]),
    )
    db.add(zone)
    db.commit()
