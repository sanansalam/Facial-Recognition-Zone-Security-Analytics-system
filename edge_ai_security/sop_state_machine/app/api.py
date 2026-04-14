"""
api.py — FastAPI REST interface for sop_state_machine.
Runs on port 8000 in a daemon thread alongside the ZMQ loop.
"""

import json
import threading
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import database as db_module
import identity_tracker as tracker

app = FastAPI(title="Edge AI Security — SOP State Machine", version="1.0.0")


# ── Pydantic request/response models ───────────────────────────────────────────

class ZoneCreate(BaseModel):
    name: str
    cam_id: str
    polygon_points: List[List[float]]   # [[x,y], ...]
    restricted_roles: List[str] = []


class PersonCreate(BaseModel):
    name: str
    role: str
    embedding: List[float]


class ZoneAccessCreate(BaseModel):
    person_id: int
    zone_id: int


# ── Zone endpoints ─────────────────────────────────────────────────────────────

@app.get("/zones")
def list_zones():
    db = db_module.get_db()
    try:
        zones = db_module.get_all_zones(db)
        return [
            {
                "id": z.id,
                "name": z.name,
                "cam_id": z.cam_id,
                "polygon_points": z.get_polygon(),
                "restricted_roles": z.get_restricted_roles(),
                "created_at": z.created_at,
            }
            for z in zones
        ]
    finally:
        db.close()


@app.get("/zones/{cam_id}")
def list_zones_for_camera(cam_id: str):
    db = db_module.get_db()
    try:
        zones = db_module.get_zones_for_camera(db, cam_id)
        return [
            {
                "id": z.id,
                "name": z.name,
                "cam_id": z.cam_id,
                "polygon_points": z.get_polygon(),
                "restricted_roles": z.get_restricted_roles(),
            }
            for z in zones
        ]
    finally:
        db.close()


@app.post("/zones", status_code=201)
def create_zone(payload: ZoneCreate):
    db = db_module.get_db()
    try:
        zone = db_module.Zone(
            name=payload.name,
            cam_id=payload.cam_id,
            polygon_points=json.dumps(payload.polygon_points),
            restricted_roles=json.dumps(payload.restricted_roles),
            created_at=datetime.utcnow().isoformat(),
        )
        db.add(zone)
        db.commit()
        db.refresh(zone)
        return {"id": zone.id, "name": zone.name, "cam_id": zone.cam_id}
    finally:
        db.close()


@app.delete("/zones/{zone_id}", status_code=200)
def delete_zone(zone_id: int):
    db = db_module.get_db()
    try:
        zone = db.query(db_module.Zone).filter(
            db_module.Zone.id == zone_id
        ).first()
        if not zone:
            raise HTTPException(status_code=404, detail="Zone not found")
        db.delete(zone)
        db.commit()
        return {"deleted": zone_id}
    finally:
        db.close()


# ── Person endpoints ───────────────────────────────────────────────────────────

@app.get("/persons")
def list_persons():
    db = db_module.get_db()
    try:
        persons = db_module.get_all_persons(db)
        return [
            {
                "id": p.id,
                "name": p.name,
                "role": p.role,
                "enrolled_at": p.enrolled_at,
                "active": p.active,
            }
            for p in persons
        ]
    finally:
        db.close()


@app.post("/persons", status_code=201)
def enroll_person(payload: PersonCreate):
    db = db_module.get_db()
    try:
        person = db_module.Person(
            name=payload.name,
            role=payload.role,
            embedding=json.dumps(payload.embedding),
            enrolled_at=datetime.utcnow().isoformat(),
        )
        db.add(person)
        db.commit()
        db.refresh(person)
        return {"id": person.id, "name": person.name, "role": person.role}
    finally:
        db.close()


@app.post("/zone_access", status_code=201)
def grant_zone_access(payload: ZoneAccessCreate):
    db = db_module.get_db()
    try:
        access = db_module.ZoneAccess(
            person_id=payload.person_id,
            zone_id=payload.zone_id,
        )
        db.add(access)
        db.commit()
        return {"person_id": payload.person_id, "zone_id": payload.zone_id}
    finally:
        db.close()


# ── Identity / trail endpoints ─────────────────────────────────────────────────

@app.get("/identities")
def list_identities():
    identities = tracker.list_identities()
    return [
        {
            "global_id":       v["global_id"],
            "person_name":     v["person_name"],
            "person_role":     v["person_role"],
            "first_seen_cam":  v["first_seen_cam"],
            "last_seen_cam":   v["last_seen_cam"],
            "trail_length":    len(v["trail"]),
            "violations":      len(v["violations"]),
        }
        for v in identities.values()
    ]


@app.get("/identities/{global_id}")
def get_identity(global_id: str):
    identity = tracker.get_identity(global_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")
    return {
        "global_id":    identity["global_id"],
        "person_name":  identity["person_name"],
        "person_role":  identity["person_role"],
        "first_seen":   identity["first_seen_cam"],
        "last_seen":    identity["last_seen_cam"],
        "trail": [
            {
                "cam_id":    t["cam_id"],
                "timestamp": datetime.utcfromtimestamp(t["timestamp"]).isoformat(),
                "zone":      t["zone"],
            }
            for t in identity["trail"]
        ],
        "violations": identity["violations"],
    }


# ── Report endpoints ───────────────────────────────────────────────────────────

@app.get("/report/summary")
def summary_report():
    db = db_module.get_db()
    try:
        events = db.query(db_module.Event).all()
        violations = [e for e in events if e.status != "AUTHORIZED"]

        by_camera: dict = {}
        by_type: dict = {}
        for v in violations:
            by_camera[v.cam_id] = by_camera.get(v.cam_id, 0) + 1
            by_type[v.status]   = by_type.get(v.status, 0) + 1

        identities = tracker.list_identities()
        return {
            "total_violations":  len(violations),
            "by_camera":         by_camera,
            "by_type":           by_type,
            "persons_tracked":   len(identities),
            "unknown_persons":   tracker.count_unknown(),
        }
    finally:
        db.close()


@app.get("/events")
def list_events(limit: int = 50):
    db = db_module.get_db()
    try:
        events = (
            db.query(db_module.Event)
            .order_by(db_module.Event.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id":          e.id,
                "timestamp":   e.timestamp,
                "cam_id":      e.cam_id,
                "zone_name":   e.zone_name,
                "person_name": e.person_name,
                "person_role": e.person_role,
                "global_id":   e.global_id,
                "status":      e.status,
                "confidence":  e.confidence,
                "message":     e.message,
            }
            for e in events
        ]
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "healthy", "service": "sop_state_machine"}


# ── Server launcher (called from main.py) ──────────────────────────────────────

def start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start uvicorn in a daemon thread. Returns immediately."""
    t = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": host, "port": port, "log_level": "warning"},
        daemon=True,
        name="fastapi",
    )
    t.start()
    return t
