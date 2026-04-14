"""
enrollment_api.py
Edge AI Security System — REST API for Face Enrollment

Launch: uvicorn enrollment_api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import sqlite3
import numpy as np
import cv2
import base64
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import insightface
from insightface.app import FaceAnalysis

# ── Ensure shared module is on the path ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ROOT = os.path.join(
    os.path.dirname(BASE_DIR), "edge_ai_security", "models", "insightface"
)
DB_PATH = os.path.join(BASE_DIR, "sop_state_machine", "security.db")
FACES_DIR = os.path.join(BASE_DIR, "data", "enrolled_faces")
os.makedirs(FACES_DIR, exist_ok=True)

# Define system roles matching the runtime dictionary
ROLES = ["Manager", "Staff", "Cashier", "Security", "Cleaner", "Customer", "Visitor", "Unknown"]

app = FastAPI(
    title="Edge AI Enrollment API",
    description="REST API for managing face enrollments and security zones.",
    version="1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── AI Model Initialization ───────────────────────────────────────────────────
print("Loading InsightFace models…")
fa = FaceAnalysis(
    name="buffalo_sc",
    root=MODEL_ROOT,
    providers=["CPUExecutionProvider"],
)
fa.prepare(ctx_id=-1, det_thresh=0.4, det_size=(640, 640))
print("Models ready.\n")


# ── DB Helpers ────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_tables():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            role         TEXT NOT NULL,
            embedding    TEXT NOT NULL,
            face_image   TEXT,
            appearances  INTEGER DEFAULT 1,
            cameras_seen TEXT DEFAULT '[]',
            enrolled_at  TEXT DEFAULT (datetime('now')),
            active       INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS zones (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL,
            cam_id           TEXT NOT NULL,
            polygon_points   TEXT DEFAULT '[]',
            restricted_roles TEXT DEFAULT '[]',
            created_at       TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS zone_access (
            person_id  INTEGER,
            zone_id    INTEGER,
            granted_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (person_id, zone_id)
        );
    """)
    conn.commit()
    conn.close()

ensure_tables()


# ── Face Extraction Logic ─────────────────────────────────────────────────────
def extract_embedding(img_bgr):
    """Return (embedding, face_crop_bgr, message)."""
    if img_bgr is None:
        return None, None, "No image provided."
        
    faces = fa.get(img_bgr)
    if not faces:
        return None, None, "No face detected. Try a clearer / closer photo."

    # Identify the largest face in the frame
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
    face = faces[0]
    emb = face.normed_embedding

    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = img_bgr.shape[:2]
    pad = 20
    crop = img_bgr[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)].copy()
    
    return emb, crop, f"Face detected with confidence {face.det_score:.2f}"


# ── Pydantic Models ───────────────────────────────────────────────────────────
class PersonResponse(BaseModel):
    id: int
    name: str
    role: str
    enrolled_at: str
    zones: List[str]

class ZoneResponse(BaseModel):
    id: int
    name: str
    cam_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/roles", tags=["Configuration"])
def get_roles():
    """Retrieve the standard list of system authorization roles."""
    return {"roles": ROLES}


@app.get("/api/zones", response_model=List[ZoneResponse], tags=["Configuration"])
def list_zones():
    """Retrieve the list of configured zones from the security database."""
    conn = get_db()
    rows = conn.execute("SELECT id, name, cam_id FROM zones ORDER BY id").fetchall()
    conn.close()
    return [{"id": r["id"], "name": r["name"], "cam_id": r["cam_id"]} for r in rows]


@app.get("/api/persons", response_model=List[PersonResponse], tags=["Management"])
def list_persons():
    """Retrieve all active enrolled persons."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, role, enrolled_at FROM persons WHERE active=1 ORDER BY id DESC"
    ).fetchall()
    
    # Batch grab access zones for speed
    z_rows = conn.execute(
        "SELECT za.person_id, z.name FROM zone_access za JOIN zones z ON za.zone_id = z.id"
    ).fetchall()
    conn.close()
    
    zones_map = {}
    for zr in z_rows:
        pid = zr["person_id"]
        if pid not in zones_map:
            zones_map[pid] = []
        if zr["name"] not in zones_map[pid]:
            zones_map[pid].append(zr["name"])

    persons = []
    for r in rows:
        pid = r["id"]
        persons.append(PersonResponse(
            id=pid,
            name=r["name"],
            role=r["role"],
            enrolled_at=r["enrolled_at"],
            zones=zones_map.get(pid, [])
        ))
    return persons


@app.post("/api/enroll", tags=["Management"])
async def enroll_person(
    name: str = Form(...),
    role: str = Form(...),
    zone_ids: str = Form("[]"),  # Expected to be a JSON string like "[1, 2, 3]"
    file: UploadFile = File(...)
):
    """
    Enroll a new person using a provided frontal image.
    Extracts the face mathematically and persists it into the identity datastore.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
        
    if role not in ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of {ROLES}.")
        
    try:
        zones_list = json.loads(zone_ids)
        if not isinstance(zones_list, list):
            zones_list = []
    except json.JSONDecodeError:
        zones_list = []

    # Read image into OpenCV format
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    # AI Face Extraction
    emb, crop_bgr, msg = extract_embedding(img_bgr)
    if emb is None:
        raise HTTPException(status_code=422, detail=msg)

    # Save visual crop locally (similar to Gradio's logic)
    tag = f"{name.replace(' ', '_')}_{int(datetime.now().timestamp())}"
    face_path = os.path.join(FACES_DIR, f"{tag}.jpg")
    cv2.imwrite(face_path, crop_bgr)
    
    emb_json = json.dumps(emb.tolist())

    # Write to Database
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO persons (name, role, embedding, face_image, enrolled_at, active) "
        "VALUES (?, ?, ?, ?, datetime('now'), 1)",
        (name, role, emb_json, face_path)
    )
    person_id = cur.lastrowid
    
    for zid in zones_list:
        # Ignore bad zone ids safely via foreign/primary constraint logic
        conn.execute(
            "INSERT OR IGNORE INTO zone_access (person_id, zone_id) VALUES (?, ?)",
            (person_id, int(zid))
        )
        
    conn.commit()
    conn.close()

    return JSONResponse(content={
        "status": "success",
        "person_id": person_id,
        "message": f"Successfully enrolled {name} as {role}. {msg}"
    })


@app.delete("/api/persons/{person_id}", tags=["Management"])
def delete_person(person_id: int):
    """
    Deactivate a person from the identity datastore. The AI engine will ignore deactivated profiles.
    """
    conn = get_db()
    cur = conn.execute("UPDATE persons SET active=0 WHERE id=?", (person_id,))
    conn.commit()
    rows_affected = cur.rowcount
    conn.close()
    
    if rows_affected == 0:
        raise HTTPException(status_code=404, detail="Person not found.")
        
    return {"status": "success", "message": f"Person ID {person_id} deactivated."}
