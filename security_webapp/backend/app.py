"""
Security Video Analysis Web Application — FastAPI Backend

This server accepts CCTV video uploads, runs them through the Edge AI
security pipeline, and serves back the generated reports + evidence.

Launch:
    cd /home/sana/Bank\ project/security_webapp
    python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 9000
"""

import os
import sys
import uuid
import json
import time
import shutil
import zipfile
import sqlite3
import subprocess
import threading
from pathlib import Path
from datetime import datetime

from typing import List, Optional
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR       = Path(__file__).resolve().parent            # backend/
PROJECT_DIR   = APP_DIR.parent                             # security_webapp/
JOBS_DIR      = PROJECT_DIR / "jobs"
FRONTEND_DIR  = PROJECT_DIR / "frontend"
PIPELINE_DIR  = Path("/home/sana/Bank project/video_ingestion_standalone")
EDGE_AI_DIR   = Path("/home/sana/Bank project/edge_ai_security")
SECURITY_DB   = PIPELINE_DIR / "sop_state_machine" / "security.db"
MODEL_ROOT    = EDGE_AI_DIR / "models" / "insightface"
FACES_DIR     = PIPELINE_DIR / "data" / "enrolled_faces"

JOBS_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)

# ── AI Model Initialization ───────────────────────────────────────────────────
print("Loading InsightFace models…")
try:
    fa = FaceAnalysis(
        name="buffalo_sc",
        root=str(MODEL_ROOT),
        providers=["CPUExecutionProvider"],
    )
    fa.prepare(ctx_id=-1, det_thresh=0.4, det_size=(640, 640))
    print("Models ready.\n")
except Exception as e:
    print(f"Warning: Failed to load InsightFace models: {e}")
    fa = None

ROLES = ["Manager", "Staff", "Cashier", "Security", "Cleaner", "Customer", "Visitor", "Unknown"]

# ── DB Helpers ────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(SECURITY_DB), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

# ── Face Extraction Logic ─────────────────────────────────────────────────────
def extract_embedding(img_bgr):
    """Return (embedding, face_crop_bgr, message)."""
    if fa is None:
        return None, None, "InsightFace models are not loaded."
    if img_bgr is None:
        return None, None, "No image provided."
        
    faces = fa.get(img_bgr)
    if not faces:
        return None, None, "No face detected. Try a clearer / closer photo."

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

# ── In-memory job tracker ─────────────────────────────────────────────────────
jobs = {}   # job_id -> { status, message, created_at, video_name, ... }

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Edge AI Security — Video Analysis",
    description="Upload CCTV footage, get AI-powered security reports.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_pipeline(job_id: str, video_paths: list):
    """
    Execute the full Edge AI pipeline against one or more video files.
    Each video is mapped to a separate camera (cam_0, cam_1, …).
    This runs in a background thread.
    """
    job = jobs[job_id]
    job_dir = JOBS_DIR / job_id
    evidence_dir = job_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)

    try:
        job["status"] = "processing"
        job["message"] = f"Resetting pipeline state… ({len(video_paths)} videos)"

        # 1. Run system_reset.py to clear old data
        subprocess.run(
            [sys.executable, "system_reset.py"],
            cwd=str(PIPELINE_DIR),
            env={**os.environ, "PYTHONPATH": str(PIPELINE_DIR)},
            capture_output=True, text=True, timeout=30
        )

        # 2. Create a temporary .env override to point at the uploaded videos
        env_path = PIPELINE_DIR / ".env"
        env_backup = PIPELINE_DIR / ".env.bak"
        shutil.copy(str(env_path), str(env_backup))

        # Read existing .env and override CAM sources
        with open(str(env_backup), "r") as f:
            env_lines = f.readlines()

        new_lines = []
        for line in env_lines:
            stripped = line.strip()
            # Remove existing CAM_*_SOURCE and CAM_*_LABEL lines
            if stripped.startswith("CAM_") and ("_SOURCE=" in stripped or "_LABEL=" in stripped):
                continue
            new_lines.append(line)

        # Add each uploaded video as a separate camera
        new_lines.append("\n")
        for i, vpath in enumerate(video_paths):
            new_lines.append(f"CAM_{i}_SOURCE={vpath}\n")
            new_lines.append(f"CAM_{i}_LABEL=Cam_{i}\n")

        with open(str(env_path), "w") as f:
            f.writelines(new_lines)

        job["message"] = "Running AI security analysis…"

        # 3. Run the pipeline
        pipeline_env = {
            **os.environ,
            "PYTHONPATH": f"{PIPELINE_DIR}:{os.environ.get('PYTHONPATH', '')}",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }

        proc = subprocess.run(
            [sys.executable, "run_all.py"],
            cwd=str(PIPELINE_DIR),
            env=pipeline_env,
            capture_output=True, text=True,
            timeout=1800  # 30 minute max
        )

        # 4. Restore the original .env
        shutil.copy(str(env_backup), str(env_path))
        env_backup.unlink(missing_ok=True)

        # 5. Collect evidence files
        pipeline_evidence = PIPELINE_DIR / "data" / "evidence"
        if pipeline_evidence.exists():
            for f in pipeline_evidence.iterdir():
                if f.suffix in (".jpg", ".jpeg", ".png"):
                    shutil.copy(str(f), str(evidence_dir / f.name))

        # 6. Collect reports
        pipeline_reports = PIPELINE_DIR / "data" / "reports"
        if pipeline_reports.exists():
            for f in pipeline_reports.iterdir():
                if f.suffix == ".html":
                    shutil.copy(str(f), str(job_dir / "report.html"))
                    break  # Copy only the latest

        # 7. Store pipeline stdout/stderr for debugging
        with open(str(job_dir / "pipeline_log.txt"), "w") as f:
            f.write("=== STDOUT ===\n")
            f.write(proc.stdout or "(empty)")
            f.write("\n\n=== STDERR ===\n")
            f.write(proc.stderr or "(empty)")

        # Check for report
        report_path = job_dir / "report.html"
        evidence_count = len(list(evidence_dir.glob("*.jpg")))

        if report_path.exists():
            job["status"] = "done"
            job["message"] = f"Analysis complete. {evidence_count} evidence files captured."
        else:
            job["status"] = "done"
            job["message"] = f"Pipeline finished. {evidence_count} evidence files captured. (No HTML report generated)"

    except subprocess.TimeoutExpired:
        job["status"] = "error"
        job["message"] = "Pipeline timed out after 30 minutes."
        # Restore .env
        env_backup = PIPELINE_DIR / ".env.bak"
        if env_backup.exists():
            shutil.copy(str(env_backup), str(PIPELINE_DIR / ".env"))
            env_backup.unlink(missing_ok=True)

    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Pipeline error: {str(e)}"
        # Restore .env
        env_backup = PIPELINE_DIR / ".env.bak"
        if env_backup.exists():
            shutil.copy(str(env_backup), str(PIPELINE_DIR / ".env"))
            env_backup.unlink(missing_ok=True)


# ── Helper Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/roles", tags=["Configuration"])
def get_roles():
    return {"roles": ROLES}

@app.get("/api/zones", response_model=List[ZoneResponse], tags=["Configuration"])
def list_zones():
    conn = get_db()
    rows = conn.execute("SELECT id, name, cam_id FROM zones ORDER BY id").fetchall()
    conn.close()
    return [{"id": r["id"], "name": r["name"], "cam_id": r["cam_id"]} for r in rows]

@app.get("/api/persons", response_model=List[PersonResponse], tags=["Management"])
def list_persons():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, role, enrolled_at FROM persons WHERE active=1 ORDER BY id DESC"
    ).fetchall()
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
            enrolled_at=r["enrolled_at"] or "",
            zones=zones_map.get(pid, [])
        ))
    return persons

@app.post("/api/enroll", tags=["Management"])
async def enroll_person(
    name: str = Form(...),
    role: str = Form(...),
    zone_ids: str = Form("[]"),
    file: UploadFile = File(...)
):
    name = name.strip()
    if not name or role not in ROLES:
        raise HTTPException(status_code=400, detail="Invalid name or role.")
        
    try:
        zones_list = json.loads(zone_ids)
        if not isinstance(zones_list, list):
            zones_list = []
    except:
        zones_list = []

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    emb, crop_bgr, msg = extract_embedding(img_bgr)
    if emb is None:
        raise HTTPException(status_code=422, detail=msg)

    tag = f"{name.replace(' ', '_')}_{int(datetime.now().timestamp())}"
    face_path = FACES_DIR / f"{tag}.jpg"
    cv2.imwrite(str(face_path), crop_bgr)
    
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO persons (name, role, embedding, face_image, enrolled_at, active) VALUES (?, ?, ?, ?, datetime('now'), 1)",
        (name, role, json.dumps(emb.tolist()), str(face_path))
    )
    person_id = cur.lastrowid
    for zid in zones_list:
        conn.execute("INSERT OR IGNORE INTO zone_access (person_id, zone_id) VALUES (?, ?)", (person_id, int(zid)))
    conn.commit()
    conn.close()

    return JSONResponse(content={"status": "success", "person_id": person_id, "message": f"Successfully enrolled {name}. {msg}"})

@app.delete("/api/persons/{person_id}", tags=["Management"])
def delete_person(person_id: int):
    conn = get_db()
    cur = conn.execute("UPDATE persons SET active=0 WHERE id=?", (person_id,))
    conn.commit()
    rows_affected = cur.rowcount
    conn.close()
    if rows_affected == 0:
        raise HTTPException(status_code=404, detail="Person not found.")
    return {"status": "success", "message": "Person deactivated."}


# ── Web App Endpoints ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the main frontend page."""
    index = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=index.read_text(), status_code=200)


@app.post("/api/upload")
async def upload_video(files: List[UploadFile] = File(...)):
    """
    Accept one or more video file uploads and start background processing.
    Each file is mapped to a separate camera feed (cam_0, cam_1, …).
    Returns a job_id for status polling.
    """
    allowed = (".avi", ".mp4", ".mkv", ".mov", ".wmv", ".flv")

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_paths = []
    total_size = 0
    filenames = []

    for i, file in enumerate(files):
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed:
            raise HTTPException(400, f"Unsupported format '{ext}' for {file.filename}. Accepted: {', '.join(allowed)}")

        video_path = job_dir / f"cam_{i}{ext}"
        content = await file.read()
        with open(str(video_path), "wb") as f:
            f.write(content)

        video_paths.append(video_path)
        total_size += len(content)
        filenames.append(file.filename)

    total_mb = total_size / (1024 * 1024)

    # Register job
    jobs[job_id] = {
        "status": "queued",
        "message": f"{len(files)} video(s) uploaded. Waiting to start…",
        "created_at": datetime.now().isoformat(),
        "video_name": ", ".join(filenames),
        "video_count": len(files),
        "file_size_mb": round(total_mb, 2),
    }

    # Start processing in background
    t = threading.Thread(target=run_pipeline, args=(job_id, video_paths), daemon=True)
    t.start()

    return {"job_id": job_id, "video_count": len(files), "message": f"Upload successful ({total_mb:.1f} MB, {len(files)} videos). Processing started."}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    """Check the processing status of a job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    return jobs[job_id]


@app.get("/api/results/{job_id}")
def get_results(job_id: str):
    """Get the analysis results including evidence file list."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")

    job = jobs[job_id]
    if job["status"] not in ("done", "error"):
        return {"status": job["status"], "message": "Still processing…"}

    job_dir = JOBS_DIR / job_id
    evidence_dir = job_dir / "evidence"

    evidence_files = []
    if evidence_dir.exists():
        for f in sorted(evidence_dir.iterdir()):
            if f.suffix in (".jpg", ".jpeg", ".png"):
                evidence_files.append(f.name)

    has_report = (job_dir / "report.html").exists()

    return {
        "status": job["status"],
        "message": job["message"],
        "video_name": job.get("video_name", ""),
        "has_report": has_report,
        "evidence_count": len(evidence_files),
        "evidence_files": evidence_files,
    }


@app.get("/api/evidence/{job_id}/{filename}")
def get_evidence_image(job_id: str, filename: str):
    """Serve an individual evidence image."""
    file_path = JOBS_DIR / job_id / "evidence" / filename
    if not file_path.exists():
        raise HTTPException(404, "Evidence file not found.")
    return FileResponse(str(file_path), media_type="image/jpeg")


@app.get("/api/report/{job_id}")
def get_report(job_id: str):
    """Serve the HTML security report."""
    report_path = JOBS_DIR / job_id / "report.html"
    if not report_path.exists():
        raise HTTPException(404, "Report not found.")
    return FileResponse(str(report_path), media_type="text/html")


@app.get("/api/download/{job_id}")
def download_zip(job_id: str):
    """Download the complete results as a ZIP archive."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")

    job_dir = JOBS_DIR / job_id
    zip_path = job_dir / "results.zip"

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        # Add report
        report = job_dir / "report.html"
        if report.exists():
            zf.write(str(report), "report.html")

        # Add evidence
        evidence_dir = job_dir / "evidence"
        if evidence_dir.exists():
            for f in evidence_dir.iterdir():
                if f.suffix in (".jpg", ".jpeg", ".png"):
                    zf.write(str(f), f"evidence/{f.name}")

        # Add pipeline log
        log = job_dir / "pipeline_log.txt"
        if log.exists():
            zf.write(str(log), "pipeline_log.txt")

    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"security_report_{job_id}.zip"
    )
