"""
sop_state_machine/app/main.py
Service 4 of the Edge AI Security Pipeline (Standalone / Local Dev Version).

Subscribes to DetectionResult from ai_inference (port 5552).
For each detected person per frame:
  1. Checks zone assignment via polygon containment (from enrollment.db)
  2. Evaluates authorization: AUTHORIZED / WRONG_ZONE / RESTRICTED / UNKNOWN
  3. Publishes ViolationEvent on port 5553 for the visualizer

Run with:
  cd /path/to/video_ingestion_standalone
  export PYTHONPATH=$PYTHONPATH:.
  python3 sop_state_machine/app/main.py
"""

import base64
import collections
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import psutil
import zmq

from shared.config import get_settings
settings = get_settings()

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from shared.message_schema import DetectionResult, Heartbeat, decode, encode
from shared.zmq_topics import DETECTION_RESULT, VIOLATION_EVENT, HEARTBEAT

# ── Logging ─────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sop_state_machine")

# ── Config ───────────────────────────────────────────────────────────────────

ZMQ_DETECTIONS_PORT = settings.zmq.inference # actually 5552
ZMQ_VIOLATIONS_PORT = settings.zmq.sop       # actually 5553
ZMQ_HEALTH_PORT     = settings.zmq.health_sop
ZMQ_HOST            = os.getenv("ZMQ_HOST", "localhost")
SIMILARITY_THRESH   = float(os.getenv("SIMILARITY_THRESH", "0.38"))
DWELL_THRESHOLD_SEC = float(os.getenv("DWELL_THRESHOLD_SEC", "5.0"))  # secs before a transit becomes a violation

THIS_DIR       = os.path.dirname(os.path.abspath(__file__))  # .../sop_state_machine/app
SOP_DIR        = os.path.dirname(THIS_DIR)                   # .../sop_state_machine
STANDALONE_DIR = os.path.dirname(SOP_DIR)                    # .../video_ingestion_standalone
PARENT_DIR     = os.path.dirname(STANDALONE_DIR)             # /home/sana/Bank project

DEFAULT_DB = os.path.join(PARENT_DIR, "edge_ai_security", "data", "enrollment.db")
ENROLLMENT_DB = os.getenv("DB_PATH", DEFAULT_DB)

# Security DB (stores zones and events) — created locally
SECURITY_DB_PATH = os.getenv(
    "SECURITY_DB",
    os.path.join(STANDALONE_DIR, "sop_state_machine", "security.db"),
)

stop_event = threading.Event()

# ── Cross-Camera Deduplicator ─────────────────────────────────────────────────
#
# When the same person is detected on multiple cameras within DEDUP_WINDOW_SEC,
# only the detection with the highest face-confidence score is published.
# Unknown persons are bucketed by a short position hash so separate unknowns
# in different zones are still treated as independent events.

DEDUP_WINDOW_SEC = 2.0  # seconds to wait before flushing

class CrossCameraDeduplicator:
    """
    Buffers (violation_dict, pub_sock) tuples per person.
    Every DEDUP_WINDOW_SEC a flush thread picks the best candidate
    (highest conf_face) and publishes it.
    """

    def __init__(self):
        # key: person_name  →  list of {viol, conf, ts}
        self._buf: dict[str, list] = collections.defaultdict(list)
        self._lock = threading.Lock()

    def add(self, person_name: str, conf: float, viol: dict, pub_sock):
        with self._lock:
            self._buf[person_name].append({
                "conf":     conf,
                "viol":     viol,
                "pub_sock": pub_sock,
                "ts":       time.time(),
            })

    def flush(self):
        """Called periodically — emit the best detection per person."""
        now = time.time()
        with self._lock:
            to_flush = {}
            remaining = collections.defaultdict(list)
            for name, entries in self._buf.items():
                expired  = [e for e in entries if now - e["ts"] >= DEDUP_WINDOW_SEC]
                fresh    = [e for e in entries if now - e["ts"] <  DEDUP_WINDOW_SEC]
                if expired:
                    to_flush[name] = expired
                remaining[name] = fresh
            self._buf = remaining

        for name, entries in to_flush.items():
            best = max(entries, key=lambda e: e["conf"])
            v    = best["viol"]
            sock = best["pub_sock"]
            # Log which camera won
            if len(entries) > 1:
                cams = [e["viol"]["cam_id"] for e in entries]
                log.debug(
                    f"[dedup] {name}: {len(entries)} cams {cams} → "
                    f"best={v['cam_id']} (conf={best['conf']:.3f})"
                )
            sock.send_multipart([VIOLATION_EVENT, json.dumps(v).encode("utf-8")])


_deduplicator = CrossCameraDeduplicator()


def _flush_loop():
    """Background thread: flush deduplicator every DEDUP_WINDOW_SEC."""
    while not stop_event.is_set():
        time.sleep(DEDUP_WINDOW_SEC)
        try:
            _deduplicator.flush()
        except Exception as e:
            log.error(f"Deduplicator flush error: {e}")


# ── Dwell Tracker (Option 1) ──────────────────────────────────────────────────
#
# A person passing THROUGH a restricted zone should not trigger a violation.
# Only fire if they STAY in the zone for >= DWELL_THRESHOLD_SEC seconds.

class DwellTracker:
    """
    Tracks how long each (person, zone) pair has been present.
    key = (person_name, zone_name)
    value = {first_seen, last_seen, fired}
    """
    def __init__(self):
        self._zones: dict = {}   # key → {first_seen, last_seen, fired}
        self._lock = threading.Lock()

    def update(self, key: tuple, now: float) -> bool:
        """
        Call every time the person is seen in this zone.
        Returns True when the violation should fire for the first time.
        """
        with self._lock:
            if key not in self._zones:
                self._zones[key] = {"first_seen": now, "last_seen": now, "fired": False}
                return False   # just entered — don't fire yet
            entry = self._zones[key]
            entry["last_seen"] = now
            if entry["fired"]:
                return False   # already fired once, don't spam
            if now - entry["first_seen"] >= DWELL_THRESHOLD_SEC:
                entry["fired"] = True
                return True    # dwell threshold crossed → fire!
            return False

    def cleanup(self, now: float, timeout: float = 15.0):
        """Remove stale entries (person hasn't been seen for >timeout seconds)."""
        with self._lock:
            stale = [k for k, v in self._zones.items() if now - v["last_seen"] > timeout]
            for k in stale:
                del self._zones[k]


_dwell = DwellTracker()


def _dwell_cleanup_loop():
    """Background thread: prune stale dwell entries every 30 seconds."""
    while not stop_event.is_set():
        time.sleep(30)
        _dwell.cleanup(time.time())


def shutdown(sig, frame):
    log.info("Shutdown signal received")
    stop_event.set()


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ── Security DB setup ────────────────────────────────────────────────────────

def init_security_db():
    """Create zones and events tables. Migrate existing DBs for new columns."""
    conn = sqlite3.connect(SECURITY_DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS zones (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL,
            cam_id           TEXT NOT NULL,
            polygon_points   TEXT NOT NULL,
            restricted_roles TEXT DEFAULT '[]',
            is_corridor      INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            cam_id      TEXT,
            zone_name   TEXT,
            person_id   INTEGER,
            person_name TEXT,
            person_role TEXT,
            status      TEXT,
            confidence  REAL DEFAULT 0.0,
            message     TEXT
        );
    """)
    # Migration: add is_corridor to existing tables that lack it
    existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(zones)").fetchall()]
    if "is_corridor" not in existing_cols:
        conn.execute("ALTER TABLE zones ADD COLUMN is_corridor INTEGER DEFAULT 0")
        conn.commit()
        log.info("Migrated zones table: added is_corridor column")
    conn.commit()
    conn.close()
    log.info(f"Security DB initialised at {SECURITY_DB_PATH}")


def get_all_zones(frame_w: int = 2880, frame_h: int = 1620):
    """Return all zones regardless of camera. Includes is_corridor flag."""
    conn = sqlite3.connect(SECURITY_DB_PATH)
    rows = conn.execute(
        "SELECT id, name, polygon_points, restricted_roles, is_corridor FROM zones"
    ).fetchall()
    if not rows:
        conn.execute(
            "INSERT INTO zones (name, cam_id, polygon_points, restricted_roles, is_corridor) VALUES (?,?,?,?,?)",
            ("Default Zone", "any",
             json.dumps([[0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]]),
             json.dumps([]), 0)
        )
        conn.commit()
        rows = conn.execute(
            "SELECT id, name, polygon_points, restricted_roles, is_corridor FROM zones"
        ).fetchall()
        log.info("Seeded default universal zone.")
    conn.close()
    return rows

def find_zone(bbox: list, frame_w=2880, frame_h=1620):
    """Find which zone a person's feet are in. Returns zone dict with is_corridor flag."""
    if not bbox or len(bbox) < 4:
        return None, "Unknown Zone"
    x1, y1, x2, y2 = bbox
    feet_x = float((x1 + x2) / 2)
    feet_y = float(y2)

    import cv2, ast
    zones = get_all_zones(frame_w, frame_h)
    for zid, zname, zpts_json, restricted_json, is_corridor in zones:
        try:
            pts = json.loads(zpts_json)
        except (json.JSONDecodeError, ValueError):
            pts = ast.literal_eval(zpts_json)
        if len(pts) < 3:
            continue
        polygon = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))
        result = cv2.pointPolygonTest(polygon, (feet_x, feet_y), False)
        if result >= 0:
            try:
                restricted = json.loads(restricted_json)
            except (json.JSONDecodeError, ValueError):
                restricted = ast.literal_eval(restricted_json) if restricted_json else []
            return {
                "id": zid, "name": zname,
                "restricted_roles": restricted,
                "is_corridor": bool(is_corridor),
            }, zname
    # Fallback → Common Area
    for zid, zname, zpts_json, restricted_json, is_corridor in zones:
        if "common area" in zname.lower():
            try:
                restricted = json.loads(restricted_json)
            except Exception:
                restricted = []
            return {"id": zid, "name": zname, "restricted_roles": restricted, "is_corridor": bool(is_corridor)}, zname
    return {"id": -1, "name": "Common Area", "restricted_roles": [], "is_corridor": False}, "Common Area"


def log_event(cam_id, zone_name, person_id, person_name, person_role, status, confidence, message):
    conn = sqlite3.connect(SECURITY_DB_PATH)
    conn.execute(
        """INSERT INTO events 
           (timestamp, cam_id, zone_name, person_id, person_name, person_role, status, confidence, message)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(), cam_id, zone_name,
         person_id, person_name, person_role, status, confidence, message)
    )
    conn.commit()
    conn.close()


# ── Enrollment DB persons ────────────────────────────────────────────────────

def load_persons():
    """Load enrolled persons from enrollment.db."""
    if not os.path.exists(ENROLLMENT_DB):
        log.warning(f"Enrollment DB not found at {ENROLLMENT_DB}")
        return []
    conn = sqlite3.connect(ENROLLMENT_DB)
    try:
        rows = conn.execute(
            "SELECT id, name, role FROM persons WHERE name != 'REJECTED'"
        ).fetchall()
        persons = [{"id": r[0], "name": r[1], "role": r[2]} for r in rows]
        log.info(f"Loaded {len(persons)} enrolled persons")
        return persons
    except Exception as e:
        log.error(f"Error loading persons: {e}")
        return []
    finally:
        conn.close()


# ── Heartbeat loop ───────────────────────────────────────────────────────────

def heartbeat_loop(health_socket, start_time):
    proc = psutil.Process()
    while not stop_event.is_set():
        hb = Heartbeat(
            service_name = "sop_state_machine",
            status       = "healthy",
            cpu_percent  = psutil.cpu_percent(),
            mem_mb       = proc.memory_info().rss / 1024 / 1024,
            uptime_sec   = time.time() - start_time,
            timestamp    = time.time(),
        )
        try:
            health_socket.send_multipart([HEARTBEAT, encode(hb)])
        except:
            pass
        stop_event.wait(5)


# ── Main loop ────────────────────────────────────────────────────────────────

def process_detection(result: DetectionResult, pub_sock: zmq.Socket, frame: np.ndarray = None):
    """Evaluate each detected person and publish violation events."""
    cam_id = result.cam_id
    
    # Resolve cam_label from settings if possible
    cam_label = result.cam_label
    if not cam_label or cam_label == "Unknown":
        for c in settings.cameras:
            if c.cam_id == cam_id:
                cam_label = c.label
                break

    for d in result.detections:
        # Detection dict fields
        if isinstance(d, dict):
            person_id   = d.get("id", -1)
            person_name = d.get("name", "Unknown")
            person_role = d.get("role", "Unknown")
            bbox        = d.get("bbox", [])
            conf_face   = d.get("conf_face", 0.0)
        else:
            person_id   = d.id
            person_name = d.name
            person_role = d.role
            bbox        = d.bbox
            conf_face   = d.conf_face

        # Zone lookup
        zone, zone_name = find_zone(bbox)
        fx, fy = (bbox[0] + bbox[2]) / 2, bbox[3]

        # ── Option 2: Corridor zones are always AUTHORIZED ─────────────────
        is_corridor = zone.get("is_corridor", False) if zone else False

        # Authorization logic
        if is_corridor:
            status  = "AUTHORIZED"
            message = f"{person_name} transiting through corridor '{zone_name}'"
        elif person_name == "Unknown" or person_id == -1:
            if "common area" in zone_name.lower():
                status  = "AUTHORIZED"
                message = f"Unrecognised person in permitted Common Area"
            else:
                status  = "UNKNOWN"
                message = f"Unrecognised person in {zone_name} (pos={int(bbox[0])},{int(bbox[1])})"
        elif zone is None:
            status  = "AUTHORIZED"
            message = f"{person_name} in unrestricted area"
        elif person_role.casefold() in [r.casefold() for r in zone.get("restricted_roles", [])]:
            status  = "RESTRICTED"
            message = f"Role {person_role} restricted from {zone_name}"
        else:
            status  = "AUTHORIZED"
            message = f"{person_name} ({person_role}) authorized in {zone_name}"

        # ── Option 1: Dwell time gate (non-AUTHORIZED only) ─────────────────
        # Don't fire immediately — wait until person has been in zone for DWELL_THRESHOLD_SEC
        if status != "AUTHORIZED":
            dwell_key = (person_name, zone_name)
            now_dwell = time.time()
            should_fire = _dwell.update(dwell_key, now_dwell)
            if not should_fire:
                # Still within grace period — log at DEBUG only, skip violation
                log.debug(
                    f"[dwell] {person_name} in {zone_name} ({cam_id}) — "
                    f"waiting for {DWELL_THRESHOLD_SEC}s threshold"
                )
                continue   # skip the rest of the loop body for this detection

        # Log
        severity_map = {
            "AUTHORIZED": logging.INFO,
            "UNKNOWN":    logging.WARNING,
            "RESTRICTED": logging.WARNING,
        }
        log.log(severity_map.get(status, logging.INFO),
                f"{status} | {person_name} | zone={zone_name} | cam={cam_id} | pos={int(fx)},{int(fy)}")

        log_event(cam_id, zone_name, person_id if person_id != -1 else None,
                  person_name, person_role, status, conf_face, message)

        # Evidence crop if violation
        evidence_b64 = ""
        if status != "AUTHORIZED" and frame is not None and bbox:
            try:
                x1, y1, x2, y2 = map(int, bbox)
                # Ensure within bounds
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    _, buffer = cv2.imencode(".jpg", crop)
                    evidence_b64 = base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                log.error(f"Error cropping evidence: {e}")

        # Build violation payload
        viol = {
            "cam_id":        cam_id,
            "cam_label":     cam_label,
            "zone_name":     zone_name,
            "person_name":   person_name,
            "person_role":   person_role,
            "person_id":     person_id,
            "status":        status,
            "message":       message,
            "bbox":          bbox,
            "timestamp":     result.timestamp,
            "evidence_jpeg": evidence_b64,
        }

        # ── Cross-camera deduplication ────────────────────────────────────
        # AUTHORIZED events pass straight through (no spam risk).
        # UNKNOWN/RESTRICTED go through the deduplicator so the same person
        # seen on N cameras within DEDUP_WINDOW_SEC only fires once (best conf).
        if status == "AUTHORIZED":
            pub_sock.send_multipart([VIOLATION_EVENT, json.dumps(viol).encode("utf-8")])
        else:
            # Use person_name for known persons; for unknowns bucket by zone
            # so separate unknowns in different zones are independent.
            dedup_key = person_name if person_name != "Unknown" else f"Unknown_{zone_name}"
            _deduplicator.add(dedup_key, conf_face, viol, pub_sock)


def main():
    os.makedirs(os.path.dirname(SECURITY_DB_PATH), exist_ok=True)
    init_security_db()

    ctx = zmq.Context()

    # Subscribe to DetectionResult from ai_inference
    sub_sock = ctx.socket(zmq.SUB)
    sub_sock.connect(f"tcp://{ZMQ_HOST}:{ZMQ_DETECTIONS_PORT}")
    sub_sock.setsockopt(zmq.SUBSCRIBE, DETECTION_RESULT)
    sub_sock.setsockopt(zmq.RCVTIMEO, 1000)
    log.info(f"Subscribed to detections on {ZMQ_HOST}:{ZMQ_DETECTIONS_PORT}")

    # Publish ViolationEvent for the visualizer
    pub_sock = ctx.socket(zmq.PUB)
    pub_sock.bind(f"tcp://0.0.0.0:{ZMQ_VIOLATIONS_PORT}")
    log.info(f"Publishing violations on port {ZMQ_VIOLATIONS_PORT}")

    # Health heartbeat
    health_sock = ctx.socket(zmq.PUB)
    health_sock.bind(f"tcp://0.0.0.0:{ZMQ_HEALTH_PORT}")
    threading.Thread(target=heartbeat_loop, args=(health_sock, time.time()),
                     daemon=True, name="heartbeat").start()

    # Start cross-camera deduplication flush loop
    threading.Thread(target=_flush_loop, daemon=True, name="dedup_flush").start()
    log.info(f"Cross-camera deduplicator active (window={DEDUP_WINDOW_SEC}s)")

    # Start dwell tracker cleanup loop
    threading.Thread(target=_dwell_cleanup_loop, daemon=True, name="dwell_cleanup").start()
    log.info(f"Dwell tracker active (threshold={DWELL_THRESHOLD_SEC}s)")


    log.info("SOP State Machine ready. Waiting for detections...")

    while not stop_event.is_set():
        try:
            parts = sub_sock.recv_multipart()
            if len(parts) < 2: continue
            
            topic = parts[0]
            payload = parts[1]
            frame_bytes = parts[2] if len(parts) > 2 else None
            
            frame_bgr = None
            if frame_bytes:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            result = decode(payload, DetectionResult)
            process_detection(result, pub_sock, frame_bgr)
        except zmq.Again:
            continue
        except Exception as e:
            log.error(f"Error: {e}")
            time.sleep(0.1)

    sub_sock.close()
    pub_sock.close()
    ctx.term()
    log.info("SOP State Machine stopped.")


if __name__ == "__main__":
    main()
