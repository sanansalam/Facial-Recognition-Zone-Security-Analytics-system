"""
sop_state_machine — Service 4
Subscribes to DetectionResult from ai_inference.
For each detected person per frame:
  1. Identifies them via face embedding (enrolled DB or Unknown)
  2. Finds which polygon zone their feet are in
  3. Evaluates authorization: AUTHORIZED / WRONG_ZONE / RESTRICTED / UNKNOWN / MASKED
  4. Tracks them across cameras using ArcFace embeddings
  5. Publishes ViolationEvent if not AUTHORIZED
  6. Serves REST API on port 8000
"""

import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime

import psutil
import zmq

sys.path.insert(0, "/app")

from shared.message_schema import (
    DetectionResult, ViolationEvent, SOPStateUpdate, Heartbeat,
    encode, decode,
)
from shared.zmq_topics import (
    DETECTION_RESULT, VIOLATION_EVENT, SOP_STATE_UPDATE, HEARTBEAT,
)

import database as db_module
import zone_manager
import identity_tracker as tracker
from api import start_api

# ── Logging ────────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sop_state_machine")

# ── Ports ──────────────────────────────────────────────────────────────────────

ZMQ_DETECTIONS_PORT = int(os.getenv("ZMQ_DETECTIONS_PORT", "5552"))
ZMQ_VIOLATIONS_PORT = int(os.getenv("ZMQ_VIOLATIONS_PORT", "5553"))
ZMQ_HEALTH_PORT     = int(os.getenv("ZMQ_HEALTH_PORT",     "5554"))

# ── Shared state ───────────────────────────────────────────────────────────────

stop_event = threading.Event()
stats_lock = threading.Lock()

detections_processed = 0
violations_raised    = 0


def shutdown(sig, frame):
    log.info("Shutdown signal received")
    stop_event.set()


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ── Severity logic ─────────────────────────────────────────────────────────────

def get_severity(status: str, role: str, zone_name: str) -> str:
    base = {
        "MASKED":     "HIGH",
        "UNKNOWN":    "HIGH",
        "RESTRICTED": "CRITICAL",
        "WRONG_ZONE": "MEDIUM",
        "AUTHORIZED": "LOW",
    }.get(status, "MEDIUM")
    return zone_manager.get_severity_upgrade(zone_name, base)


# ── Core decision logic ────────────────────────────────────────────────────────

def process_detection(result: DetectionResult,
                      frame_bytes: bytes,
                      violation_socket: zmq.Socket,
                      db_persons: list):
    global detections_processed, violations_raised

    cam_id    = result.cam_id
    timestamp = result.timestamp
    persons   = result.detections

    log.info(
        f"Detection received: cam={cam_id} persons={len(persons)} "
        f"time={datetime.utcfromtimestamp(timestamp).strftime('%H:%M:%S')}"
    )

    db = db_module.get_db()
    try:
        for person in persons:
            embedding    = person.get("embedding", [])
            bbox         = person.get("bbox", [0, 0, 0, 0])
            track_id     = person.get("track_id", -1)
            face_conf    = person.get("face_confidence", 0.0)

            # ── Step 1: Identity ────────────────────────────────────────────────
            # 1a. Check persisted track identity (YOLO track memory)
            cached = tracker.get_track_identity(cam_id, track_id)
            if cached:
                person_id, person_name, person_role = cached
                status = "AUTHORIZED"
                confidence = 1.0 # High confidence from previous match
                message = f"Track Persistence: {person_name} ({person_role})"
                global_id = tracker.get_or_create_global_id([], None, cam_id)
            
            # 1b. No cache, but has embedding -> run identification
            elif embedding:
                # Special case: masked person
                if 0.0 < face_conf < 0.4:
                    status, person_name, person_role, person_id = "MASKED", "Masked Person", "Unknown", None
                    global_id = tracker.get_or_create_global_id(embedding, None, cam_id)
                    message, confidence = "Face concealment detected", face_conf
                else:
                    match, score = tracker.identify_person(embedding, db_persons)
                    confidence = score
                    if match:
                        tracker.add_identity_vote(cam_id, track_id, match, score)
                    
                    final_match = tracker.get_majority_identity(cam_id, track_id, min_votes=3)
                    
                    if final_match:
                        person_name, person_role, person_id = final_match.name, final_match.role, final_match.id
                        tracker.link_track_to_person(cam_id, track_id, final_match)
                        global_id = tracker.get_or_create_global_id(embedding, final_match, cam_id)
                        status = "AUTHORIZED"
                        message = f"Identified: {person_name} ({person_role})"
                    else:
                        person_name, person_role, person_id = "Unknown", "Unknown", None
                        global_id = tracker.get_or_create_global_id(embedding, None, cam_id)
                        message, status = "Unknown person - not in database" if not match else "Verifying...", "UNKNOWN"
            
            # 1c. No cache and no embedding
            else:
                status, person_name, person_role, person_id = "UNKNOWN", "Unknown", "Unknown", None
                confidence = 0.0
                global_id = tracker.get_or_create_global_id([], None, cam_id)
                message = "Unidentifiable person - no face detected"

            # ── Step 2: Zone ────────────────────────────────────────────────────
            zone      = zone_manager.find_zone_for_person(cam_id, bbox)
            zone_name = zone.name if zone else "Unknown Zone"
            zone_id   = zone.id   if zone else None

            # ── Step 3: Authorization ───────────────────────────────────────────
            # --- STEP 4: Reporting ---
            v_ms = getattr(result, "video_pos_ms", 0.0)
            v_min = int(v_ms // 60000)
            v_sec = int((v_ms % 60000) // 1000)
            v_time = f"{v_min:02d}:{v_sec:02d}"

            if status == "AUTHORIZED":
                log.info(f"AUTHORIZED: {person_name} in {zone_name} cam={cam_id} [video={v_time}]")
            
            elif status in ("MASKED", "UNKNOWN", "RESTRICTED"):
                description = f"{status}: {person_name} in {zone_name}"
                if status == "RESTRICTED":
                    severity = "CRITICAL"
                elif status == "MASKED":
                    severity = "HIGH"
                else:
                    severity = "HIGH"

                log.warning(f"VIOLATION: {description} | cam={cam_id} | sev={severity} [video={v_time}]")

            # ── Step 4: Cross-camera trail ──────────────────────────────────────
            prev_cam = tracker.global_identities.get(
                global_id, {}
            ).get("last_seen_cam")
            tracker.update_trail(global_id, cam_id, timestamp, zone_name)

            if prev_cam and prev_cam != cam_id:
                log.info(
                    f"Cross-camera match: {global_id} seen on {cam_id} "
                    f"previously on {prev_cam}"
                )
                trail = tracker.global_identities.get(global_id, {}).get("trail", [])
                if len(trail) >= 2:
                    cams = " → ".join(t["cam_id"] for t in trail[-3:])
                    log.info(f"Movement trail: {global_id} {cams}")

            # ── Step 5: Log event ───────────────────────────────────────────────
            db_module.log_event(
                db,
                timestamp=datetime.utcfromtimestamp(timestamp).isoformat(),
                cam_id=cam_id,
                zone_name=zone_name,
                person_id=person_id,
                person_name=person_name,
                person_role=person_role,
                global_id=global_id,
                status=status,
                confidence=confidence,
                message=message,
            )

            # ── Step 6: Publish violation ────────────────────────────────
            if status != "AUTHORIZED":
                severity = get_severity(status, person_role, zone_name)
                tracker.record_violation(
                    global_id, status, zone_name, cam_id, timestamp
                )

                viol = ViolationEvent(
                    violation_id=f"{cam_id}_{track_id}_{int(timestamp)}",
                    violation_type=status,
                    cam_id=cam_id,
                    zone_id=str(zone_id) if zone_id else "none",
                    description=message,
                    severity=severity,
                    track_ids=[track_id],
                    timestamp=timestamp,
                )
                violation_socket.send_multipart(
                    [VIOLATION_EVENT, encode(viol), frame_bytes if frame_bytes else b""]
                )

                state_upd = SOPStateUpdate(
                    zone_id=str(zone_id) if zone_id else "none",
                    previous_state="MONITORING",
                    current_state=status,
                    trigger=f"{person_name} track={track_id}",
                    timestamp=timestamp,
                )
                violation_socket.send_multipart(
                    [SOP_STATE_UPDATE, encode(state_upd)]
                )

                with stats_lock:
                    violations_raised += 1

        with stats_lock:
            detections_processed += 1

    finally:
        db.close()


# ── Heartbeat ──────────────────────────────────────────────────────────────────

def heartbeat_loop(health_socket: zmq.Socket, start_time: float):
    proc = psutil.Process()
    while not stop_event.is_set():
        uptime = time.time() - start_time
        with stats_lock:
            dp = detections_processed
            vr = violations_raised

        identities  = tracker.list_identities()
        n_tracked   = len(identities)
        n_unknown   = tracker.count_unknown()
        cpu         = psutil.cpu_percent(interval=None)
        mem_mb      = proc.memory_info().rss / 1024 / 1024
        status      = "healthy"

        # Periodic cleanup of old track mappings
        tracker.clear_old_tracks(max_age_sec=60)

        hb = Heartbeat(
            service_name="sop_state_machine",
            status=status,
            cpu_percent=cpu,
            mem_mb=mem_mb,
            uptime_sec=uptime,
            details=(
                f"processed={dp} violations={vr} "
                f"tracked={n_tracked} unknown={n_unknown}"
            ),
            timestamp=time.time(),
        )
        health_socket.send_multipart([HEARTBEAT, encode(hb)])
        log.info(
            f"[HEARTBEAT] sop_state_machine | {status} | "
            f"processed={dp} violations={vr} "
            f"tracked={n_tracked}"
        )
        stop_event.wait(5)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Initialise SQLite
    db_module.init_db()
    log.info("Database initialised")

    # Start REST API
    start_api(port=8000)
    log.info("FastAPI started on port 8000")

    # ZMQ setup
    ctx = zmq.Context()

    det_socket = ctx.socket(zmq.SUB)
    det_socket.setsockopt(zmq.RCVTIMEO, 1000)
    det_socket.setsockopt(zmq.SUBSCRIBE, DETECTION_RESULT)
    det_socket.connect(f"tcp://ai_inference:{ZMQ_DETECTIONS_PORT}")
    log.info(f"ZMQ SUB connected to ai_inference:{ZMQ_DETECTIONS_PORT}")

    viol_socket = ctx.socket(zmq.PUB)
    viol_socket.bind(f"tcp://0.0.0.0:{ZMQ_VIOLATIONS_PORT}")
    log.info(f"ZMQ PUB bound on port {ZMQ_VIOLATIONS_PORT}")

    health_socket = ctx.socket(zmq.PUB)
    health_socket.bind(f"tcp://0.0.0.0:{ZMQ_HEALTH_PORT}")
    log.info(f"ZMQ PUB bound on port {ZMQ_HEALTH_PORT}")

    time.sleep(0.5)
    start_time = time.time()

    # Heartbeat thread
    threading.Thread(
        target=heartbeat_loop,
        args=(health_socket, start_time),
        daemon=True,
        name="heartbeat",
    ).start()

    log.info("Waiting for detection results...")

    # Main loop
    while not stop_event.is_set():
        # Reload persons from DB each cycle (cheap — small table)
        db_local = db_module.get_db()
        try:
            db_persons = db_module.get_all_persons(db_local)
        finally:
            db_local.close()

        try:
            parts = det_socket.recv_multipart()
            if len(parts) == 3:
                topic, data, frame_bytes = parts
            else:
                topic, data = parts
                frame_bytes = None
        except zmq.Again:
            continue
        except zmq.ZMQError as e:
            log.error(f"ZMQ receive error: {e}")
            continue

        try:
            result = decode(data, DetectionResult)
        except Exception as e:
            log.warning(f"DetectionResult decode error: {e}")
            continue

        process_detection(result, frame_bytes, viol_socket, db_persons)

    # Teardown
    log.info("Shutting down...")
    det_socket.close()
    viol_socket.close()
    health_socket.close()
    ctx.term()
    log.info("Stopped cleanly.")


if __name__ == "__main__":
    main()
