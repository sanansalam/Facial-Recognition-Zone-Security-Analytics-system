import os
import sys
import time
import json
import sqlite3
import base64
import hashlib
import logging
import signal
import collections
from datetime import datetime
import threading
import zmq
from shared.message_schema import ViolationEvent, Heartbeat, decode
from shared.zmq_topics import VIOLATION_EVENT, HEARTBEAT

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
APP_DIR     = os.path.dirname(THIS_DIR)              # event_logger
STANDALONE  = os.path.dirname(APP_DIR)               # video_ingestion_standalone
DATA_DIR    = os.path.join(STANDALONE, "data")
EVIDENCE_DIR= os.path.join(DATA_DIR, "evidence")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
DB_PATH     = os.path.join(DATA_DIR, "events.db")

# Create dirs
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Path setup for shared schema
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STANDALONE_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
if STANDALONE_DIR not in sys.path:
    sys.path.insert(0, STANDALONE_DIR)

from shared.message_schema import ViolationEvent, Heartbeat, decode
from shared.zmq_topics import VIOLATION_EVENT, HEARTBEAT

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("event_logger")

ZMQ_VIOLATION_PORT = 5553
ZMQ_HEALTH_PORT    = 5554
ZMQ_HOST           = os.getenv("ZMQ_HOST", "localhost")

# Global State
stop_event = threading.Event()
cooldown_dict = {}  # key: f"{person_name}_{zone_name}_{cam_id}", val: timestamp
last_heartbeats = {} # key: service_name, val: timestamp
clip_socket = None # ZMQ REQ for clips
clip_lock = threading.Lock()
clip_queue = collections.deque() # To-be-processed clip requests
clip_worker_lock = threading.Lock()

# DB Setup
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            cam_id       TEXT,
            cam_label    TEXT,
            zone_name    TEXT,
            person_name  TEXT,
            person_role  TEXT,
            status       TEXT,
            severity     TEXT,
            message      TEXT,
            evidence_path TEXT,
            cam_position TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS service_health (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            service_name TEXT,
            status       TEXT,
            cpu_percent  REAL,
            mem_mb       REAL,
            uptime_sec   REAL
        )
    """)
    conn.commit()
    conn.close()

def shutdown(sig, frame):
    stop_event.set()

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ---------------------------------------------------------
# REPORT GENERATOR
# ---------------------------------------------------------
def generate_report():
    now = datetime.now()
    report_file = os.path.join(REPORTS_DIR, f"report_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    
    # Time window (for this demo, we'll just grab last 1 hour of events or all if testing)
    conn = get_db()
    
    # 1 hr ago
    time_threshold = now.timestamp() - 3600
    iso_thresh = datetime.fromtimestamp(time_threshold).isoformat()
    
    violations = conn.execute("SELECT * FROM violations WHERE timestamp >= ? ORDER BY timestamp ASC", (iso_thresh,)).fetchall()
    
    total_viols = len(violations)
    unknowns = sum(1 for v in violations if v["status"] == "UNKNOWN")
    restricted = sum(1 for v in violations if v["status"] == "RESTRICTED")
    wrong_zone = sum(1 for v in violations if v["status"] == "WRONG_ZONE")
    
    cam_counts = collections.defaultdict(int)
    cam_labels = {}
    zone_counts = collections.defaultdict(int)
    known_persons = collections.defaultdict(lambda: collections.defaultdict(int))
    
    evidence_files = []
    
    for v in violations:
        cam_counts[v["cam_id"]] += 1
        cam_labels[v["cam_id"]] = v["cam_label"]
        zone_counts[v["zone_name"]] += 1
        
        if v["person_name"] != "Unknown":
            known_persons[v["person_name"]]["role"] = v["person_role"]
            known_persons[v["person_name"]]["count"] += 1
            
        if v["evidence_path"]:
            evidence_files.append(v["evidence_path"])
    
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "SECURITY REPORT — Jewellery Shop",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Period: {datetime.fromtimestamp(time_threshold).strftime('%Y-%m-%d %H:%M:%S')} to {now.strftime('%H:%M:%S')}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "SUMMARY",
        f"  Total violations:     {total_viols}",
        f"  Unknown persons:      {unknowns}",
        f"  Restricted access:    {restricted}",
        f"  Wrong zone:           {wrong_zone}",
        f"  Cameras monitored:    {len(cam_counts)}",
        "",
        "VIOLATIONS BY CAMERA"
    ]
    
    for cid, count in sorted(cam_counts.items(), key=lambda x: -x[1]):
        clabel = cam_labels[cid]
        lines.append(f"  {cid} ({clabel}): {count} violations")
        
    lines.extend(["", "VIOLATIONS BY ZONE"])
    for zname, count in sorted(zone_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {zname:15s}: {count}")
        
    lines.extend(["", "KNOWN PERSONS SEEN"])
    for pname, pdata in sorted(known_persons.items(), key=lambda x: -x[1]["count"]):
        role = pdata["role"]
        count = pdata["count"]
        lines.append(f"  {pname:10s} ({role:10s}) → AUTHORIZED {count} times")
        
    lines.extend(["", "VIOLATION DETAILS"])
    for v in violations:
        dtstr = datetime.fromisoformat(v["timestamp"]).strftime('%H:%M:%S')
        lines.append(f"  {dtstr} | {v['status']:10s} | {v['person_name']:10s} | {v['zone_name']} | {v['cam_id']}")
        
    lines.extend(["", "EVIDENCE FILES"])
    for ef in evidence_files[-10:]:  # max 10 to keep report clean
        lines.append(f"  {ef}")
        
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    report_text = "\n".join(lines)
    
    with open(report_file, "w") as f:
        f.write(report_text)
        
    print("\n" + report_text + "\n", flush=True)
    log.info(f"Report generated: {report_file}")
    conn.close()

def clip_worker_thread():
    """Background thread to process clip requests without blocking alerts."""
    log.info("Clip worker thread started.")
    while not stop_event.is_set():
        if not clip_queue:
            time.sleep(0.5)
            continue
            
        with clip_worker_lock:
            try:
                cam_id, timestamp, person_name, ev_path_container = clip_queue.popleft()
            except IndexError:
                continue
        
        log.info(f"Background: Requesting evidence clip for {cam_id} at {timestamp}...")
        clip_bytes = request_clip(cam_id, timestamp)
        
        if clip_bytes:
            ts_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
            c_path = os.path.join(EVIDENCE_DIR, f"{cam_id}_{ts_str}_{person_name.replace(' ','_')}.mp4")
            try:
                with open(c_path, "wb") as f:
                    f.write(clip_bytes)
                log.info(f"Evidence clip saved: {c_path} ({len(clip_bytes)//1024} KB)")
                # Update the database record if possible? 
                # For now, we just save the file. The original event already points to the JPG or the MP4 path we predicted.
            except Exception as e:
                log.error(f"Failed to save background clip: {e}")
        else:
            log.warning(f"No clip returned for {cam_id} at {timestamp}")

def request_clip(cam_id, timestamp):
    """Requests .mp4 bytes from ai_inference on port 5555."""
    global clip_socket
    if clip_socket is None:
        return None
    
    with clip_lock:
        try:
            req = json.dumps({
                "cam_id": cam_id,
                "timestamp": timestamp
            }).encode()
            clip_socket.send(req)
            
            # Use long timeout
            clip_bytes = clip_socket.recv()
            if clip_bytes == b"NO_CLIP" or not clip_bytes:
                return None
            return clip_bytes
        except zmq.ZMQError as e:
            log.warning(f"Clip request ZMQ error: {e}. Recreating socket...")
            # Recreate socket to clear state
            try:
                clip_socket.setsockopt(zmq.LINGER, 0)
                clip_socket.close()
            except:
                pass
            clip_socket = ctx_global.socket(zmq.REQ)
            clip_socket.connect(f"tcp://{ZMQ_HOST}:5558")
            clip_socket.setsockopt(zmq.RCVTIMEO, 15000)
            clip_socket.setsockopt(zmq.SNDTIMEO, 5000)
            return None
        except Exception as e:
            log.warning(f"Clip request failed: {e}")
            return None

ctx_global = None

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("[INFO] Event logger started")
    print(f"[INFO] Database: {DB_PATH}")
    print(f"[INFO] Evidence: {EVIDENCE_DIR}")
    print(f"[INFO] Reports:  {REPORTS_DIR}")
    print(f"[INFO] Subscribed to violations on port {ZMQ_VIOLATION_PORT}")
    print(f"[INFO] Subscribed to heartbeats on port {ZMQ_HEALTH_PORT}")

    init_db()
    ctx = zmq.Context()

    # Violation socket
    sub_viol = ctx.socket(zmq.SUB)
    sub_viol.connect(f"tcp://localhost:{ZMQ_VIOLATION_PORT}")
    sub_viol.setsockopt(zmq.SUBSCRIBE, VIOLATION_EVENT)

    # Health socket
    sub_health = ctx.socket(zmq.SUB)
    sub_health.connect(f"tcp://{ZMQ_HOST}:{ZMQ_HEALTH_PORT}")
    sub_health.setsockopt(zmq.SUBSCRIBE, HEARTBEAT)

    # Clip Request Socket (REQ)
    global clip_socket, ctx_global
    ctx_global = ctx
    clip_socket = ctx.socket(zmq.REQ)
    clip_socket.connect(f"tcp://{ZMQ_HOST}:5558")
    clip_socket.setsockopt(zmq.RCVTIMEO, 15000) # Increased to 15s
    clip_socket.setsockopt(zmq.SNDTIMEO, 5000)
    clip_socket.setsockopt(zmq.LINGER, 0)

    poller = zmq.Poller()
    poller.register(sub_viol, zmq.POLLIN)
    poller.register(sub_health, zmq.POLLIN)

    # Start Clip Worker
    worker = threading.Thread(target=clip_worker_thread, daemon=True)
    worker.start()

    conn = get_db()
    
    last_report_ts = time.time()
    last_health_check = time.time()

    while not stop_event.is_set():
        try:
            # Poll frequently (100ms) so it can exit quickly on shutdown
            socks = dict(poller.poll(100))
        except KeyboardInterrupt:
            break

        now = time.time()
        
        # 1. Garbage collect every 60s (without printing summary)
        if now - last_report_ts >= 60:
            import gc
            gc.collect()
            last_report_ts = now
            
        # 2. Check health staleness every 5s
        if now - last_health_check >= 5:
            for sname, lseen in last_heartbeats.items():
                if now - lseen > 30:
                    log.warning(f"Service {sname} may be down (no heartbeat for {int(now-lseen)}s)")
            last_health_check = now

        # Handle Heartbeat
        if sub_health in socks:
            topic, payload = sub_health.recv_multipart()
            try:
                hb = decode(payload, Heartbeat)
                last_heartbeats[hb.service_name] = hb.timestamp
                
                isotime = datetime.fromtimestamp(hb.timestamp).isoformat()
                conn.execute("""
                    INSERT INTO service_health 
                    (timestamp, service_name, status, cpu_percent, mem_mb, uptime_sec)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (isotime, hb.service_name, hb.status, hb.cpu_percent, hb.mem_mb, hb.uptime_sec))
                conn.commit()
            except Exception as e:
                log.error(f"Health parse error: {e}")

        # Handle Violation
        if sub_viol in socks:
            topic, payload = sub_viol.recv_multipart()
            try:
                data = json.loads(payload.decode('utf-8'))
                
                import dataclasses
                known = {f.name for f in dataclasses.fields(ViolationEvent)}
                filtered = {k: v for k, v in data.items() if k in known}
                v = ViolationEvent(**filtered)
                
                # 1. IMMEDIATE NOTIFICATION
                severity = v.severity if v.severity else "LOW"
                if v.status != "AUTHORIZED":
                    severity = "HIGH"
                
                if severity in ["HIGH", "CRITICAL"]:
                    icon = "🚨" if v.status == "UNKNOWN" else "🛑"
                    if v.person_name == "Unknown":
                        msg_body = f"Unknown person in {v.zone_name} [{v.cam_label}]"
                    else:
                        msg_body = f"{v.person_name} ({v.person_role}) in {v.zone_name} [{v.cam_label}]"
                    
                    os.system(f'notify-send "{icon} {v.status}" "{msg_body}" --urgency=critical')
                    time.sleep(0.05)
                
                # 2. Cooldown Check
                key = f"{v.person_name}_{v.zone_name}_{v.cam_id}"
                if key in cooldown_dict:
                    if now - cooldown_dict[key] < 30:
                        continue
                cooldown_dict[key] = now
                
                # 3. Evidence JPEG
                ev_path = None
                if getattr(v, "evidence_jpeg", ""):
                    try:
                        jpeg_bytes = base64.b64decode(v.evidence_jpeg)
                        digest = hashlib.md5(f"{v.person_name}_{v.zone_name}".encode()).hexdigest()[:6]
                        fname = f"{v.cam_id}_{int(v.timestamp)}_{v.person_name.replace(' ','_')}_{digest}.jpg"
                        ev_path = os.path.join(EVIDENCE_DIR, fname)
                        with open(ev_path, "wb") as f:
                            f.write(jpeg_bytes)
                    except Exception as e:
                        log.error(f"Failed to save evidence JPEG: {e}")
                
                # 4. Queue Video Clip Request (Asynchronous)
                ts_str = datetime.fromtimestamp(v.timestamp).strftime("%Y%m%d_%H%M%S")
                mp4_fname = f"{v.cam_id}_{ts_str}_{v.person_name.replace(' ','_')}.mp4"
                possible_mp4_path = os.path.join(EVIDENCE_DIR, mp4_fname)
                
                with clip_worker_lock:
                    clip_queue.append((v.cam_id, v.timestamp, v.person_name, possible_mp4_path))
                
                # If we expect a clip, we set the path now (it will be filled later)
                # This ensures the DB record points to where the file WILL be.
                ev_path = possible_mp4_path if not ev_path else ev_path

                # 5. Save to DB
                pos_str = f"{v.position[0]},{v.position[1]}" if v.position and len(v.position)>=2 else ""
                if not pos_str and v.bbox and len(v.bbox)>=4:
                    pos_str = f"{(v.bbox[0]+v.bbox[2])//2},{v.bbox[3]}"
                    
                isotime = datetime.fromtimestamp(v.timestamp).isoformat()
                
                conn.execute("""
                    INSERT INTO violations 
                    (timestamp, cam_id, cam_label, zone_name, person_name, person_role, status, severity, message, evidence_path, cam_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    isotime, v.cam_id, v.cam_label, v.zone_name, v.person_name, 
                    v.person_role, v.status, severity, v.message, ev_path, pos_str
                ))
                conn.commit()
                
                # Terminal Log
                time_str = datetime.fromtimestamp(v.timestamp).strftime('%H:%M:%S')
                print(f"[EVENT] {v.status} | {v.person_name} | {v.zone_name} | {v.cam_id} | {time_str}", flush=True)
                
            except Exception as e:
                log.error(f"Event parsing error: {e}")

    conn.close()
    
    log.info("Generating final report...")
    generate_report()
    log.info("Event logger stopped. Report saved.")

if __name__ == "__main__":
    main()
