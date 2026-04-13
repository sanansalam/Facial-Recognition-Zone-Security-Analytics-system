import collections
import json
import logging
import queue
import signal
import sys
import threading
import time
from typing import List

import cv2
import numpy as np
import sqlite3
import zmq
import psutil
import gc

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STANDALONE_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
if STANDALONE_DIR not in sys.path:
    sys.path.insert(0, STANDALONE_DIR)

from shared.message_schema import FrameMessage, Detection, DetectionResult, Heartbeat, encode, decode
from shared.zmq_topics import RAW_FRAME, DETECTION_RESULT, HEARTBEAT

from shared.config import get_settings
settings = get_settings()

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -- Logging --
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ai_inference")

# -- Configuration --
ZMQ_RAW_FRAMES_PORT = settings.zmq.raw_frames
ZMQ_INFERENCE_PORT  = settings.zmq.inference
ZMQ_HEALTH_PORT     = settings.zmq.health_inference
ZMQ_HOST            = os.getenv("ZMQ_HOST", "localhost")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AI_INF_DIR = os.path.dirname(THIS_DIR)              
STANDALONE_DIR = os.path.dirname(AI_INF_DIR)        
PARENT_DIR = os.path.dirname(STANDALONE_DIR)         

DEFAULT_DB = os.path.join(STANDALONE_DIR, "sop_state_machine", "security.db")
DEFAULT_MODELS = os.path.join(PARENT_DIR, "edge_ai_security", "models")

DB_PATH             = os.getenv("DB_PATH", DEFAULT_DB)
MODEL_ROOT          = os.getenv("MODEL_ROOT", DEFAULT_MODELS)
YOLO_MODEL          = os.getenv("YOLO_MODEL", os.path.join(PARENT_DIR, "edge_ai_security", "yolo11n.pt"))
SIMILARITY_THRESH   = float(os.getenv("SIMILARITY_THRESH", "0.35"))
MARGIN_THRESH       = float(os.getenv("MARGIN_THRESH", "0.02"))   # min gap between 1st and 2nd place
MIN_FACE_PX         = int(os.getenv("MIN_FACE_PX", "35"))          # allow smaller CCTV faces
VOTE_WINDOW         = 2   # frames that must agree before confirming identity
ZMQ_MOTION_PORT     = int(os.getenv("ZMQ_MOTION_PORT", "5551"))

# Per camera ring buffer — stores compressed JPEG bytes
# 150 frames at ~10KB each = 1.5MB per camera
RING_BUFFER_MAXLEN = 150
ring_buffers: dict[str, collections.deque] = {}
ring_lock = threading.Lock()
# Each entry: {"ts": float, "jpg": bytes, "cam_id": str}

stop_event = threading.Event()

def shutdown(sig, frame):
    # Signal handlers should be minimal to avoid reentrant calls (e.g. logging)
    stop_event.set()

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

cam_stats = {}  # for heartbeat
stats_lock = threading.Lock()

# Queues
queue_b = queue.Queue(maxsize=10) # (cam_id, frame_bytes, timestamp, w, h, label)
queue_c = queue.Queue(maxsize=10) # (cam_id, yolo_res, frame_bytes, timestamp, w, h, label)
queue_d = queue.Queue(maxsize=20) # (DetectionResult, frame_jpg_bytes)
motion_states = {} # cam_id -> bool

# Temporal vote buffers: key=(cam_id, track_slot), value=deque of recent name guesses
vote_buffers = collections.defaultdict(lambda: collections.deque(maxlen=VOTE_WINDOW))
vote_lock = threading.Lock()
buffer_lock = threading.Lock()
ring_lock = threading.Lock()
RING_BUFFER_MAXLEN = 150 # 15 seconds at 10 fps
ring_buffers = {} # cam_id -> deque

# Identity persistence cache: key=(cam_id, track_id), value=(name, role, id, last_seen)
identity_cache = {}
identity_lock = threading.Lock()

class FaceDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.persons = []
        self.load()

    def load(self):
        try:
            if not os.path.exists(self.db_path):
                log.warning(f"Database not found at {self.db_path}. No persons will be recognized.")
                return
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT id, name, role, embedding FROM persons WHERE name != 'REJECTED'")
            rows = cursor.fetchall()
            self.persons = []
            for r in rows:
                self.persons.append({
                    "id":   r[0],
                    "name": r[1],
                    "role": r[2],
                    "emb":  np.array(json.loads(r[3]))
                })
            conn.close()
            log.info(f"Loaded {len(self.persons)} persons from database.")
        except Exception as e:
            log.error(f"Error loading database: {e}")

    def find_match(self, emb):
        """Match using cosine similarity with strict threshold + margin gate."""
        if not self.persons:
            return -1, "Unknown", "Unknown", 0.0

        sims = [(p["id"], p["name"], p["role"], float(np.dot(emb, p["emb"])))
                for p in self.persons]
        sims.sort(key=lambda x: x[3], reverse=True)

        top_id, top_name, top_role, top_sim = sims[0]
        second_sim = sims[1][3] if len(sims) > 1 else 0.0
        margin = top_sim - second_sim

        if top_sim >= SIMILARITY_THRESH and margin >= MARGIN_THRESH:
            log.info(f"Match found: {top_name} (sim={top_sim:.3f}, margin={margin:.3f})")
            return top_id, top_name, top_role, top_sim
        return -1, "Unknown", "Unknown", top_sim

yolo = None
face_app = None
rec_sess = None
rec_input = None
db = None

def load_models():
    global yolo, face_app, db
    from ultralytics import YOLO
    from insightface.app import FaceAnalysis

    db = FaceDatabase(DB_PATH)

    log.info("Loading YOLO11 Person detection model...")
    yolo = YOLO(YOLO_MODEL)
    
    log.info("Loading Face Detection (InsightFace)...")
    face_app = FaceAnalysis(name="buffalo_sc", root=MODEL_ROOT, providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))

def motion_subscriber(ctx: zmq.Context):
    """Listens for motion events to gate AI."""
    sub_motion = ctx.socket(zmq.SUB)
    sub_motion.connect(f"tcp://{ZMQ_HOST}:{ZMQ_MOTION_PORT}")
    sub_motion.setsockopt(zmq.SUBSCRIBE, b"") 
    sub_motion.setsockopt(zmq.RCVTIMEO, 100)
    log.info(f"Motion Monitor connected to port {ZMQ_MOTION_PORT}")

    while not stop_event.is_set():
        try:
            topic, payload = sub_motion.recv_multipart()
            data = json.loads(payload.decode("utf-8"))
            cam_id = data.get("cam_id")
            if topic == b"motion_detected":
                motion_states[cam_id] = True
            elif topic == b"motion_cleared":
                motion_states[cam_id] = False
        except zmq.Again:
            continue
        except Exception as e:
            log.error(f"Motion error: {e}")

def thread_a_collector(ctx: zmq.Context):
    log.info("Thread A (Collector) started.")
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.setsockopt(zmq.RCVTIMEO, 200)
    socket.setsockopt(zmq.SUBSCRIBE, RAW_FRAME)
    socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_RAW_FRAMES_PORT}")

    while not stop_event.is_set():
        try:
            parts = socket.recv_multipart()
        except zmq.Again:
            continue
        except zmq.ZMQError as e:
            continue

        if len(parts) == 3:
            topic, data, frame_bytes = parts
        else:
            continue

        try:
            msg = decode(data, FrameMessage)
        except Exception as e:
            continue

        with stats_lock:
            if msg.cam_id not in cam_stats:
                cam_stats[msg.cam_id] = {"inferred": 0}

        if not motion_states.get(msg.cam_id, True):
            continue

        # -- ADD TO RING BUFFER --
        # Compress aggressively to save RAM (Quality 40)
        frame_raw = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        with ring_lock:
            # 2. Add to Ring Buffer (Compressed to save RAM)
            with buffer_lock:
                if msg.cam_id not in ring_buffers:
                    ring_buffers[msg.cam_id] = collections.deque(maxlen=RING_BUFFER_MAXLEN)
                
                # Compress to JPEG Quality 40
                frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                _, buf = cv2.imencode('.jpg', frame_arr, [cv2.IMWRITE_JPEG_QUALITY, 40])
                
                ring_buffers[msg.cam_id].append({
                    'timestamp': msg.timestamp,
                    'frame': buf.tobytes()
                })
            
            # 3. Queue for YOLO (skip if inference too slow)
            if queue_b.full():
                try: queue_b.get_nowait()
                except: pass
            queue_b.put((msg.cam_id, frame_bytes, msg.timestamp, msg.width, msg.height, msg.cam_label))
            
    log.info("Thread A stopped.")

def thread_b_filter():
    log.info("Thread B (Filter) started.")
    while not stop_event.is_set():
        try:
            item = queue_b.get(timeout=1.0)
            cam_id, frame_bytes, timestamp, w, h, label = item
            
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, 3))

            try:
                # Use track() with persist=True and imgsz=640 for performance
                results = yolo.track(frame, imgsz=640, verbose=False, persist=True)[0]
            except Exception as e:
                log.error(f"YOLO track error: {e}")
                continue

            # Filter detections (Person)
            if results.boxes is not None:
                filtered = [d for d in results.boxes if int(d.cls[0]) == 0]
                if filtered:
                    queue_c.put((cam_id, results, frame_bytes, timestamp, w, h, label))
            
            queue_b.task_done()
        except queue.Empty:
            continue
        
        # Reduce GC frequency
        if timestamp % 5 < 0.1: # approx every 5 seconds per cam
            gc.collect()
    log.info("Thread B stopped.")

def thread_c_identifier():
    log.info("Thread C (Identifier) started.")
    while not stop_event.is_set():
        try:
            item = queue_c.get(timeout=1.0)
            cam_id, yolo_res, frame_bytes, timestamp, w, h, label = item
            
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, 3))
            detections = []
            
            for r in yolo_res.boxes:
                if int(r.cls) != 0: continue
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                conf_person = float(r.conf)
                track_id = int(r.id) if r.id is not None else -1
                
                p_id, p_name, p_role, conf_face = -1, "Unknown", "Unknown", 0.0
                
                # Check identity cache first
                with identity_lock:
                    if (cam_id, track_id) in identity_cache:
                        name, role, pid, last_ts = identity_cache[(cam_id, track_id)]
                        if time.time() - last_ts < 30: # 30s TTL
                            p_name, p_role, p_id = name, role, pid
                            conf_face = 0.99 # indicate cached confidence
                
                # If not in cache or unknown, try to identify
                if p_name == "Unknown":
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size > 0:
                        crop_h, crop_w = person_crop.shape[:2]
                        if crop_h >= MIN_FACE_PX and crop_w >= MIN_FACE_PX:
                            faces = face_app.get(person_crop)
                            if faces:
                                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                                if face.normed_embedding is not None:
                                    emb = face.normed_embedding
                                    raw_id, raw_name, raw_role, raw_sim = db.find_match(emb)
                                    p_id, p_name, p_role, conf_face = raw_id, raw_name, raw_role, raw_sim
                                    
                                    # Update cache if identified
                                    if p_name != "Unknown":
                                        with identity_lock:
                                            identity_cache[(cam_id, track_id)] = (p_name, p_role, p_id, time.time())

                detections.append(Detection(
                    id=p_id, name=p_name, role=p_role,
                    bbox=[x1, y1, x2, y2],
                    conf_person=conf_person, conf_face=conf_face
                ))

            # 4. Final Result
            res = DetectionResult(
                cam_id    = cam_id,
                cam_label = label,
                timestamp = timestamp,
                detections = detections
            )
            # Encode to JPG for bandwidth saving
            _, frame_jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            queue_d.put((res, frame_jpg.tobytes()))
            queue_c.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            log.error(f"Identifier error: {e}")
    log.info("Thread C stopped.")

def extract_clip_from_buffer(clip_frames: List[dict]) -> bytes:
    """
    Extract frames from ring buffer around violation timestamp.
    Returns mp4 video as bytes.
    """
    if len(clip_frames) < 5:
        return b""

    # Decode JPEG bytes back to numpy arrays
    decoded = []
    for entry in clip_frames:
        arr = np.frombuffer(entry["frame"], dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            decoded.append(frame)

    if not decoded:
        return b""

    # Write mp4 to BytesIO in memory via tempfile
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_name = tmp.name
    tmp.close()

    try:
        h, w = decoded[0].shape[:2]
        writer = cv2.VideoWriter(
            tmp_name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            10.0,
            (w, h)
        )
        for frame in decoded:
            writer.write(frame)
        writer.release()

        with open(tmp_name, "rb") as f:
            clip_bytes = f.read()
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    return clip_bytes

def thread_clip_server(ctx: zmq.Context):
    """Responds to clip requests from event_logger on port 5558."""
    socket = ctx.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5558")
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.setsockopt(zmq.LINGER, 0)
    log.info("Clip server listening on port 5558")

    while not stop_event.is_set():
        try:
            msg_bytes = socket.recv()
            req = json.loads(msg_bytes.decode())
            cam_id = req.get("cam_id")
            request_ts = req.get("timestamp")
            log.info(f"Clip request received: cam={cam_id}, ts={request_ts}")

            if cam_id not in ring_buffers:
                log.warning(f"Cam {cam_id} not in ring_buffers")
                socket.send(b"NO_CLIP")
                continue

            with buffer_lock:
                buf_snap = list(ring_buffers[cam_id])
            
            if not buf_snap:
                log.warning(f"Ring buffer for {cam_id} is empty")
                socket.send(b"NO_CLIP")
                continue

            clip_frames = [
                f for f in buf_snap
                if abs(f['timestamp'] - request_ts) <= 5.0
            ]
            
            if len(clip_frames) < 5:
                # Log the range of the buffer to debug timestamp mismatch
                first_ts = buf_snap[0]['timestamp']
                last_ts = buf_snap[-1]['timestamp']
                log.info(f"Not enough frames for clip: found {len(clip_frames)} in range {first_ts:.2f}-{last_ts:.2f} (req={request_ts:.2f})")
                socket.send(b"NO_CLIP")
                continue
            
            log.info(f"Assembling clip for {cam_id} ({len(clip_frames)} frames)...")
            clip_bytes = extract_clip_from_buffer(clip_frames)
            socket.send(clip_bytes if clip_bytes else b"NO_CLIP")
            
        except zmq.Again:
            continue
        except Exception as e:
            log.warning(f"Clip server error: {e}")
            try:
                socket.send(b"NO_CLIP")
            except:
                pass

def thread_d_publisher(ctx: zmq.Context):
    socket = ctx.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 10)
    socket.bind(f"tcp://0.0.0.0:{ZMQ_INFERENCE_PORT}")

    while not stop_event.is_set():
        try:
            res, frame_jpg_bytes = queue_d.get(timeout=1.0)
            socket.send_multipart([DETECTION_RESULT, encode(res), frame_jpg_bytes])
            with stats_lock:
                cam_stats[res.cam_id]["inferred"] = cam_stats[res.cam_id].get("inferred", 0) + 1
            queue_d.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            log.error(f"Publisher error: {e}")
    socket.close()
    log.info("Thread D stopped.")

def heartbeat_loop(ctx: zmq.Context, start_time: float):
    health_socket = ctx.socket(zmq.PUB)
    health_socket.bind(f"tcp://0.0.0.0:{ZMQ_HEALTH_PORT}")
    
    proc = psutil.Process()
    while not stop_event.is_set():
        uptime = time.time() - start_time
        with stats_lock:
            snap_stats = dict(cam_stats)
        
        cpu = psutil.cpu_percent(interval=None)
        mem_mb = proc.memory_info().rss / 1024 / 1024
        total_inferred = sum(v.get("inferred", 0) for v in snap_stats.values())
        status = "healthy" if snap_stats else "degraded"

        hb = Heartbeat(
            service_name="ai_inference",
            status=status,
            cpu_percent=cpu,
            mem_mb=mem_mb,
            uptime_sec=uptime,
            details=f"cameras={len(snap_stats)} inferred={total_inferred}",
            timestamp=time.time(),
        )
        health_socket.send_multipart([HEARTBEAT, encode(hb)])
        stop_event.wait(5)
    
    health_socket.close()

def main():
    load_models()
    log.info("Models ready.")

    ctx = zmq.Context()
    start_time = time.time()

    threads = [
        threading.Thread(target=motion_subscriber, args=(ctx,), name="Motion-Sub", daemon=True),
        threading.Thread(target=thread_a_collector, args=(ctx,), name="Collector-ThreadA"),
        threading.Thread(target=thread_b_filter, name="Filter-ThreadB"),
        threading.Thread(target=thread_c_identifier, name="Identifier-ThreadC"),
        threading.Thread(target=thread_d_publisher, args=(ctx,), name="Publisher-ThreadD"),
        threading.Thread(target=thread_clip_server, args=(ctx,), name="ClipServer-REP"),
        threading.Thread(target=heartbeat_loop, args=(ctx, start_time), name="Heartbeat", daemon=True)
    ]

    for t in threads:
        t.start()

    stop_event.wait()
    
    log.info("Shutting down cleanly...")
    for t in threads:
        t.join(timeout=3.0)
    
    ctx.term()
    log.info("Done.")

if __name__ == "__main__":
    main()
