"""
motion_detection — Service 2 (multi-camera)
One MOG2 background subtractor + motion state per camera.
Subscribes to raw_frame from video_ingestion (topic: b"raw_frame").
Routes each frame to the correct per-camera pipeline based on cam_id.
Publishes motion_detected / motion_cleared with cam_id on port 5551.
Publishes a combined Heartbeat every 5 seconds on port 5554.
"""

import base64
import logging
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
import psutil
import zmq

sys.path.insert(0, "/app")

from shared.config import get_settings
from shared.message_schema import (
    FrameMessage, MotionEvent, MotionCleared, Heartbeat,
    encode, decode,
)
from shared.zmq_topics import (
    RAW_FRAME, MOTION_DETECTED, MOTION_CLEARED, HEARTBEAT,
)

# ── Settings & logging ─────────────────────────────────────────────────────────

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("motion_detection")

# ── Shared state ───────────────────────────────────────────────────────────────

stop_event = threading.Event()
state_lock = threading.Lock()   # protects cam_states

# Per-camera state dict — created at startup for known cameras,
# also created on-demand when a new cam_id appears in frames.
#
# cam_states = {
#   "cam_0": {
#       "bg_subtractor": cv2.BackgroundSubtractorMOG2,
#       "motion_active":     bool,
#       "no_motion_frames":  int,
#       "warmup_count":      int,
#       "frames_processed":  int,
#       "last_motion_score": float,
#       "status":            str,   # "starting" | "active"
#   },
#   ...
# }
cam_states: dict = {}

# Motion detection constants (may be overridden by env)
NO_MOTION_THRESHOLD = 10
WARMUP_FRAMES = 25


def shutdown(sig, frame):
    log.info("Shutdown signal received")
    stop_event.set()


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ── Helper: create a fresh per-camera state entry ──────────────────────────────

def _make_cam_state(history: int) -> dict:
    return {
        "bg_subtractor": cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=50,
            detectShadows=False,
        ),
        "motion_active": False,
        "no_motion_frames": 0,
        "warmup_count": 0,
        "frames_processed": 0,
        "last_motion_score": 0.0,
        "status": "starting",
    }


# ── Per-camera motion pipeline ─────────────────────────────────────────────────

def process_frame(cam_id: str,
                  frame_bytes: bytes,
                  width: int,
                  height: int,
                  timestamp: float,
                  motion_socket: zmq.Socket,
                  sock_lock: threading.Lock,
                  min_area: int,
                  blur_size: int,
                  history: int) -> None:
    """
    Run one frame through the MOG2 pipeline for a single camera.
    cam_states must already be locked by the caller (state_lock).
    """

    # Step 1: Get or create cam state (dynamic camera discovery)
    if cam_id not in cam_states:
        cam_states[cam_id] = _make_cam_state(history)
        log.info(f"[{cam_id}] New camera registered dynamically")

    state = cam_states[cam_id]

    # Step 2: Decode frame
    try:
        if frame_bytes is None:
            log.warning(f"[{cam_id}] Frame bytes is missing")
            return
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
    except Exception as exc:
        log.warning(f"[{cam_id}] Frame decode error: {exc}")
        return

    # Step 3: Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # Step 4: Apply MOG2
    fg_mask = state["bg_subtractor"].apply(gray)

    # Step 5: Warmup check
    state["warmup_count"] += 1
    warmup_count = state["warmup_count"]

    if warmup_count < WARMUP_FRAMES:
        return                  # skip during warmup — do NOT count frames

    if warmup_count == WARMUP_FRAMES:
        state["status"] = "active"
        log.info(f"[{cam_id}] Warmup complete, motion detection active")

    # Step 6: Find contours + calculate score
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    motion_score = sum(cv2.contourArea(c) for c in valid)

    if valid:
        largest = max(valid, key=cv2.contourArea)
        bbox = list(cv2.boundingRect(largest))
    else:
        bbox = [0, 0, 0, 0]

    # Step 7: State machine — publish only on state transition
    if motion_score > 0:
        state["no_motion_frames"] = 0
        state["last_motion_score"] = motion_score

        if not state["motion_active"]:
            state["motion_active"] = True
            event = MotionEvent(
                cam_id=cam_id,
                motion_score=motion_score,
                bbox=bbox,
                timestamp=timestamp,       # ← original frame timestamp
            )
            with sock_lock:
                motion_socket.send_multipart([MOTION_DETECTED, encode(event)])
            log.info(
                f"[{cam_id}] Motion detected score={motion_score:.0f}"
                f" time={timestamp}"
            )
    else:
        state["no_motion_frames"] += 1
        if (state["no_motion_frames"] >= NO_MOTION_THRESHOLD
                and state["motion_active"]):
            state["motion_active"] = False
            cleared = MotionCleared(
                cam_id=cam_id,
                timestamp=timestamp,       # ← original frame timestamp
            )
            with sock_lock:
                motion_socket.send_multipart([MOTION_CLEARED, encode(cleared)])
            log.info(f"[{cam_id}] Motion cleared time={timestamp}")

    state["frames_processed"] += 1


# ── Heartbeat thread ───────────────────────────────────────────────────────────

def heartbeat_loop(health_socket: zmq.Socket,
                   start_time: float,
                   sock_lock: threading.Lock):
    proc = psutil.Process()
    while not stop_event.is_set():
        uptime = time.time() - start_time

        with state_lock:
            snap = {k: dict(v, bg_subtractor=None)
                    for k, v in cam_states.items()}

        cpu = psutil.cpu_percent(interval=None)
        mem_mb = proc.memory_info().rss / 1024 / 1024
        total_frames = sum(v["frames_processed"] for v in snap.values())
        status = "healthy" if total_frames > 0 else "degraded"

        motion_str = " ".join(
            f"{k}={'Y' if v['motion_active'] else 'N'}"
            for k, v in sorted(snap.items())
        )

        hb = Heartbeat(
            service_name="motion_detection",
            status=status,
            cpu_percent=cpu,
            mem_mb=mem_mb,
            uptime_sec=uptime,
            details=f"motion={motion_str} frames={total_frames}",
            timestamp=time.time(),
        )
        with sock_lock:
            health_socket.send_multipart([HEARTBEAT, encode(hb)])

        log.info(
            f"[HEARTBEAT] motion_detection | {status} |"
            f" motion={motion_str} | frames={total_frames}"
        )
        stop_event.wait(5)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    min_area = settings.motion_min_area
    blur_size = settings.motion_blur_size
    if blur_size % 2 == 0:
        blur_size += 1
    history = settings.motion_history

    # ── Camera discovery (same pattern as video_ingestion) ──────────────────
    known_cameras: list = []
    n = 0
    while True:
        source = os.getenv(f"CAM_{n}_SOURCE")
        if source is None:
            break
        known_cameras.append(f"cam_{n}")
        n += 1

    if not known_cameras:
        known_cameras = ["cam_0"]   # fallback for development

    log.info(f"Discovered {len(known_cameras)} cameras: {', '.join(known_cameras)}")

    # Pre-create cam_states for all known cameras
    for cam_id in known_cameras:
        cam_states[cam_id] = _make_cam_state(history)

    # ── ZMQ setup ───────────────────────────────────────────────────────────
    sock_lock = threading.Lock()
    ctx = zmq.Context()

    frame_socket = ctx.socket(zmq.SUB)
    frame_socket.setsockopt(zmq.RCVTIMEO, 500)
    frame_socket.setsockopt(zmq.SUBSCRIBE, RAW_FRAME)
    frame_socket.connect(f"tcp://video_ingestion:{settings.zmq.raw_frames}")
    log.info(f"ZMQ SUB connected to video_ingestion:{settings.zmq.raw_frames}")

    motion_socket = ctx.socket(zmq.PUB)
    motion_socket.bind(f"tcp://0.0.0.0:{settings.zmq.motion}")
    log.info(f"ZMQ PUB bound on port {settings.zmq.motion}")

    health_socket = ctx.socket(zmq.PUB)
    health_socket.bind(f"tcp://0.0.0.0:{settings.zmq.health}")
    log.info(f"ZMQ PUB bound on port {settings.zmq.health}")

    time.sleep(0.5)
    start_time = time.time()

    # ── Heartbeat thread ────────────────────────────────────────────────────
    hb_thread = threading.Thread(
        target=heartbeat_loop,
        args=(health_socket, start_time, sock_lock),
        daemon=True,
    )
    hb_thread.start()

    # ── Main dispatch loop ──────────────────────────────────────────────────
    # Single subscriber thread reads ALL frames from port 5550.
    # Routes each frame to the correct per-camera processor based on cam_id.
    while not stop_event.is_set():
        try:
            parts = frame_socket.recv_multipart()
            if len(parts) == 3:
                topic, data, frame_bytes = parts
            else:
                topic, data = parts
                frame_bytes = None
        except zmq.Again:
            continue
        except zmq.ZMQError as exc:
            log.error(f"ZMQ receive error: {exc}")
            continue

        try:
            msg = decode(data, FrameMessage)
        except Exception as exc:
            log.warning(f"FrameMessage decode error: {exc}")
            continue

        cam_id = msg.cam_id
        timestamp = msg.timestamp      # ← preserve original frame timestamp

        with state_lock:
            process_frame(
                cam_id=cam_id,
                frame_bytes=frame_bytes,
                width=msg.width,
                height=msg.height,
                timestamp=timestamp,
                motion_socket=motion_socket,
                sock_lock=sock_lock,
                min_area=min_area,
                blur_size=blur_size,
                history=history,
            )

    # ── Teardown ────────────────────────────────────────────────────────────
    log.info("Shutting down...")
    frame_socket.close()
    motion_socket.close()
    health_socket.close()
    ctx.term()
    log.info("Stopped cleanly.")


if __name__ == "__main__":
    main()
