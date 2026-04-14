"""
video_ingestion — Service 1 (multi-camera)
Discovers all CAM_N_SOURCE entries from .env at startup.
Spawns one thread per camera. All threads publish on the
same ZMQ PUB socket (protected by a lock).
Video files loop seamlessly. Webcams retry on failure.
"""

import base64
import logging
import os
import signal
import sys
import threading
import time
from collections import deque

import cv2
import psutil
import zmq

sys.path.insert(0, "/app")

from shared.message_schema import FrameMessage, Heartbeat, encode
from shared.zmq_topics import RAW_FRAME, HEARTBEAT

# ── Logging ────────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("video_ingestion")

# ── Camera discovery ───────────────────────────────────────────────────────────

def discover_cameras():
    cameras = []
    n = 0
    while True:
        source = os.getenv(f"CAM_{n}_SOURCE")
        if source is None:
            break
        label = os.getenv(f"CAM_{n}_LABEL", f"Camera {n}")
        try:
            source = int(source)
            is_file = False
        except ValueError:
            is_file = True
        cameras.append({
            "id":      f"cam_{n}",
            "source":  source,
            "label":   label,
            "is_file": is_file,
        })
        n += 1
    return cameras


cameras = discover_cameras()

if not cameras:
    log.error("No cameras found. Set CAM_0_SOURCE at minimum.")
    sys.exit(1)

log.info(f"Discovered {len(cameras)} camera(s): "
         f"{[c['label'] for c in cameras]}")

# ── Shared state ───────────────────────────────────────────────────────────────

stop_event  = threading.Event()
socket_lock = threading.Lock()
stats_lock  = threading.Lock()

stats = {
    c["id"]: {
        "fps":       0.0,
        "published": 0,
        "status":    "starting",
        "source":    str(c["source"]),
    }
    for c in cameras
}

FRAME_SKIP      = int(os.getenv("FRAME_SKIP", "3"))
ZMQ_FRAMES_PORT = int(os.getenv("ZMQ_RAW_FRAMES_PORT", "5550"))
ZMQ_HEALTH_PORT = int(os.getenv("ZMQ_HEALTH_PORT", "5554"))


def shutdown(sig, frame):
    log.info("Shutdown signal received")
    stop_event.set()


signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ── Camera open helper ─────────────────────────────────────────────────────────

def open_camera(source):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        return None
    return cap

# ── Per-camera thread ──────────────────────────────────────────────────────────

def camera_thread(cam_id, cam_source, cam_label, is_file, frame_socket):
    raw_count         = 0
    published_count   = 0
    consecutive_fails = 0
    fps_times         = deque(maxlen=30)
    cap               = None

    log.info(f"[{cam_id}] Thread started "
             f"(source={cam_source}, is_file={is_file})")

    while not stop_event.is_set():

        # ── Open camera / video file ────────────────────────────────────────
        if cap is None or not cap.isOpened():
            cap = open_camera(cam_source)
            if cap is None:
                with stats_lock:
                    stats[cam_id]["status"] = "retrying"
                log.error(f"[{cam_id}] Cannot open source, retrying in 5s...")
                stop_event.wait(5)
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(f"[{cam_id}] Opened: {cam_source} at {w}x{h}")
            consecutive_fails = 0

        # ── Read frame ──────────────────────────────────────────────────────
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        raw_count += 1

        if not ret:
            if is_file:
                # Video file ended — loop back to beginning seamlessly
                log.info(f"[{cam_id}] End of video — looping back to start")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                raw_count = 0
                continue
            else:
                # Real camera failure — retry
                consecutive_fails += 1
                if consecutive_fails > 30:
                    log.error(f"[{cam_id}] Too many failures, reopening...")
                    cap.release()
                    cap = None
                continue

        consecutive_fails = 0

        # ── Frame skip ──────────────────────────────────────────────────────
        if raw_count % FRAME_SKIP != 0:
            continue

        # ── Frame Resolution ────────────────────────────────────────────────
        h_frame, w_frame = frame.shape[:2]

        # No JPEG encoding, use raw numpy bytes directly
        frame_bytes = frame.tobytes()

        # ── FPS tracking ────────────────────────────────────────────────────
        now = time.time()
        fps_times.append(now)
        fps = 0.0
        if len(fps_times) > 1:
            elapsed = fps_times[-1] - fps_times[0]
            if elapsed > 0:
                fps = (len(fps_times) - 1) / elapsed

        # ── Build & publish ─────────────────────────────────────────────────
        msg = FrameMessage(
            cam_id     = cam_id,
            cam_label  = cam_label,
            width      = w_frame,
            height     = h_frame,
            timestamp  = now,
            video_pos_ms = pos_ms,
            sequence   = published_count,
        )
        with socket_lock:
            frame_socket.send_multipart([RAW_FRAME, encode(msg), frame_bytes])

        # ── Update stats ────────────────────────────────────────────────────
        with stats_lock:
            stats[cam_id]["fps"]       = fps
            stats[cam_id]["published"] = published_count
            stats[cam_id]["status"]    = "healthy"

        if published_count % 30 == 0:
            log.info(f"[{cam_id}] seq={published_count} fps={fps:.1f}")

        published_count += 1

    if cap is not None:
        cap.release()
    log.info(f"[{cam_id}] Thread stopped.")

# ── Heartbeat thread ───────────────────────────────────────────────────────────

def heartbeat_loop(health_socket, start_time):
    proc = psutil.Process()
    while not stop_event.is_set():
        uptime = time.time() - start_time
        with stats_lock:
            snap = {k: dict(v) for k, v in stats.items()}

        statuses = [v["status"] for v in snap.values()]
        if all(s in ("retrying", "starting") for s in statuses):
            overall = "error"
        elif any(s in ("retrying", "starting") for s in statuses):
            overall = "degraded"
        else:
            overall = "healthy"

        cpu    = psutil.cpu_percent(interval=None)
        mem_mb = proc.memory_info().rss / 1024 / 1024

        hb = Heartbeat(
            service_name = "video_ingestion",
            status       = overall,
            cpu_percent  = cpu,
            mem_mb       = mem_mb,
            uptime_sec   = uptime,
            details      = str({"cameras": snap}),
            timestamp    = time.time(),
        )
        with socket_lock:
            health_socket.send_multipart([HEARTBEAT, encode(hb)])

        cam_summary = " ".join(
            f"{k}={v['fps']:.1f}fps"
            if v["status"] == "healthy"
            else f"{k}={v['status']}"
            for k, v in sorted(snap.items())
        )
        log.info(f"[HEARTBEAT] video_ingestion | {overall} | {cam_summary}")
        stop_event.wait(5)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ctx = zmq.Context()

    frame_socket = ctx.socket(zmq.PUB)
    frame_socket.bind(f"tcp://0.0.0.0:{ZMQ_FRAMES_PORT}")
    log.info(f"ZMQ PUB bound on port {ZMQ_FRAMES_PORT}")

    health_socket = ctx.socket(zmq.PUB)
    health_socket.bind(f"tcp://0.0.0.0:{ZMQ_HEALTH_PORT}")
    log.info(f"ZMQ PUB bound on port {ZMQ_HEALTH_PORT}")

    time.sleep(0.5)
    start_time = time.time()

    threads = []
    for cam in cameras:
        t = threading.Thread(
            target  = camera_thread,
            args    = (cam["id"], cam["source"], cam["label"],
                       cam["is_file"], frame_socket),
            daemon  = True,
            name    = f"cam-{cam['id']}",
        )
        t.start()
        threads.append(t)
        log.info(f"Launched thread for {cam['id']} ({cam['label']})")

    threading.Thread(
        target = heartbeat_loop,
        args   = (health_socket, start_time),
        daemon = True,
        name   = "heartbeat",
    ).start()

    stop_event.wait()
    log.info("Shutting down — joining threads...")
    for t in threads:
        t.join(timeout=8)

    frame_socket.close()
    health_socket.close()
    ctx.term()
    log.info("Stopped cleanly.")


if __name__ == "__main__":
    main()
