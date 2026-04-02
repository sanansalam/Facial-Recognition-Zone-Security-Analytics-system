"""
motion_detection — Service 2 (multi-camera)
"""
import base64, logging, os, signal, sys, threading, time
import cv2, numpy as np, psutil, zmq
from shared.config import get_settings
from shared.message_schema import FrameMessage, MotionEvent, MotionCleared, Heartbeat, encode, decode
from shared.zmq_topics import RAW_FRAME, MOTION_DETECTED, MOTION_CLEARED, HEARTBEAT

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("motion_detection")

stop_event, state_lock, cam_states = threading.Event(), threading.Lock(), {}
NO_MOTION_THRESHOLD, WARMUP_FRAMES = 10, 25

def shutdown(sig, frame): 
    log.info("Shutdown signal received"); stop_event.set()

signal.signal(signal.SIGINT, shutdown); signal.signal(signal.SIGTERM, shutdown)

def _make_cam_state(history):
    return {"bg_subtractor": cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=50, detectShadows=False),
            "motion_active": False, "no_motion_frames": 0, "warmup_count": 0, "frames_processed": 0, "last_motion_score": 0.0, "status": "starting"}

def process_frame(cam_id, frame_jpeg, timestamp, motion_socket, sock_lock, min_area, blur_size, history):
    if cam_id not in cam_states: cam_states[cam_id] = _make_cam_state(history)
    state = cam_states[cam_id]
    try:
        arr = np.frombuffer(base64.b64decode(frame_jpeg), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None: return
    except Exception as exc: return
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (blur_size, blur_size), 0)
    fg_mask = state["bg_subtractor"].apply(gray)
    state["warmup_count"] += 1
    if state["warmup_count"] < WARMUP_FRAMES: return
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    motion_score = sum(cv2.contourArea(c) for c in valid)
    if motion_score > 0:
        state["no_motion_frames"] = 0
        if not state["motion_active"]:
            state["motion_active"] = True
            with sock_lock: motion_socket.send_multipart([MOTION_DETECTED, encode(MotionEvent(cam_id=cam_id, motion_score=motion_score, bbox=list(cv2.boundingRect(max(valid, key=cv2.contourArea))), timestamp=timestamp))])
            log.info(f"[{cam_id}] Motion detected")
    else:
        state["no_motion_frames"] += 1
        if state["no_motion_frames"] >= NO_MOTION_THRESHOLD and state["motion_active"]:
            state["motion_active"] = False
            with sock_lock: motion_socket.send_multipart([MOTION_CLEARED, encode(MotionCleared(cam_id=cam_id, timestamp=timestamp))])
            log.info(f"[{cam_id}] Motion cleared")
    state["frames_processed"] += 1

def heartbeat_loop(health_socket, start_time, sock_lock):
    proc = psutil.Process()
    while not stop_event.is_set():
        uptime = time.time() - start_time
        with state_lock: total_frames = sum(v["frames_processed"] for v in cam_states.values())
        hb = Heartbeat(service_name="motion_detection", status="healthy" if total_frames > 0 else "degraded", cpu_percent=psutil.cpu_percent(), mem_mb=proc.memory_info().rss / 1024 / 1024, uptime_sec=uptime, timestamp=time.time())
        with sock_lock: health_socket.send_multipart([HEARTBEAT, encode(hb)])
        stop_event.wait(5)

def main():
    ctx, sock_lock = zmq.Context(), threading.Lock()
    frame_socket = ctx.socket(zmq.SUB)
    frame_socket.setsockopt(zmq.SUBSCRIBE, RAW_FRAME)
    frame_socket.connect(f"tcp://localhost:{settings.zmq.raw_frames}")
    motion_socket = ctx.socket(zmq.PUB)
    motion_socket.bind(f"tcp://0.0.0.0:{settings.zmq.motion}")
    health_socket = ctx.socket(zmq.PUB)
    health_socket.bind(f"tcp://0.0.0.0:{settings.zmq.health}")
    time.sleep(0.5); start_time = time.time()
    threading.Thread(target=heartbeat_loop, args=(health_socket, start_time, sock_lock), daemon=True).start()
    while not stop_event.is_set():
        try:
            topic, data = frame_socket.recv_multipart(); msg = decode(data, FrameMessage)
            with state_lock: process_frame(msg.cam_id, msg.frame_jpeg, msg.timestamp, motion_socket, sock_lock, settings.motion_min_area, settings.motion_blur_size, settings.motion_history)
        except Exception: continue
    frame_socket.close(); motion_socket.close(); health_socket.close(); ctx.term()

if __name__ == "__main__": main()
