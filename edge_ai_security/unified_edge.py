import base64
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import psutil
import gc

# Add project root and module paths to sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "sop_state_machine", "app"))

# Ensure relative data/models paths work for native execution
os.environ.setdefault("DATA_DIR", "./data")
os.environ.setdefault("MODELS_DIR", "./models")
os.environ.setdefault("YOLO_MODEL_PATH", "./models/yolo11n.onnx")

from shared.config import get_settings, setup_logging
from shared.message_schema import FrameMessage, DetectionResult, DetectedObject
import database as db_module
import zone_manager
import identity_tracker as tracker

# ── Configuration ──────────────────────────────────────────────────────────────

settings = get_settings()
log = setup_logging("unified_edge")

# Override paths for native environment
DATA_DIR = os.getenv("DATA_DIR", "./data")
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Optimization Constants
MAX_INFERENCE_FPS = float(os.getenv("MAX_INFERENCE_FPS", "2.0"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))
MOTION_GATED_INFERENCE = os.getenv("MOTION_GATED_INFERENCE", "true").lower() == "true"

# ── Shared Models ──────────────────────────────────────────────────────────────

yolo_model = None
face_app = None
model_lock = threading.Lock()

def load_models():
    global yolo_model, face_app
    from ultralytics import YOLO
    from insightface.app import FaceAnalysis

    yolo_path = os.path.join(MODELS_DIR, "yolo11n.onnx")
    if not os.path.exists(yolo_path):
        log.error(f"YOLO model not found at {yolo_path}. Run download_models.sh first.")
        sys.exit(1)

    log.info("Loading YOLO11n (ONNX)...")
    yolo_model = YOLO(yolo_path, task="detect")
    
    model_name = os.getenv("INSIGHTFACE_MODEL", "buffalo_sc")
    log.info(f"Loading InsightFace ({model_name})...")
    
    # Ultra-low resource mode: Use even smaller detection size if requested
    det_size = (settings.internal_width, settings.internal_height)
    if settings.ultra_low_resource:
        det_size = (320, 320)
        
    face_app = FaceAnalysis(
        name=model_name,
        root=os.path.join(MODELS_DIR, "insightface"),
        providers=["CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=-1, det_size=det_size)
    log.info(f"Models loaded. Internal resolution: {settings.internal_width}x{settings.internal_height}, Face det_size: {det_size}")

# ── Per-Camera Processor ───────────────────────────────────────────────────────

class CameraProcessor(threading.Thread):
    def __init__(self, cam_config):
        super().__init__(name=f"Proc-{cam_config.cam_id}", daemon=True)
        self.config = cam_config
        self.stop_event = threading.Event()
        
        # Motion detection state
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=settings.motion_history,
            varThreshold=50,
            detectShadows=False
        )
        self.motion_active = False
        self.warmup_count = 0
        self.no_motion_frames = 0
        
        # Performance tracking
        self.last_inference_time: float = 0.0
        self.frame_count = 0
        self.published_count = 0
        
    def run(self):
        log.info(f"[{self.config.cam_id}] Starting processor for {self.config.source}")
        cap = cv2.VideoCapture(self.config.source)
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log.warning(f"[{self.config.cam_id}] Lost feed, retrying...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(self.config.source)
                continue
            
            # Small sleep to yield CPU if running too fast (e.g. file source)
            time.sleep(0.01)
            
            self.frame_count += 1
            if self.frame_count % FRAME_SKIP != 0:
                continue
            
            # Keep original frame for high-res face cropping
            original_frame = frame.copy()
            orig_h, orig_w = original_frame.shape[:2]
            
            # Optimization: Resize frame early to save memory/CPU
            # Ultra-low resource: override to very small resolution
            width, height = settings.internal_width, settings.internal_height
            if settings.ultra_low_resource:
                width, height = 320, 180
                
            frame = cv2.resize(frame, (width, height))
            scale_x = orig_w / width
            scale_y = orig_h / height
                
            # 1. Motion Detection (Always run to keep BG model fresh)
            motion_score, bbox = self._detect_motion(frame)
            
            # 2. Logic to decide if we run AI
            now = time.time()
            time_since_last = now - self.last_inference_time
            
            should_infer = False
            if MOTION_GATED_INFERENCE:
                if self.motion_active and time_since_last >= (1.0 / MAX_INFERENCE_FPS):
                    should_infer = True
            else:
                if time_since_last >= (1.0 / MAX_INFERENCE_FPS):
                    should_infer = True
                    
            if should_infer:
                self.last_inference_time = now
                self._run_full_pipeline(frame, bbox, original_frame, scale_x, scale_y)

        cap.release()

    def _detect_motion(self, frame):
        # Resize for faster motion detection (even smaller than internal)
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        fg_mask = self.bg_subtractor.apply(gray)
        self.warmup_count += 1
        if self.warmup_count < 25:
            return 0, [0, 0, 0, 0]
            
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > (settings.motion_min_area / 4)] # Adjust for resize
        score = sum(cv2.contourArea(c) for c in valid)
        
        if valid:
            self.no_motion_frames = 0
            if not self.motion_active:
                self.motion_active = True
                log.info(f"[{self.config.cam_id}] Motion detected.")
            # Scale bbox back to original size if needed (not strictly needed for just motion flag)
            return score, [0,0,0,0]
        else:
            self.no_motion_frames += 1
            if self.no_motion_frames > 30 and self.motion_active:
                self.motion_active = False
                log.info(f"[{self.config.cam_id}] Motion cleared.")
            return 0, [0, 0, 0, 0]

    def _run_full_pipeline(self, frame, motion_bbox, original_frame, scale_x, scale_y):
        t0 = time.time()
        detected_objects = []
        
        # A. YOLO Inference
        with model_lock:
            results = yolo_model.track(
                frame, conf=settings.detection_confidence, classes=[0],
                tracker="bytetrack.yaml", persist=True, verbose=False
            )
        
        boxes = results[0].boxes if results and results[0].boxes is not None else []
        annotated = frame.copy()
        
        db = db_module.get_db()
        try:
            db_persons = db_module.get_all_persons(db)
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                
                # B. Face Recognition (Inside YOLO loop)
                pad = 10
                # Scale up to original frame size and crop
                orig_x1 = max(0, int(int(x1) * scale_x) - pad * int(scale_x))
                orig_y1 = max(0, int(int(y1) * scale_y) - pad * int(scale_y))
                orig_x2 = min(original_frame.shape[1], int(int(x2) * scale_x) + pad * int(scale_x))
                orig_y2 = min(original_frame.shape[0], int(int(y2) * scale_y) + pad * int(scale_y))
                
                crop = original_frame[orig_y1:orig_y2, orig_x1:orig_x2]
                
                embedding, face_conf, f_bbox = [], 0.0, []
                person_name, person_role, person_id = None, None, None
                status, message = None, None
                
                # B. Face Recognition (TIERED: Only run if a person is actually detected)
                # In ultra-low mode, we skip face recognition if confidence is too low or track is already identified
                cached = tracker.get_track_identity(self.config.cam_id, track_id)
                
                if cached:
                    person_id, person_name, person_role = cached
                    status = "AUTHORIZED"
                    message = f"Persistent: {person_name}"
                elif crop.size > 0 and confidence > 0.6:
                    with model_lock:
                        faces = face_app.get(crop)
                    if faces:
                        f = faces[0]
                        embedding = f.embedding.tolist()
                        face_conf = float(f.det_score)
                        
                        # Scale face bbox back down for annotation on the small frame
                        if f.bbox is not None:
                            fx1 = int((orig_x1 + f.bbox[0]) / scale_x)
                            fy1 = int((orig_y1 + f.bbox[1]) / scale_y)
                            fx2 = int((orig_x1 + f.bbox[2]) / scale_x)
                            fy2 = int((orig_y1 + f.bbox[3]) / scale_y)
                            f_bbox = [fx1, fy1, fx2, fy2]

                        # 1. Identity Check
                        match, score = tracker.identify_person(embedding, db_persons)
                        if match:
                            tracker.add_identity_vote(self.config.cam_id, track_id, match, score)
                            
                        # Use majority vote instead of instant match
                        final_match = tracker.get_majority_identity(self.config.cam_id, track_id, min_votes=3)
                        
                        if final_match:
                            person_name, person_role, person_id = final_match.name, final_match.role, final_match.id
                            tracker.link_track_to_person(self.config.cam_id, track_id, final_match)
                            status = "AUTHORIZED"
                            message = f"Identified: {person_name}"
                        else:
                            if match:
                                status = "UNKNOWN"
                                message = "Verifying..."
                
                # C. SOP Logic (Identity -> Zone -> Violation)
                person_name = person_name if person_name else "Unknown"
                person_role = person_role if person_role else "Unknown"
                status = status if status else "UNKNOWN"
                message = message if message else "Unidentifiable"
                
                # 2. Zone Check
                zone = zone_manager.find_zone_for_person(self.config.cam_id, [x1, y1, x2, y2])
                zone_name = zone.name if zone else "Default Zone"
                
                # D. Log to DB
                db_module.log_event(
                    db, timestamp=datetime.utcnow().isoformat(),
                    cam_id=self.config.cam_id, zone_name=zone_name,
                    person_id=person_id, person_name=person_name, person_role=person_role,
                    global_id=f"T_{track_id}", status=status, confidence=confidence,
                    message=message, frame_snapshot=None # Don't save full frames to SQLite to save disk/RAM
                )
                
                # Visualization (Internal only)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, f"{person_name} ({status})", (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        finally:
            db.close()
            
        ms = (time.time() - t0) * 1000
        if len(boxes) > 0:
            log.info(f"[{self.config.cam_id}] Pipeline: {len(boxes)} persons in {ms:.0f}ms")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=== STARTING UNIFIED EDGE ENGINE (OPTIMIZED) ===")
    
    # 1. Initialize DB
    db_module.init_db()
    
    # 2. Load Models Once
    load_models()
    
    # 3. Start Processors
    processors = []
    for cam_cfg in settings.cameras:
        p = CameraProcessor(cam_cfg)
        p.start()
        processors.append(p)
        
    # 4. Handle Exit
    def handle_sig(sig, frame):
        log.info("Shutdown signal received.")
        for p in processors:
            p.stop_event.set()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)
    
    log.info("System fully operational. Running in monolithic mode.")
    
    # 5. Heartbeat loop (just log stats to console)
    while True:
        cpu = psutil.cpu_percent()
        mem = psutil.Process().memory_info().rss / 1e6
        log.info(f"[STATUS] CPU: {cpu}% | RAM: {mem:.1f}MB")
        
        if settings.low_resources_mode:
            gc.collect()
            
        time.sleep(10)

if __name__ == "__main__":
    main()
