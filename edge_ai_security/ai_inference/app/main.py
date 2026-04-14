"""
ai_inference — Multi-threaded Pipeline (A/B/C/D architecture)

Architecture:
  - Thread A (Collector): ZMQ SUB on 5551, drops frames if busy to avoid buffer bloat.
  - Thread B (Filter): YOLO11 Inference, outputs Person Crops to Thread C.
  - Thread C (Identifier): RetinaFace + ArcFace embedding generation.
  - Thread D (Publisher): ZMQ PUB on 5552, packages results and raw byte payload.
"""

import logging
import queue
import signal
import sys
import threading
import time
from typing import List

import cv2
import numpy as np
import psutil
import zmq

sys.path.insert(0, "/app")

from shared.config import get_settings
from shared.message_schema import (
    FrameMessage, DetectedObject, DetectionResult, Heartbeat,
    encode, decode,
)
from shared.zmq_topics import RAW_FRAME, DETECTION_RESULT, HEARTBEAT

# ── Settings & logging ─────────────────────────────────────────────────────────

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ai_inference")

stop_event = threading.Event()

# Pipeline Queues
# We keep capacities small so if downstream is blocked, upstream will drop frames
# to process the absolute latest frames.
queue_a_b = queue.Queue(maxsize=10)
queue_b_c = queue.Queue(maxsize=10)
queue_out = queue.Queue(maxsize=10)

cam_stats = {}  # for heartbeat
stats_lock = threading.Lock()

# ── Models ─────────────────────────────────────────────────────────────────────

yolo_model = None
face_app = None

def load_models():
    """Load models globally for threads B and C."""
    global yolo_model, face_app
    from ultralytics import YOLO
    from insightface.app import FaceAnalysis
    import os

    yolo_path = settings.yolo_model_path
    if not os.path.exists(yolo_path):
        log.error(f"YOLO model not found at {yolo_path}.")
        sys.exit(1)

    yolo_model = YOLO(yolo_path, task="detect")
    log.info(f"YOLO model loaded: {yolo_path}")

    face_app = FaceAnalysis(
        name="buffalo_sc",
        root="/models/insightface",
        providers=["CPUExecutionProvider"],
    )
    # Thread C usually runs on CPU with RetinaFace so setting ctx_id = -1
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    log.info("InsightFace loaded: buffalo_sc")


# ── Shutdown ───────────────────────────────────────────────────────────────────

def shutdown(sig, frame):
    log.info("Shutdown signal received")
    stop_event.set()

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ── Thread A: Collector ────────────────────────────────────────────────────────

def thread_a_collector(ctx: zmq.Context):
    """
    Role: ZeroMQ Subscriber.
    Action: Constantly listens for frames on port 5551.
    """
    log.info("Thread A (Collector) started.")
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVTIMEO, 200)
    socket.setsockopt(zmq.SUBSCRIBE, RAW_FRAME)
    socket.connect(f"tcp://video_ingestion:{settings.zmq.raw_frames}")

    while not stop_event.is_set():
        try:
            parts = socket.recv_multipart()
        except zmq.Again:
            continue
        except zmq.ZMQError as e:
            log.error(f"Collector zmq error: {e}")
            continue

        if len(parts) == 3:
            topic, data, frame_bytes = parts
        else:
            log.warning("Received frame missing raw bytes multipart.")
            continue

        try:
            msg = decode(data, FrameMessage)
        except Exception as e:
            log.warning(f"FrameMessage decode error: {e}")
            continue

        with stats_lock:
            if msg.cam_id not in cam_stats:
                cam_stats[msg.cam_id] = {"inferred": 0}

        try:
            queue_a_b.put_nowait((msg, frame_bytes))
        except queue.Full:
            # Drain oldest element to make room for newest (zero-wait)
            try:
                queue_a_b.get_nowait()
                queue_a_b.put_nowait((msg, frame_bytes))
            except (queue.Empty, queue.Full):
                pass

    log.info("Thread A stopped.")


# ── Thread B: Filter ───────────────────────────────────────────────────────────

def thread_b_filter():
    """
    Role: YOLO11 Inference.
    Action: Pulls latest frame, look for Person or Bag.
    If Person -> crop -> send to Thread C.
    """
    log.info("Thread B (Filter) started.")
    # Target Classes: 0 (Person), 24 (Backpack), 26 (Handbag), 28 (Suitcase)
    target_classes = [0, 24, 26, 28]

    while not stop_event.is_set():
        try:
            msg, frame_bytes = queue_a_b.get(timeout=1.0)
        except queue.Empty:
            continue

        t0 = time.time()
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        
        # We need a writable copy if we want to run Bytrack tracker and bounding boxes
        frame = frame.copy()

        try:
            results = yolo_model.track(
                frame,
                conf=settings.detection_confidence,
                classes=target_classes,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )
        except Exception as e:
            log.warning(f"[{msg.cam_id}] YOLO error: {e}")
            continue

        boxes = results[0].boxes if results and results[0].boxes is not None else []
        final_objects = []
        people_crops = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if cls_id == 0:
                cls_name = "person"
            elif cls_id == 24:
                cls_name = "backpack"
            elif cls_id == 26:
                cls_name = "handbag"
            elif cls_id == 28:
                cls_name = "suitcase"
            else:
                cls_name = "object"

            obj = DetectedObject(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                track_id=track_id
            )

            if cls_id == 0:  # Person -> prepare for Thread C
                pad = 20
                px1 = max(0, int(x1) - pad)
                py1 = max(0, int(y1) - pad)
                px2 = min(frame.shape[1], int(x2) + pad)
                py2 = min(frame.shape[0], int(y2) + pad)
                crop = frame[py1:py2, px1:px2]

                if crop.size > 0:
                    people_crops.append((obj, crop, (px1, py1)))
                else:
                    final_objects.append(obj)
            else:
                final_objects.append(obj)

        if len(people_crops) > 0:
            # Person found, pass to Thread C
            try:
                queue_b_c.put_nowait((msg, frame, people_crops, final_objects, t0))
            except queue.Full:
                try:
                    queue_b_c.get_nowait()
                    queue_b_c.put_nowait((msg, frame, people_crops, final_objects, t0))
                except (queue.Empty, queue.Full):
                    pass
        else:
            # No person found, immediately discard frame as requested:
            # "If No Person is found: It discards the frame immediately."
            pass

    log.info("Thread B stopped.")


# ── Thread C: Identifier ───────────────────────────────────────────────────────

def thread_c_identifier():
    """
    Role: RetinaFace + ArcFace.
    Action: Takes Person Crop, generated 512-D embedding. Heavy lifter.
    """
    log.info("Thread C (Identifier) started.")

    while not stop_event.is_set():
        try:
            msg, frame, crops, final_objects, t0 = queue_b_c.get(timeout=1.0)
        except queue.Empty:
            continue

        for obj, crop, offset in crops:
            px1, py1 = offset
            try:
                faces = face_app.get(crop)
                if faces:
                    f = faces[0]
                    obj.embedding = f.embedding.tolist()
                    obj.face_confidence = float(f.det_score)
                    
                    fx1 = int(f.bbox[0]) + px1
                    fy1 = int(f.bbox[1]) + py1
                    fx2 = int(f.bbox[2]) + px1
                    fy2 = int(f.bbox[3]) + py1
                    obj.face_bbox = [fx1, fy1, fx2, fy2]
            except Exception as e:
                log.warning(f"InsightFace error: {e}")
            
            final_objects.append(obj)

        # Pass combined output to Publisher
        try:
            queue_out.put_nowait((msg, frame, final_objects, t0))
        except queue.Full:
            try:
                queue_out.get_nowait()
                queue_out.put_nowait((msg, frame, final_objects, t0))
            except (queue.Empty, queue.Full):
                pass
            
    log.info("Thread C stopped.")


# ── Thread D: Publisher ────────────────────────────────────────────────────────

def thread_d_publisher(ctx: zmq.Context):
    """
    Role: ZeroMQ Publisher.
    Action: Collects results, packages to JSON DetectionResult, sends to 5552.
    """
    log.info("Thread D (Publisher) started.")
    
    socket = ctx.socket(zmq.PUB)
    socket.bind(f"tcp://0.0.0.0:{settings.zmq.detections}")
    log.info(f"ZMQ PUB bound on port {settings.zmq.detections}")

    while not stop_event.is_set():
        try:
            msg, frame, objects, t0 = queue_out.get(timeout=1.0)
        except queue.Empty:
            continue

        # Draw annotations
        annotated = frame.copy()
        for d in objects:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{d.class_name}:{d.track_id}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if d.face_bbox:
                fx1, fy1, fx2, fy2 = d.face_bbox
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
        
        annotated_bytes = annotated.tobytes()

        ms = (time.time() - t0) * 1000

        res = DetectionResult(
            cam_id=msg.cam_id,
            cam_label=msg.cam_label,
            detections=[{
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": d.bbox,
                "track_id": d.track_id,
                "embedding": d.embedding,
                "face_bbox": d.face_bbox,
                "face_confidence": d.face_confidence,
            } for d in objects],
            inference_ms=ms,
            timestamp=msg.timestamp,
            video_pos_ms=msg.video_pos_ms,
        )

        try:
            socket.send_multipart([
                DETECTION_RESULT,
                encode(res),
                annotated_bytes
            ])
            with stats_lock:
                cam_stats[msg.cam_id]["inferred"] = cam_stats[msg.cam_id].get("inferred", 0) + 1
        except Exception as e:
            log.error(f"Publisher socket error: {e}")

        # Reduce logging chatter to INFO to only log meaningful batches or optionally log every process
        log.info(f"[{msg.cam_id}] Processed {len(objects)} object(s) in {ms:.1f}ms")

    log.info("Thread D stopped.")


# ── Heartbeat ──────────────────────────────────────────────────────────────────

def heartbeat_loop(ctx: zmq.Context, start_time: float):
    health_socket = ctx.socket(zmq.PUB)
    health_socket.bind(f"tcp://0.0.0.0:{settings.zmq.health}")
    
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_models()
    log.info("Models ready.")

    ctx = zmq.Context()
    start_time = time.time()

    threads = [
        threading.Thread(target=thread_a_collector, args=(ctx,), name="Collector-ThreadA"),
        threading.Thread(target=thread_b_filter, name="Filter-ThreadB"),
        threading.Thread(target=thread_c_identifier, name="Identifier-ThreadC"),
        threading.Thread(target=thread_d_publisher, args=(ctx,), name="Publisher-ThreadD"),
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