"""
All ZeroMQ message payloads as Python dataclasses.
Serialized as JSON bytes over the wire.

Usage:
    # Sending
    msg = FrameMessage(cam_id="cam0", ...)
    socket.send_multipart([RAW_FRAME, encode(msg)])

    # Receiving
    topic, data = socket.recv_multipart()
    msg = decode(data, FrameMessage)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json
import time


# ── Video ingestion → everyone ────────────────────────

@dataclass
class FrameMessage:
    """
    One video frame from one camera.
    frame_jpeg is base64-encoded JPEG bytes.
    Sent on RAW_FRAME topic.
    """
    cam_id:     str
    cam_label:  str
    width:      int
    height:     int
    timestamp:  float = field(default_factory=time.time)
    video_pos_ms: float = 0.0
    sequence:   int   = 0


# ── Motion detection → AI inference ──────────────────

@dataclass
class MotionEvent:
    """
    Fired when background subtraction detects movement.
    Sent on MOTION_DETECTED topic.
    """
    cam_id:       str
    motion_score: float       # contour area in pixels
    bbox:         List[int]   # [x, y, w, h]
    timestamp:    float = field(default_factory=time.time)


@dataclass
class MotionCleared:
    """
    Fired when motion stops for N consecutive frames.
    Sent on MOTION_CLEARED topic.
    """
    cam_id:    str
    timestamp: float = field(default_factory=time.time)


# ── AI inference → SOP state machine ─────────────────

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class DetectedObject:
    """
    One detected object inside a DetectionResult.
    class_name examples: person, face, bag
    """
    class_id:         int
    class_name:       str
    confidence:       float
    bbox:             List[int]        # [x1, y1, x2, y2]
    track_id:         Optional[int] = None
    embedding:        List[float] = field(default_factory=list)  # 512-dim ArcFace vector
    face_bbox:        List[float] = field(default_factory=list)  # [x1, y1, x2, y2] in crop
    face_confidence:  float = 0.0


@dataclass
class DetectionResult:
    """
    All objects detected in one frame.
    Sent on DETECTION_RESULT topic.
    frame_jpeg is the annotated frame (base64 JPEG).
    """
    cam_id:        str
    cam_label:     str
    detections:    List[dict]   # list of DetectedObject dicts
    inference_ms:  float
    timestamp:     float = field(default_factory=time.time)
    video_pos_ms:  float = 0.0


# ── SOP state machine → event logger ─────────────────

@dataclass
class ViolationEvent:
    """
    A confirmed SOP violation.
    Sent on VIOLATION_EVENT topic.
    evidence_jpeg is the frame that triggered it.
    """
    violation_id:   str         # uuid4
    violation_type: str         # e.g. UNAUTHORIZED_ACCESS
    cam_id:         str
    zone_id:        str
    description:    str
    severity:       str         # LOW|MEDIUM|HIGH|CRITICAL
    track_ids:      List[int]   # person track IDs involved
    timestamp:      float = field(default_factory=time.time)


@dataclass
class SOPStateUpdate:
    """
    State machine emits this on every state transition.
    Sent on SOP_STATE_UPDATE topic.
    """
    zone_id:        str
    previous_state: str
    current_state:  str
    trigger:        str         # what caused the transition
    timestamp:      float = field(default_factory=time.time)


# ── Heartbeat — every service → health watchdog ───────

@dataclass
class Heartbeat:
    """
    Published by every service every 5 seconds.
    health_watchdog alerts if a service goes silent.
    """
    service_name: str
    status:       str     # healthy | degraded | error
    cpu_percent:  float
    mem_mb:       float
    uptime_sec:   float
    details:      str = ""
    timestamp:    float = field(default_factory=time.time)


# ── Serialization helpers ─────────────────────────────

def encode(msg) -> bytes:
    """Serialize a message dataclass to JSON bytes."""
    return json.dumps(asdict(msg)).encode("utf-8")


def decode(raw: bytes, cls):
    """
    Deserialize JSON bytes into a message dataclass.
    Example: msg = decode(raw, FrameMessage)
    """
    data = json.loads(raw.decode("utf-8"))
    # Filter to only known fields of the dataclass
    import dataclasses
    known = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items()
                if k in known}
    return cls(**filtered)
