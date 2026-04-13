from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json
import time

@dataclass
class FrameMessage:
    cam_id:     str
    cam_label:  str
    width:      int
    height:     int
    timestamp:  float = field(default_factory=time.time)
    sequence:   int   = 0

@dataclass
class Heartbeat:
    service_name: str
    status:       str
    cpu_percent:  float
    mem_mb:       float
    uptime_sec:   float
    details:      str = ""
    timestamp:    float = field(default_factory=time.time)

@dataclass
class MotionEvent:
    cam_id:       str
    motion_score: float
    bbox:         List[int] = field(default_factory=list)
    timestamp:    float = field(default_factory=time.time)

@dataclass
class MotionCleared:
    cam_id:    str
    timestamp: float = field(default_factory=time.time)

@dataclass
class Detection:
    id:          int      # Person ID from DB (-1 for unknown)
    name:        str      # Person Name
    role:        str      # Role (Staff, Customer, etc)
    bbox:        List[int] # [x1, y1, x2, y2]
    conf_person: float
    conf_face:   float

@dataclass
class DetectionResult:
    cam_id:      str
    cam_label:   str
    detections:   List[Detection]
    timestamp:    float = field(default_factory=time.time)

def encode(msg) -> bytes:
    return json.dumps(asdict(msg)).encode("utf-8")

def decode(raw: bytes, cls):
    data = json.loads(raw.decode("utf-8"))
    import dataclasses
    known = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)

@dataclass
class ViolationEvent:
    cam_id: str = ""
    cam_label: str = ""
    zone_id: int = -1
    zone_name: str = ""
    person_id: int = -1
    person_name: str = "Unknown"
    person_role: str = "Unknown"
    status: str = "UNKNOWN"
    severity: str = "LOW"
    message: str = ""
    violation_id: str = ""
    violation_type: str = ""
    evidence_jpeg: str = ""
    track_ids: List[int] = field(default_factory=list)
    position: List[int] = field(default_factory=list)
    bbox: List[int] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
