from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json
import time

@dataclass
class FrameMessage:
    cam_id:     str
    cam_label:  str
    frame_jpeg: str
    width:      int
    height:     int
    timestamp:  float = field(default_factory=time.time)
    sequence:   int   = 0

@dataclass
class MotionEvent:
    cam_id:       str
    motion_score: float
    bbox:         List[int]
    timestamp:    float = field(default_factory=time.time)

@dataclass
class MotionCleared:
    cam_id:    str
    timestamp: float = field(default_factory=time.time)

@dataclass
class Heartbeat:
    service_name: str
    status:       str
    cpu_percent:  float
    mem_mb:       float
    uptime_sec:   float
    details:      str = ""
    timestamp:    float = field(default_factory=time.time)

def encode(msg) -> bytes:
    return json.dumps(asdict(msg)).encode("utf-8")

def decode(raw: bytes, cls):
    data = json.loads(raw.decode("utf-8"))
    import dataclasses
    known = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)
