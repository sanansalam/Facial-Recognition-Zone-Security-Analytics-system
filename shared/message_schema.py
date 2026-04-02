from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json
import time

@dataclass
class FrameMessage:
    cam_id:     str
    cam_label:  str
    frame_jpeg: str       # base64 encoded JPEG string
    width:      int
    height:     int
    timestamp:  float = field(default_factory=time.time)
    sequence:   int   = 0

@dataclass
class Heartbeat:
    service_name: str
    status:       str     # healthy | degraded | error
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
