from dataclasses import dataclass
from typing import List, Union
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class CameraConfig:
    cam_id: str
    source: Union[int, str]
    label:  str

@dataclass
class ZMQPorts:
    raw_frames:       int
    motion:           int
    inference:        int
    sop:              int
    health_ingestion: int
    health_motion:    int
    health_inference: int
    health_sop:       int

@dataclass
class Settings:
    cameras: List[CameraConfig]
    zmq: ZMQPorts
    motion_min_area:  int
    motion_blur_size: int
    motion_history:   int
    log_level: str

def get_settings() -> Settings:
    cameras = []
    i = 0
    while True:
        src = os.environ.get(f"CAM_{i}_SOURCE")
        if src is None: break
        label = os.environ.get(f"CAM_{i}_LABEL", f"Camera {i}")
        try: src = int(src)
        except ValueError: pass
        cameras.append(CameraConfig(cam_id=f"cam_{i}", source=src, label=label))
        i += 1
    if not cameras: cameras.append(CameraConfig(cam_id="cam_0", source=0, label="Dev Camera"))

    return Settings(
        cameras=cameras,
        zmq=ZMQPorts(
            raw_frames=int(os.environ.get("ZMQ_RAW_FRAMES_PORT", 5550)),
            motion=int(os.environ.get("ZMQ_MOTION_PORT", 5551)),
            inference=int(os.environ.get("ZMQ_INFERENCE_PORT", 5552)),
            sop=int(os.environ.get("ZMQ_VIOLATIONS_PORT", 5553)),
            health_ingestion=int(os.environ.get("ZMQ_HEALTH_INGESTION", 5554)),
            health_motion=int(os.environ.get("ZMQ_HEALTH_MOTION", 5555)),
            health_inference=int(os.environ.get("ZMQ_HEALTH_INFERENCE", 5556)),
            health_sop=int(os.environ.get("ZMQ_HEALTH_SOP", 5557)),
        ),
        motion_min_area=int(os.environ.get("MOTION_MIN_AREA", 500)),
        motion_blur_size=int(os.environ.get("MOTION_BLUR_SIZE", 21)),
        motion_history=int(os.environ.get("MOTION_HISTORY", 500)),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
