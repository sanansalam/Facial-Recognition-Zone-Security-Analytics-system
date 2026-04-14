"""
Central configuration loader for all services.
Reads from environment variables.
Every service does:
    from shared.config import get_settings
    settings = get_settings()
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Union


def _str(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def _int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

def _float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


@dataclass
class CameraConfig:
    """Configuration for one camera source."""
    cam_id: str
    source: Union[int, str]   # int=USB index, str=RTSP URL
    label:  str


@dataclass
class ZMQPorts:
    """ZeroMQ port numbers for each topic bus."""
    raw_frames:  int
    motion:      int
    detections:  int
    violations:  int
    health:      int


@dataclass
class Settings:
    # Hardware
    hardware_target: str

    # Cameras
    cameras: List[CameraConfig]

    # ZeroMQ
    zmq: ZMQPorts

    # Inference
    yolo_model_path:      str
    inference_device:     str
    detection_confidence: float
    frame_skip:           int

    # Motion
    motion_min_area:  int
    motion_blur_size: int
    motion_history:   int

    # Storage
    data_dir:   str
    models_dir: str
    clips_dir:  str

    # Buffer
    buffer_seconds: int

    # Optimization
    internal_width:     int
    internal_height:    int
    low_resources_mode: bool
    ultra_low_resource: bool

    # Logging
    log_level: str


def _parse_cameras() -> List[CameraConfig]:
    """
    Parse CAM_N_SOURCE / CAM_N_LABEL from environment.
    Supports unlimited cameras — stops when CAM_N_SOURCE
    is not found.
    """
    cameras = []
    i = 0
    while True:
        src = os.environ.get(f"CAM_{i}_SOURCE")
        if src is None:
            break
        label = os.environ.get(
            f"CAM_{i}_LABEL", f"Camera {i}")
        # Convert to int if it is a plain number
        try:
            src = int(src)
        except ValueError:
            pass   # keep as RTSP string
        cameras.append(CameraConfig(
            cam_id=f"cam_{i}",
            source=src,
            label=label
        ))
        i += 1
    # If no cameras defined, add a default dev webcam
    if not cameras:
        cameras.append(CameraConfig(
            cam_id="cam_0",
            source=0,
            label="Dev Camera"
        ))
    return cameras


def get_settings() -> Settings:
    """
    Build and return a Settings object from environment.
    Call once at service startup.
    """
    return Settings(
        hardware_target=_str(
            "HARDWARE_TARGET", "cpu_only"),

        cameras=_parse_cameras(),

        zmq=ZMQPorts(
            raw_frames=_int("ZMQ_RAW_FRAMES_PORT", 5550),
            motion=_int("ZMQ_MOTION_PORT",         5551),
            detections=_int("ZMQ_DETECTIONS_PORT", 5552),
            violations=_int("ZMQ_VIOLATIONS_PORT", 5553),
            health=_int("ZMQ_HEALTH_PORT",         5554),
        ),

        yolo_model_path=_str(
            "YOLO_MODEL_PATH", "/models/yolo11n.onnx"),
        inference_device=_str("INFERENCE_DEVICE", "cpu"),
        detection_confidence=_float(
            "DETECTION_CONFIDENCE", 0.5),
        frame_skip=_int("FRAME_SKIP", 3),

        motion_min_area=_int("MOTION_MIN_AREA",   500),
        motion_blur_size=_int("MOTION_BLUR_SIZE", 21),
        motion_history=_int("MOTION_HISTORY",     500),

        data_dir=_str("DATA_DIR",     "/data"),
        models_dir=_str("MODELS_DIR", "/models"),
        clips_dir=_str("CLIPS_DIR",   "/data/clips"),

        buffer_seconds=_int("BUFFER_SECONDS", 60),

        internal_width=_int("INTERNAL_WIDTH", 640),
        internal_height=_int("INTERNAL_HEIGHT", 360),
        low_resources_mode=_str("LOW_RESOURCES_MODE", "true").lower() == "true",
        ultra_low_resource=_str("ULTRA_LOW_RESOURCE", "false").lower() == "true",

        log_level=_str("LOG_LEVEL", "INFO"),
    )


def setup_logging(service_name: str,
                  level: str = "INFO"):
    """
    Configure logging for a service.
    Call at the top of every service main.py.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=(f"[%(asctime)s] [{service_name}]"
                f" %(levelname)s — %(message)s"),
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(service_name)
