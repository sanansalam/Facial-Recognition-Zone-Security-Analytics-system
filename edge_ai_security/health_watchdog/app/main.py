import signal
import time
import sys
import os

sys.path.insert(0, "/app")

from shared.config import get_settings, setup_logging

SERVICE_NAME = "health_watchdog"
DESCRIPTION  = "CPU/temp/service monitor"

settings = get_settings()
log      = setup_logging(SERVICE_NAME,
                         settings.log_level)

running = True

def shutdown(sig, frame):
    global running
    log.info("Shutdown signal received.")
    running = False

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

log.info(f"Starting — {DESCRIPTION}")
log.info(f"Hardware target : {settings.hardware_target}")
log.info(f"Cameras         : {len(settings.cameras)}")
log.info(f"ZMQ raw frames  : "
         f"port {settings.zmq.raw_frames}")
log.info("Stub mode — real logic added in Prompt 2+")

start_time = time.time()
tick = 0

while running:
    tick += 1
    uptime = time.time() - start_time
    log.info(
        f"Heartbeat #{tick} — "
        f"uptime {uptime:.0f}s — "
        f"waiting for Prompt 2 logic"
    )
    time.sleep(5)

log.info("Stopped cleanly.")
