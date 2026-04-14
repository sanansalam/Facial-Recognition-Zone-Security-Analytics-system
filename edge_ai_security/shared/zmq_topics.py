"""
ZeroMQ topic byte strings used by every service.
Always import from here. Never hardcode topic strings.

PUB/SUB topic matching in ZeroMQ is prefix-based.
All topics are bytes objects.
"""

# video_ingestion publishes on this topic
# motion_detection and ai_inference subscribe to it
RAW_FRAME = b"raw_frame"

# motion_detection publishes on these
# ai_inference subscribes to wake up / sleep
MOTION_DETECTED = b"motion_detected"
MOTION_CLEARED  = b"motion_cleared"

# ai_inference publishes on this
# sop_state_machine subscribes to it
DETECTION_RESULT = b"detection_result"

# sop_state_machine publishes on these
# event_logger subscribes to both
VIOLATION_EVENT  = b"violation_event"
SOP_STATE_UPDATE = b"sop_state_update"

# Every service publishes heartbeat every 5 seconds
# health_watchdog subscribes to all
HEARTBEAT = b"heartbeat"

# health_watchdog publishes this to restart a service
WATCHDOG_RESTART = b"watchdog_restart"
