"""
Debug viewer — subscribes to detection_result on ZMQ port 5552
and pops up a live OpenCV window showing the annotated frames.

Run on the HOST (not in Docker):
  pip install pyzmq opencv-python numpy
  python3 scripts/debug_viewer.py
"""

import base64
import json
import sys

import cv2
import numpy as np
import zmq

ZMQ_PORT = 5552
TOPIC = b"detection_result"

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.setsockopt(zmq.RCVTIMEO, 2000)
sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
sock.connect(f"tcp://127.0.0.1:{ZMQ_PORT}")
print(f"Subscribed to tcp://127.0.0.1:{ZMQ_PORT} — waiting for frames...")
print("Press 'q' in the window to quit.")

while True:
    try:
        topic, data = sock.recv_multipart()
    except zmq.Again:
        print("No frames received in 2s (is ai_inference running and motion detected?)")
        # Still show blank window so user can quit
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for detections...",
                    (60, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 255), 2)
        cv2.imshow("AI Inference — Debug Viewer", blank)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        continue

    msg = json.loads(data.decode("utf-8"))

    # Decode annotated frame
    jpg_bytes = base64.b64decode(msg["frame_jpeg"])
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        continue

    # Overlay stats
    n = len(msg.get("detections", []))
    ms = msg.get("inference_ms", 0)
    cam = msg.get("cam_id", "?")
    overlay = f"cam={cam}  persons={n}  {ms:.0f}ms"
    cv2.putText(frame, overlay, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("AI Inference — Debug Viewer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sock.close()
ctx.term()
