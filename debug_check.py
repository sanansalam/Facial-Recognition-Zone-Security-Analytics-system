"""
Debug script for V2: Video Ingestion + Motion Detection.
Subscribes to BOTH raw frames and motion events.
"""
import zmq
import json
import base64
import cv2
import numpy as np

# Ports from .env.example
FRAME_PORT = 5550
MOTION_PORT = 5551

RAW_FRAME = b"raw_frame"
MOTION_DETECTED = b"motion_detected"
MOTION_CLEARED = b"motion_cleared"

def main():
    ctx = zmq.Context()
    
    # Subscriber for frames
    frame_sock = ctx.socket(zmq.SUB)
    frame_sock.connect(f"tcp://localhost:{FRAME_PORT}")
    frame_sock.setsockopt(zmq.SUBSCRIBE, RAW_FRAME)
    
    # Subscriber for motion
    motion_sock = ctx.socket(zmq.SUB)
    motion_sock.connect(f"tcp://localhost:{MOTION_PORT}")
    motion_sock.setsockopt(zmq.SUBSCRIBE, MOTION_DETECTED)
    motion_sock.setsockopt(zmq.SUBSCRIBE, MOTION_CLEARED)

    print(f"[*] V2 Debug: Waiting for frames ({FRAME_PORT}) and motion ({MOTION_PORT})...")
    
    last_motion = {} # cam_id -> bbox

    try:
        while True:
            # Check for motion events (non-blocking)
            try:
                topic, data = motion_sock.recv_multipart(zmq.NOBLOCK)
                msg = json.loads(data.decode("utf-8"))
                cam_id = msg.get("cam_id")
                if topic == MOTION_DETECTED:
                    print(f"[MOTION] {cam_id} DETECTED! score={msg.get('motion_score'):.0f}")
                    last_motion[cam_id] = msg.get("bbox")
                elif topic == MOTION_CLEARED:
                    print(f"[MOTION] {cam_id} CLEARED")
                    last_motion.pop(cam_id, None)
            except zmq.Again:
                pass

            # Check for frames (blocking with timeout)
            try:
                if frame_sock.poll(100):
                    topic, data = frame_sock.recv_multipart()
                    msg = json.loads(data.decode("utf-8"))
                    cam_id = msg.get("cam_id")
                    
                    img_bytes = base64.b64decode(msg["frame_jpeg"])
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        # Draw motion bbox if active
                        if cam_id in last_motion:
                            x, y, w, h = last_motion[cam_id]
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(img, "MOTION", (x, y - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        cv2.imshow(f"V2 Debug: {cam_id}", img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            except zmq.Again:
                continue

    except KeyboardInterrupt:
        print("\n[*] Stopped.")
    finally:
        frame_sock.close()
        motion_sock.close()
        ctx.term()

if __name__ == "__main__":
    main()
