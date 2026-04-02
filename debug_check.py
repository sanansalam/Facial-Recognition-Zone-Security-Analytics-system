"""
Quick debug script to verify if video_ingestion is publishing frames.
Run this while video_ingestion is running!
"""
import zmq
import json
import base64
import cv2
import numpy as np

ZMQ_PORT = 5550
TOPIC = b"raw_frame"

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://localhost:{ZMQ_PORT}")
    sock.setsockopt(zmq.SUBSCRIBE, TOPIC)

    print(f"[*] Waiting for frames on port {ZMQ_PORT}...")
    
    try:
        while True:
            topic, data = sock.recv_multipart()
            msg = json.loads(data.decode("utf-8"))
            
            cam_id = msg.get("cam_id")
            seq = msg.get("sequence")
            ts = msg.get("timestamp")
            
            print(f"[OK] Received: {cam_id} | seq={seq} | ts={ts}")
            
            # Optional: Show the frame to prove it's real
            img_bytes = base64.b64decode(msg["frame_jpeg"])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imshow("Debug Check", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\n[*] Stopped.")
    finally:
        sock.close()
        ctx.term()

if __name__ == "__main__":
    main()
