import cv2
from enrollment_tool import FaceProcessor, MODEL_DIR

p = FaceProcessor(MODEL_DIR)
cap = cv2.VideoCapture('data/videos/219_8_Jewellery_IPC6_aef7458404ad4b2c89a99ec41882bdca_20260214182622.avi')
found_faces = 0

for i in range(100):
    ret, frame = cap.read()
    if not ret:
        break
    faces = p.detect_faces(frame)
    found_faces += len(faces)

print(f"Test complete. Total faces found in first 100 frames: {found_faces}")
