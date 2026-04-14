import cv2
import numpy as np
import onnxruntime as ort
from enrollment_tool import FaceProcessor, MODEL_DIR, INPUT_SIZE

p = FaceProcessor(MODEL_DIR)
frame = cv2.imread('test_frame.jpg')

faces = p.detect_faces(frame)
print(f"Class method found {len(faces)} boxes via detect_faces()")
for f in faces:
    print(f)
