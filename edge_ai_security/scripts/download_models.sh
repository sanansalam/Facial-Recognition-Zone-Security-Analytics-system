#!/bin/bash
# Run ONCE before starting the system.
# Downloads YOLO11n ONNX model into ./models/
# This is the ONLY script allowed to use the internet.
# After this runs the system is fully offline.

set -e

echo "======================================"
echo " Model Downloader — run once only"
echo "======================================"
echo ""

mkdir -p ./models

echo "Downloading YOLO11n and exporting to ONNX..."
pip install ultralytics --quiet

python3 - <<'EOF'
from ultralytics import YOLO
import shutil, os

print("Loading YOLO11n...")
model = YOLO("yolo11n.pt")

print("Exporting to ONNX form
at...")
model.export(format="onnx", imgsz=640, opset=17)

src = "yolo11n.onnx"
dst = "./models/yolo11n.onnx"
shutil.move(src, dst)
size_mb = os.path.getsize(dst) / 1e6
print(f"Saved: {dst}  ({size_mb:.1f} MB)")
EOF

echo ""
echo "Model ready. System is now fully offline."
echo ""
echo "Next step:"
echo "  docker compose -f docker-compose.yml \\"
echo "                 -f docker-compose.dev.yml \\"
echo "                 up --build"
