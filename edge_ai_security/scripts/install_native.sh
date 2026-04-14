#!/bin/bash
# install_native.sh — Setup a native environment for the Unified Edge Engine.

set -e

echo "Setting up native environment for Edge AI Security..."

# 1. Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# 2. Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install consolidated requirements
echo "Installing dependencies (this may take a few minutes)..."
pip install \
    pyzmq \
    python-dotenv \
    numpy \
    psutil \
    opencv-python-headless \
    onnxruntime \
    ultralytics \
    insightface \
    fastapi \
    uvicorn \
    sqlalchemy \
    shutil \
    "pydantic[email]"

# 5. Ensure models directory exists and has models
mkdir -p models
if [ ! -f "models/yolo11n.onnx" ]; then
    echo "YOLO model not found. Running downloader..."
    bash scripts/download_models.sh
fi

echo ""
echo "Setup complete!"
echo "To run the system:"
echo "  source venv/bin/activate"
echo "  export DATA_DIR=./data"
echo "  export MODELS_DIR=./models"
echo "  python3 unified_edge.py"
echo ""
