#!/bin/bash

# Ultra-Low Resource Startup Script for Edge AI Security System
# Designed for "Simple Local Machines" (2GB - 4GB RAM).

# Determine project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "=== STARTING ULTRA-LOW RESOURCE EDGE ENGINE ==="

# Check for dependencies
python3 -c "import sqlalchemy, cv2, ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing dependencies. Please run:"
    echo "pip install -r edge_ai_security/requirements.txt"
    exit 1
fi

# Force extreme optimization environment variables
export ULTRA_LOW_RESOURCE=true
export LOW_RESOURCES_MODE=true
export FRAME_SKIP=10
export MAX_INFERENCE_FPS=1.0
export INTERNAL_WIDTH=480
export INTERNAL_HEIGHT=270
export INFERENCE_DEVICE=cpu
export LOG_LEVEL=INFO

# Ensure models and data directories exist
mkdir -p ./models ./data/clips

# Run the unified engine
python3 unified_edge.py
