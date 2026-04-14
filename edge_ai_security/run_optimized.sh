#!/bin/bash

# Optimized Startup Script for Edge AI Security System
# Designed for 8GB RAM and low-end CPU machines.

# Determine project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "=== STARTING OPTIMIZED EDGE ENGINE ==="

# Check for dependencies
python3 -c "import sqlalchemy, cv2, ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing dependencies. Please run:"
    echo "pip install -r edge_ai_security/requirements.txt"
    exit 1
fi

# Force optimized environment variables (overrides .env if needed)
export FRAME_SKIP=5
export MAX_INFERENCE_FPS=1.5
export INTERNAL_WIDTH=640
export INTERNAL_HEIGHT=360
export LOW_RESOURCES_MODE=true
export INFERENCE_DEVICE=cpu
export LOG_LEVEL=INFO

# Ensure models and data directories exist
mkdir -p ./models ./data/clips

# Run the unified engine
# Using 'python3' explicitly
python3 unified_edge.py
