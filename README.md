# 🔐 Edge AI Security System

A modular, real-time AI-powered CCTV security system that runs entirely on local hardware — no cloud required.

## Features

- 👁️ **Real-time person detection** using YOLOv11
- 🧠 **Face recognition** using InsightFace (ArcFace)
- 🗺️ **Zone-based access control** — define restricted areas per camera
- 🎬 **Video evidence capture** — 10s rolling clip saved on violation
- 🖥️ **Web enrollment dashboard** — add persons via webcam or photo upload
- 📄 **Auto security reports** generated on shutdown
- 🔔 **Instant desktop alerts** via `notify-send`

---

## Architecture

```
video_ingestion  →  motion_detection  →  ai_inference  →  sop_state_machine  →  event_logger
     (ZMQ PUB)          (ZMQ PUB)          (ZMQ PUB)          (ZMQ PUB)
```

All services communicate via **ZeroMQ** (no Docker required for standalone mode).

---

## Quick Start

### 1. Install dependencies

```bash
pip install ultralytics insightface pyzmq psutil opencv-python-headless python-dotenv gradio onnxruntime
```

### 2. Configure cameras

Edit `.env` to point to your video files or live cameras:

```env
CAM_0_SOURCE=/path/to/video.avi
CAM_0_LABEL=Cam_0

CAM_1_SOURCE=0          # webcam index for live camera
CAM_1_LABEL=Front Door
```

### 3. Enroll persons (first time)

```bash
python3 enrollment_dashboard.py
```
Open **http://localhost:7860** → capture face via webcam or upload a photo → assign name, role, and zones.

### 4. Draw zones (first time)

```bash
python3 draw_zones.py
```
Click to define polygons for restricted areas on each camera frame.

### 5. Run the system

```bash
python3 run_all.py
```

Press `Ctrl+C` to stop. A security report is automatically saved to `data/reports/`.

### 6. Reset between sessions

```bash
python3 system_reset.py
```
Clears old events and evidence. **Enrolled persons are preserved.**

---

## Project Structure

```
video_ingestion_standalone/
├── video_ingestion/        # Reads cameras, publishes frames
├── motion_detection/       # Detects motion, gates AI pipeline
├── ai_inference/           # YOLO + InsightFace, produces detections
├── sop_state_machine/      # Zone access logic, generates violations
├── event_logger/           # Logs events, saves evidence clips, reports
├── shared/                 # Config, ZMQ topics, message schemas
├── data/
│   ├── evidence/           # Violation video clips & snapshots
│   └── reports/            # End-of-session security reports
├── enrollment_dashboard.py # Web UI for person enrollment (Gradio)
├── draw_zones.py           # Tool to draw zone polygons on camera frames
├── system_reset.py         # Clears events/evidence, keeps enrollment
├── run_all.py              # Launches all 5 services
├── docker-compose.yml      # Optional Docker deployment
└── .env                    # Camera and port configuration
```

---

## Docker (Optional)

For server deployment or reproducible environments:

```bash
docker-compose up --build
```

---

## Roles & Zone Access

Roles available: `Manager`, `Staff`, `Cashier`, `Security`, `Cleaner`, `Customer`, `Visitor`, `Unknown`

Each zone can have a list of **restricted roles** — any person with that role entering the zone triggers a `RESTRICTED` violation.

---

## Evidence

- **Known persons** → `.mp4` video clip (10s pre-violation)
- **Unknown persons** → `.jpg` snapshot

Evidence is stored in `data/evidence/` with filename format:
```
cam_0_20260413_112647_remya.mp4
```

---

## Requirements

| Component | Minimum |
|-----------|---------|
| RAM | 4 GB (8 GB recommended) |
| CPU | 4 cores |
| OS | Ubuntu 20.04+ |
| Python | 3.10+ |
| Models | InsightFace `buffalo_sc`, YOLOv11n |

> **Note:** Model files (`*.pt`, `*.onnx`) and CCTV videos are excluded from this repository due to size. Place them in the paths configured in `.env`.
