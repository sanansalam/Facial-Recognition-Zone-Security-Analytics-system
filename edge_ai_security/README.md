# Edge AI CCTV Security System

## Hardware
Primary  : Raspberry Pi 5 + Hailo-8 M.2
Secondary: NVIDIA Jetson Orin Nano
Dev      : Ubuntu laptop, CPU only

## Services
| Service            | Port | Role                        |
|--------------------|------|-----------------------------|
| video_ingestion    | —    | RTSP/USB frame reader       |
| motion_detection   | —    | Background subtraction gate |
| ai_inference       | —    | YOLO11 on NPU/GPU           |
| sop_state_machine  | 8000 | Zone sequence logic         |
| event_logger       | —    | SQLite + video buffer       |
| health_watchdog    | —    | CPU/temp/restart monitor    |

## Communication
ZeroMQ PUB/SUB on internal Docker network (ainet).
See shared/zmq_topics.py for all topic names.
See shared/message_schema.py for all message formats.

## Quick start (dev — laptop)
  cp .env.example .env
  bash scripts/download_models.sh
  docker compose -f docker-compose.yml \
                 -f docker-compose.dev.yml up --build

## Build order
  Prompt 1 — Scaffold (this file) ← YOU ARE HERE
  Prompt 2 — Video ingestion service
  Prompt 3 — Motion detection service
  Prompt 4 — AI inference service
  Prompt 5 — SOP state machine service
  Prompt 6 — Event logger + media buffer
  Prompt 7 — Health watchdog service
  Prompt 8 — Hailo-8 swap + Pi 5 deployment
