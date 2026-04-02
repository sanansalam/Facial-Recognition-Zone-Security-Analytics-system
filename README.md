# Facial Recognition & Zone Security (V2)

Version 2 adds **Motion Detection** to the video ingestion pipeline.

## Features
- **Video Ingestion (V1)**: Multi-camera capture and ZMQ publishing.
- **Motion Detection (V2)**: MOG2-based background subtraction per camera.
- **Microservices Architecture**: Services communicate over ZeroMQ and run in Docker.

## Setup
1. Copy `.env.example` to `.env`.
2. Edit `.env` with your camera sources.
3. Run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

## Verification (Is it working?)
To verify V2:
1. **Run the services** using `docker-compose up`.
2. **Run the Debug Script**:
   ```bash
   python3 debug_check.py
   ```
   You should see the video window. If there is movement, a **RED BOX** will appear around the motion area.

## Git Versioning
This project is built in stages:
- **v1**: Video Ingestion only.
- **v2**: Video Ingestion + Motion Detection.
