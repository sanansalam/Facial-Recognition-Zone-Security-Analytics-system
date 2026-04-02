# Facial Recognition & Zone Security Analytics System

This repository contains the progressive stages of the Edge AI Security system, organized into versioned folders.

## Folder Structure

### [v1/](./v1) - Video Ingestion
- Core service for camera discovery and frame streaming.
- Standalone service with basic ZeroMQ publishing.

### [v2/](./v2) - Video Ingestion + Motion Detection
- Adds a background subtraction service.
- Includes a multi-service `docker-compose.yml` for unified deployment.

## How to use
Each folder is a standalone version of the project at that stage. Navigate into a folder to see its specific setup instructions and code.

```bash
# Example: Running V2
cd v2
docker-compose up --build
```
