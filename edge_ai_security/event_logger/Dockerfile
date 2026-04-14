FROM python:3.11-slim

WORKDIR /app

# System libraries needed by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY event_logger/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module (available to all services)
COPY shared/ ./shared/

# Copy this service's application code
COPY event_logger/app/ ./app/

# Healthcheck — each service exposes a /health
# endpoint in later prompts. For now just check process.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

CMD ["python", "-u", "app/main.py"]
