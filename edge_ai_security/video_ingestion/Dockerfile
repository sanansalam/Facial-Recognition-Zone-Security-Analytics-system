FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

# Build context is project root (.)
COPY video_ingestion/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY video_ingestion/app/ ./app/

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

CMD ["python", "-u", "app/main.py"]
