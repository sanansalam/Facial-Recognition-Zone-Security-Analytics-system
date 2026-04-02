FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 libxext6 curl && rm -rf /var/lib/apt/lists/*
COPY motion_detection/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ ./shared/
COPY motion_detection/app/ ./app/
CMD ["python", "-u", "app/main.py"]
